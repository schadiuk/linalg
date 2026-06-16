# Cholesky factorisation: reference
This document covers the implementation of Cholesky decomposition algorithm.
> **Source file:** `cholesky.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundations](#1-mathematical-foundations)
- [Storage layout](#2-storage-layout)
- [Algorithm](#3-potrf-the-blocked-algorithm)
- [Public API](#4-public-api)
---
## 0. Preamble and notation

Matrices are $n \times n$ with entries in $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. Elements are denoted either as $A_{ij}$ or as $A(i,j)$ using **0-based indexing throughout**, matching standard C++ array conventions.

The following quantities appear in the analysis below.

| Symbol | Meaning |
|---|---|
| $\kappa(A)$ | Condition number of the matrix $\|A\|_2 \|A^{-1}\|_2$. |
| $\varepsilon$ | Machine epsilon ($\approx 2.2 \times 10^{-16}$ for `double`). |
| $k_b$ | Block width; equal to `CHOL_BLOCK` in the outer loop. |

**Result type.** After factorisation, `potrf` returns a `CholeskyResult<T,L>` structure containing the triangular factor and a `char` tag indicating which triangle was used. [Unlike `LUResult`](/reference/LU.md#2-packed-storage-layout), the factor matrix has the same dimensions as the input and the opposite triangle is exactly zero.

**Naming.** Rather than using the *(somewhat lengthy)* surname-based naming, the public API (`potrf`, `potrs`, `potri`) occasionally uses the [LAPACK convention](https://www.netlib.org/lapack/explore-html/d2/d09/group__potrf.html) for positive-definite triangular routines.

---
## 1. Mathematical foundations

### 1.1 The decomposition

Cholesky decomposition process factorises a Hermitian ($A = A^H$) positive-definite (HPD) matrix $A \in \mathbb{F}^{n \times n}$ as:
$$A = L L^H$$

where $L$ is lower-triangular with strictly positive real diagonal entries. An equivalent upper form $A = U^H U$ with $U = L^H$ is also supported. The decomposition exists and is **unique** among all such factorisations with positive diagonal; proof of existence is constructive and is discussed below.

**The algorithm** processes one column at a time. At column $j$, it computes:
$$d_j = A_{jj} -\sum_{p=0}^{j-1} |L_{jp}| ^2 $$

The HPD assumption guarantees that $d_j > 0$ at every step. Setting $L_{jj} = \sqrt{d_j}$ and:
$$L_{ij} = \frac{A_{ij} - \displaystyle\sum_{p=0}^{j-1} L_{ip}\, \overline{L_{jp}}}{L_{jj}}, \quad i > j$$

gives a valid lower-triangular factor. This determines $L$ completely column by column, so the algorithm terminates without any other prerequisites.

**Uniqueness.** Suppose $A = L_1 L_1^H = L_2 L_2^H$ with both $L_1, L_2$ lower-triangular with positive diagonals. Then:
$$L_2^{-1} L_1 = L_2^H (L_1^H)^{-1}$$

The left side is lower-triangular and the right side is upper-triangular. A matrix that is simultaneously lower- and upper-triangular is diagonal; call it $D$. Then $L_1 = L_2 D$ and $A = L_2 D D^H L_2^H$. Since $A = L_2 L_2^H$ we get $D D^H = I$. A real diagonal matrix satisfying $D^2 = I$ has entries $\pm 1$, and since both $L_1$ and $L_2$ have positive diagonals, $D = I$. Hence $L_1 = L_2$.

The argument for the upper form is identical by symmetry.

### 1.2 Schur complement and blocking

The blocked algorithm's correctness rests on the same Schur complement identity used in [blocked LU](/reference/LU.md#14-blocked-lu-schur-complement). After processing the first $k_b$ columns, the working matrix decomposes as:
$$A = \begin{bmatrix} L_{11} & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} L_{11}^H & L_{21}^H \\ 0 & S \end{bmatrix}$$

where $L_{11}$ is $k_b \times k_b$ lower-triangular and the Schur complement is: $S = A_{22} - L_{21} L_{21}^H$.

$S$ is itself HPD (as the Schur complement of a positive-definite submatrix), so the factorisation can be applied recursively to $S$. This induction over block columns proves that the blocked algorithm computes the same result as the column-by-column algorithm, differing only in the order of arithmetic operations.

### 1.3 Comparison with LU decomposition

LU with partial pivoting requires $\tfrac{2}{3}n^3$ arithmetic operations. Cholesky requires $\tfrac{1}{3}n^3$ - exactly half - because the HPD structure means that factoring the lower triangle fully determines the upper, so only one triangle of work is ever performed. Storage is similarly halved: the factor occupies one triangle of the result matrix with the other zeroed.

**Stability.** Cholesky without pivoting is unconditionally backward stable for HPD matrices. The growth factor - the ratio of the largest element generated during factorisation to the largest element of $A$ - is bounded by 1:
$$\max_{ij} |L_{ij}| \leq \max_{ij} |A_{ij}|$$

This is a stronger guarantee than LU with partial pivoting, whose growth factor is bounded by $2^{n-1}$ in the worst case. In practice, Cholesky applied to HPD matrices produces residuals at the level of $\varepsilon \cdot \kappa(A)$, proportional to the condition number and not amplified by the factorisation itself.

---
## 2. Storage layout
After `potrf` returns, the `factor` field of `CholeskyResult` holds:
$$\text{factor}[i,j] = \begin{cases} L_{ij} & i \geq j \\ 0 & i < j \end{cases}$$

for `uplo == 'L'`, and correspondingly:
$$\text{factor}[i,j] = \begin{cases} U_{ij} & i \leq j \\ 0 & i > j \end{cases}$$

for `uplo == 'U'`.

### Input handling
`potrf` copies only the relevant triangle of the input $A$ into a working matrix $W$ before factorisation. The copy is parallelised over rows via `parallel_for`. The opposite triangle of $W$ is explicitly set to to zero in the `Matrix<T,L>(n, n, T(0))` constructor, so it is never read from $A$. After the kernel returns, `zero_off_triangle` makes a further explicit zeroing pass (also parallelised) to overwrite any incidental writes from the blocked steps, guaranteeing the invariant above in the returned `factor`.

---
## 3. `potrf`: the blocked algorithm

### 3.1 Block structure

An unblocked column-by-column Cholesky performs $O(n^3/3)$ arithmetic operations, but each rank-1 trailing update at step $j$ reads and writes the full trailing submatrix - a bandwidth-bound operation. Blocking reorganises the work into three phases per block column. The block width is set by `L2_BLOCK = 128`. At outer step $k$, let $k_b = \min(\text{L2\_BLOCK},\, n - k)$ and partition the working matrix as:
$$A^{(k)} = \begin{bmatrix} A_{11} & \cdots \\ A_{21} & A_{22} \end{bmatrix}$$

where the relevant blocks have the dimensions summarised below:

| Block | Rows | Columns | Description |
|---|---|---|---|
| $A_{11}$ | $k$ | $k$ | Already factored; holds $L_{11}$ (lower) and zeros (upper). |
| $A_{21}$ | $k_b$ | $k_b$ | Current panel diagonal block. |
| $A_{31}$ | $n - k - k_b$ | $k_b$ | Panel below diagonal. |
| $A_{33}$ | $n - k - k_b$ | $n - k - k_b$ | Trailing submatrix. |

The algorithm per block column is as follows:

**Step 1: unblocked panel factorisation.** `chol_unblocked_lower(A, k, kb)` function factors the $k_b \times k_b$ diagonal block in-place, producing $L_{11}$. After this step, $A_{21}$ holds $L_{11}$ and $A_{31}$ holds the raw entries $A_{31}$.

**Step 2: panel triangular solve.** The off-diagonal block satisfies $A_{31} = L_{31} L_{11}^H$, so:
$$L_{31} = A_{31} L_{11}^{-H}$$

solved via `trsm` call. The notation `L11_view` is a `MatrixView` pointing directly into $A$ at offset $(k, k)$ with stride `lda = n` - no copy of the $k_b \times k_b$ panel is made (cf. the related [discussion](#33-submatrix-copying)).

**Step 3: Schur complement update.** Update the trailing submatrix:
$$A_{33} \leftarrow A_{33} - L_{31} L_{31}^H$$

via `herk` routine. This is the critical [Level-3 operation](/reference/BLAS.md#syrk-and-herk-rank-k-updates); for large $n$, essentially all $\tfrac{1}{3} n^3$ arithmetic is performed here across all block iterations.

### 3.2 Unblocked panel kernel

The kernel `chol_unblocked_lower` operates in-place on the $k_b \times k_b$ submatrix of $A$ at global offset $(k, k)$. For each column $j = 0, \ldots, k_b - 1$ (global index $g_j = k + j$):

**Diagonal.** Compute the Schur complement residual and take its square root:
$$d = Re(A_{g_j,\, g_j}) - \sum_{p=0}^{j-1} |A_{g_j,\, k+p}|^2$$

If $d \leq 0$, the matrix is not positive-definite: the kernel returns `false` and `potrf` throws `std::runtime_error`. Otherwise $L_{g_j,\, g_j} = \sqrt{d}$.

**Sub-diagonal column.** For each row $i = g_j + 1, \ldots, k + k_b - 1$:
$$L_{i,\, g_j} = \frac{A_{i,\, g_j} - \displaystyle\sum_{p=0}^{j-1} A_{i,\, k+p}\, \overline{A_{g_j,\, k+p}}}{L_{g_j,\, g_j}}$$

### 3.3 Submatrix copying

`sub_copy_in` and `sub_copy_out` copy rectangular blocks between the working matrix $A$ and contiguous temporary matrices. They exist for the same reason as their specialised counterparts in `lu.hpp`: `trsm` and `herk` require operands whose leading dimension equals the block width to hit the optimised path; a subview of $A$ has `lda = n` rather than $k_b$. The copy normalises the stride.
 
The exception is $L_{11}$ (and $U_{11}$ in the upper variant), which is passed to `trsm` as a `MatrixView` directly into $A$ - no copy. The view carries `lda = n` but `trsm`'s triangular solve reads only the active $k_b \times k_b$ triangle, so the large stride imposes no correctness issue and avoids an $O(k_b^2)$ allocation and copy.
 
The parallelisation threshold for copy operations is `PARALLEL_THRESHOLD_SIMPLE / nc` for `RowMajor` mode of storage (each thread handles at least one complete row) and `PARALLEL_THRESHOLD_SIMPLE / nr` for `ColMajor`, ensuring task granularity remains coarse enough to amortise spawn overhead.

---
## 4. Public API

### 4.1 `potrf()`: master function
```cpp
template<typename T, Layout L>
CholeskyResult<T, L> potrf(const Matrix<T, L>& A, char uplo = 'L');

// Expression overload.
template<typename T, Layout L, typename E>
CholeskyResult<T, L> potrf(const MatExpr<E>& e, char uplo = 'u');
```

The functions performs **PO**sitive definite **TR**iangular **F**actorisation of the given matrix $A$ by copying the relevant triangle into a working matrix and then calling specialised kernel in-place. The copy is limited to the relevant triangle to avoid reading uninitialised or garbage values from the opposite triangle of the caller's matrix. The `MatExpr` overload materialises the expression into a `Matrix<T,L>` object first; it accepts expression-template inputs such as `hermitian(B)` without additional allocation beyond the internal working copy.

**Throws** `std::runtime_error` if a non-positive diagonal pivot is encountered. The `uplo` argument is case-insensitive (`'l'` and `'u'` are accepted alongside `'L'` and `'U'`).

### 4.2 `potrs()`: single and multiple RHS
```cpp
template<typename T, Layout L>
void potrs(const CholeskyResult<T, L>& res, Matrix<T, L>& B);

template<typename T, Layout L>
void potrs(const CholeskyResult<T, L>& res, Vector<T>& b);
```

The matrix overload uses `trsm` (parallelised over RHS columns). The vector overload uses `trsv` (sequential, no allocation, no copy) - perhaps, the most reasonable choice for single-vector solves where `trsm` thread-spawn overhead would dominate.

### 4.3 Determinant computation
```cpp
template<typename T, Layout L>
double cholesky_logdet(const CholeskyResult<T, L>& res);

template<typename T, Layout L>
double cholesky_det(const CholeskyResult<T, L>& res);
```

The choice of log-det strategy is motivated by overflow safety concerns (cf. the relevant discussion [here](/reference/LU.md#43-determinant-computation)). The computation is done as follows:
$$\log|\det A| = 2\sum_{i=0}^{n-1} \log L_{ii}$$
 
All $L_{ii}$ are strictly positive real by construction, so the logarithm is always defined. `cholesky_det` returns $\exp(\texttt{logdet}(\texttt{res}))$; for large $n$ this may itself overflow, and `cholesky_logdet` is preferred when only the log-scale quantity is needed (e.g. for log-likelihood evaluation in Gaussian process models).

### 4.4 Marix inverse
```cpp
template<typename T, Layout L>
Matrix<T, L> potri(const CholeskyResult<T, L>& res);
```

Computes $A^{-1}$ from a previously computed Cholesky factor without modifying `res`. Returns the full Hermitian inverse. The computation proceeds in three steps:
 
**Step 1: triangular inversion (`trtri`).** The factor is copied to a working matrix `Finv` and inverted in-place using a blocked algorithm with block size `TRTRI_BLOCK = 64`. For the lower case, the block-column identity:
 
$$F^{-1} = \begin{bmatrix} F_{11}^{-1} & 0 \\ -F_{22}^{-1} F_{21} F_{11}^{-1} & F_{22}^{-1} \end{bmatrix}$$
 
is applied block-column by block-column. `trtri_unblocked_lower` inverts the diagonal block sequentially ($O(k_b^3)$, cache-resident for `TRTRI_BLOCK = 64`); then two `trsm` calls form the off-diagonal:
```cpp
// tmp = F21 * F11^{-1}  (F11 accessed as MatrixView into Finv).
trsm('R', 'L', 'N', 'N', T(1), F11_view, F21);
// F21_new = -F22^{-1} * tmp  (F22 not yet inverted; also MatrixView).
trsm('L', 'L', 'N', 'N', T(-1), F22_view, F21);
```

**Step 2: Hermitian product via `herk`.** Given $F^{-1} = L^{-1}$ (or $U^{-1}$):
$$A^{-1} = (L^{-1})^H L^{-1}$$

`herk` fills only the requested triangle of `C`, achieving [BLAS-3](/reference/BLAS.md#syrk-and-herk-rank-k-updates) arithmetic intensity for the dominant $O(n^3/3)$ work.

**Step 3: reflection.** The computed triangle is reflected to the opposite triangle via conjugation, parallelised with `parallel_for`, producing the full $n \times n$ Hermitian result matrix.