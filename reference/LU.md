# LU decomposition: reference
This document covers the implementation of LU algorithm present in, and used by the `linalg` library.
> **Source file:** `lu.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundation)
- [Packed storage](#2-packed-storage-layout)
- [Algorithm](#3-lu_factor-the-blocked-algorithm)
- [Public API](#4-public-api)
---
## 0. Preamble and notation
Matrices are $m \times n$ with entries in $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. Elements are denoted either as $A_{ij}$ or as $A(i,j)$ using **0-based indexing throughout**, matching standard C++ array conventions.
 
The following norms and quantities appear in the analysis below.
 
| Symbol | Meaning |
|---|---|
| $\|A\|_F$ | Frobenius norm $\sqrt{\sum_{ij} \|A_{ij}\|^2}$. |
| $\|A\|_2$ | Spectral norm (largest singular value). |
| $\|A\|_\infty$ | Maximum absolute row sum. |
| $\kappa(A)$ | Condition number $\|A\|_2 \|A^{-1}\|_2$. |
| $k$ | $k = \min(m, n)$ throughout. |
| $\varepsilon$ | Machine epsilon ($\approx 2.2 \times 10^{-16}$ for `double`). |

**Packed vs. unpacked representation.** After factorisation, `lu_factor` stores both factors in the original matrix footprint - the *packed* matrix. The strict lower triangle (below the main diagonal) holds the multipliers of $L$; the upper triangle (including the diagonal) holds $U$. The unit diagonal of $L$ is not stored. The *unpacked* representation splits these into separate `L` and `U` matrices and is produced on demand by the `unpack_L` / `unpack_U` helpers. Solvers always operate on the packed form; the unpacked form is provided for display and verification.
 
**Pivot vector convention.** The library uses a *sequential transposition* encoding: `piv[j]` stores the row index that was physically swapped with row $j$ at step $j$ of the unblocked panel factorisation. This is distinct from a *destination permutation*, where `piv[j]` would name the final resting place of original row $j$. The sequential encoding is the LAPACK convention and is what `lu_solve` applies directly (left-to-right swap replay).

---
## 1. Mathematical foundations

### 1.1 Gaussian elimination

LU decomposition is a formal account of Gaussian elimination. To motivate the matrix identity $PA = LU$, we trace through elimination algebraically for a square $n \times n$ matrix; the rectangular case follows by stopping at column yielded by $\min(m,n)$.

**Step 0.** Suppose no pivoting is needed for now. At the first elimination step, subtract multiples of row 0 from all rows below it to zero column 0:

$$M_0 A = \begin{bmatrix} 1 & & & \\ -\ell_{10} & 1 & & \\ \vdots & & \ddots & \\ -\ell_{n-1,0} & & & 1 \end{bmatrix} A = A^{(1)}$$

where $\ell_{i0} = A_{i0}/A_{00}$ for $i \geq 1$ and $A^{(1)}$ has zeros in column 0 below the diagonal. $M_0$ is a [Frobenius matrix](https://en.wikipedia.org/wiki/Frobenius_matrix) (in our case, differing from the identity only in column 0).

**General step $j$.** Apply $M_j$ to $A^{(j)}$ to zero column $j$ below the diagonal, producing $A^{(j+1)}$. The Frobenius matrix $M_j$ has $-\ell_{ij} = -A^{(j)}_{ij}/A^{(j)}_{jj}$ in position $(i,j)$ for $i > j$, and 1s on the diagonal.

After $n-1$ steps:

$$M_{n-2} \cdots M_1 M_0 A = U$$

where $U$ is upper triangular. Inverting:

$$A = M_0^{-1} M_1^{-1} \cdots M_{n-2}^{-1} U$$

The inverse of a Frobenius matrix simply negates its subdiagonal column, and the product of Frobenius matrices with non-overlapping subdiagonal columns is assembled without fill-in:

$$M_j^{-1} = I + \ell_j e_j^\top$$

$$L := M_0^{-1} M_1^{-1} \cdots M_{n-2}^{-1} = I + \sum_{j=0}^{n-2} \ell_j e_j^\top$$

which is unit lower triangular with $L_{ij} = \ell_{ij}$ for $i > j$. This gives $A = LU$.

***With partial pivoting.*** At step $j$, before applying $M_j$, a permutation $P_j$ swaps row $j$ with the row of largest absolute value in column $j$ below the diagonal (the pivot row). The product of all step-wise permutations is $P = P_{n-2} \cdots P_1 P_0$, and the factorisation satisfies:

$$PA = LU$$

The Frobenius matrices $M_j$ are recomputed after each swap, ensuring $|L_{ij}| \leq 1$ for all $i > j$ - a consequence of dividing by the maximum element in the column.

***General $m \times n$ case.*** For $m \times n$ matrices, $k = \min(m,n)$ elimination steps suffice. $L \in \mathbb{F}^{m \times k}$ is $m \times k$ unit lower triangular, $U \in \mathbb{F}^{k \times n}$ is $k \times n$ upper triangular, and $P \in \mathbb{F}^{m \times m}$ is an $m \times m$ permutation. The identity $PA = LU$ holds exactly.

### 1.2 Existence and uniqueness

**Existence** follows by the constructive argument above: the algorithm always terminates. If $A^{(j)}_{jj} = 0$ and all entries of column $j$ below row $j$ are also zero, the multipliers $\ell_{ij}$ are zero (no update is needed) and the algorithm skips to column $j+1$. This handles singular and rank-deficient matrices without special treatment.
 
**Uniqueness.** Suppose $PA = L_1 U_1 = L_2 U_2$ with both $L_1, L_2$ unit lower triangular and $U_1, U_2$ upper triangular. Then:

$$L_2^{-1} L_1 = U_2 U_1^{-1}$$

The left side is unit lower triangular (product of unit lower triangular matrices); the right side is upper triangular. A matrix that is simultaneously unit lower triangular and upper triangular must be the identity. Hence $L_2^{-1} L_1 = I$ and $U_2 U_1^{-1} = I$, giving $L_1 = L_2$ and $U_1 = U_2$.

This argument requires $U_1$ to be invertible, i.e. $A$ to have full column rank. If $\text{rank}(A) = r < \min(m,n)$, then $U$ has zeros on the diagonal at positions $r, r+1, \ldots$ and the factorisation is not unique in those degrees of freedom - the multipliers filling column $j$ when $U_{jj} = 0$ can be chosen arbitrarily. The implementation sets them to zero by leaving the subdiagonal of column $j$ unchanged (since $1/\text{diag} = 1/0$ is guarded by `if (diag != T(0))`), which is a valid but not unique choice.

### 1.3 The pivot vector and permutation

**Sequential transposition encoding.** The `piv` vector of length $k = \min(m,n)$ records, at each step $j$, the row index `piv[j]` that was physically swapped with row $j$. The full permutation $P$ is the composition of elementary transpositions $\tau_j = (j,\, \text{piv}[j])$:

$$P = \tau_{k-1} \cdots \tau_1 \tau_0$$

where $\tau_j$ is applied first (rightmost in the product). Applying the swaps left-to-right in `lu_solve` (in order $j = 0, 1, \ldots, k-1$) replays this composition faithfully.

**Recovering $P$ as a matrix.** The `piv_to_P` reconstruction:
```cpp
Matrix<T, L> P = Matrix<T, L>::identity(m);
for (size_t i = 0; i < piv.size(); ++i)
    if (piv[i] != i)
        for (size_t j = 0; j < m; ++j)
            std::swap(P(i, j), P(piv[i], j));
return P;
```

accumulates the permutation by applying each swap to the current $P$ in order. This is equivalent to left-multiplying the identity by $\tau_0, \tau_1, \ldots$ in sequence, which builds $\tau_{k-1} \cdots \tau_0 \cdot I = P$.

**Sign of the permutation.** The determinant of $P$ is $\pm 1$, namely  $(-1)^s$ where $s$ is the minimum number of transpositions needed to express $P$. Every permutation decomposes uniquely (up to ordering) into disjoint cycles; an $\ell$-cycle requires exactly $\ell - 1$ transpositions, contributing $(-1)^{\ell-1}$ to the sign. Summing over all $c$ cycles:

$$\det P = (-1)^{\sum_{\text{cycles}} (\ell_i - 1)} = (-1)^{n - c}$$

since $\sum_i \ell_i = n$ (every element belongs to exactly one cycle). This is the formula used in `lu_det`.

**Why counting fixed points is wrong:** a naive approach is to count the number of entries $j$ where `piv[j] != j`, treating each as one transposition. But a 3-cycle (i.e. $a \to b \to c \to a$) has 3 non-fixed points yet decomposes into exactly 2 transpositions: $(a\,c) \circ (a\,b)$. The naive count gives 3 transpositions, flipping the sign.

### 1.4 Blocked LU: Schur complement

The blocked algorithm's correctness rests on the **Schur complement identity**. After processing the first block column of width $k_b$, suppose the working matrix has been factored so that:
 
$$A^{(0)} = \begin{bmatrix} L_{11} & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} U_{11} & U_{12} \\ 0 & S \end{bmatrix}$$
 
where $L_{11}$ is $k_b \times k_b$ unit lower triangular, $U_{11}$ is $k_b \times k_b$ upper triangular, and $S$ is the Schur complement:
 
$$S = A_{22} - L_{21} U_{12}$$
 
$S$ is exactly the trailing submatrix that remains after eliminating the first $k_b$ columns. The key point is that $S$ has the same LU factorisation structure as a standalone matrix: factoring $S = \hat{L} \hat{U}$ (with appropriate pivoting applied to the full trailing rows) gives the remaining blocks of $L$ and $U$. Induction over block columns proves that the blocked algorithm computes the same factorisation as the unblocked algorithm, differing only in the order of arithmetic operations.
 
In the implementation, [Step 3](#31-block-structure) (Schur complement update) computes exactly $A_{33} \leftarrow A_{33} - L_{31} U_{23}$. The subsequent block iteration then factors this updated $A_{33}$ as if it were an independent matrix, which is precisely the inductive step.

---
## 2. Packed storage layout
After `lu_factor` returns, the in-place matrix `packed` holds:

$$\text{packed}[i,j] = \begin{cases} L_{ij} & i > j \quad (\text{strict lower triangle; unit diagonal not stored}) \\ U_{ij} & i \leq j \quad (\text{upper triangle including diagonal}) \end{cases}$$

Let us consider a concrete $4 \times 4$ example. Suppose the factorisation yielded:

$$L = \begin{bmatrix} 1 & & & \\ 0.5 & 1 & & \\ 0.2 & 0.8 & 1 & \\ 0.4 & 0.1 & 0.6 & 1 \end{bmatrix}, \qquad U = \begin{bmatrix} 2 & 3 & 1 & 4 \\ & 5 & 2 & 1 \\ & & 3 & 2 \\ & & & 7 \end{bmatrix}$$

The packed matrix representation is:

$$\text{packed} = \begin{bmatrix} 2 & 3 & 1 & 4 \\ 0.5 & 5 & 2 & 1 \\ 0.2 & 0.8 & 3 & 2 \\ 0.4 & 0.1 & 0.6 & 7 \end{bmatrix}$$

The unit  diagonal of $L$ is implicit; the "freed" positions store $U_{ii}$ instead. This is precisely the information the [BLAS solvers](/reference/BLAS.md#trsm-multiple-rhs-triangular-solve) need: `trsm('L','L','N','U',...)` reads only the strict lower triangle (the multipliers) and assumes unit diagonal, while `trsm('L','U','N','N',...)` reads the full upper triangle including the diagonal.

### `unpack_L` and `unpack_U`

Both helpers allocate a fresh output matrix and fill it from `packed`, parallelised over the major dimension. The separate allocation is unavoidable in the public API: `packed` must remain intact and unmodified for use by `lu_solve`, `lu_det`, and `lu_inverse`. Any in-place extraction would corrupt the packed representation.

For instance, the row-major path in `unpack_L` is as follows:
```cpp
parallel_for(m, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / mn),
    [&](size_t rs, size_t re) {
        for (size_t i = rs; i < re; ++i)
            for (size_t j = 0; j < mn; ++j) {
                if (j == i) Lo(i, j) = T(1);
                else if (j <  i) Lo(i, j) = pk(i, j);
                // j > i: left at T(0) from construction.
            };
});
```

---
## 3. `lu_factor`: the blocked algorithm

### 3.1 Block structure

An unblocked LU decomposition performs $O(n^3/3)$ arithmetic operations, but each rank-1 trailing update reads and writes the full trailing submatrix - a bandwidth-bound Level-2 operation. For a 2048-element-wide double matrix the trailing update at step $k = 0$ writes approximately $4\,\text{MB}$ per element, far exceeding typical L1 and L2 cache.

Blocking reorganises the work into three phases per block column. The block width is set as a constant `LU_BLOCK = 64`. At outer step $k$, let $k_b = \min(\text{LU\_BLOCK},\, mn - k)$ and partition the current working matrix as:

$$A^{(k)} = \begin{bmatrix} A_{11} & A_{12} & A_{13} \\ A_{21} & A_{22} & A_{23} \\ A_{31} & A_{32} & A_{33} \end{bmatrix}$$

where the blocks have dimensions:

| Block | Rows | Columns | Description |
|---|---|---|---|
| $A_{11}$ | $k$ | $k$ | Already factored. |
| $A_{22}$ | $k_b$ | $k_b$ | Current panel diagonal. |
| $A_{32}$ | $m - k - k_b$ | $k_b$ | Panel below diagonal. |
| $A_{23}$ | $k_b$ | $n - k - k_b$ | Row block to the right. |
| $A_{33}$ | $m - k - k_b$ | $n - k - k_b$ | Trailing submatrix. |

The three steps per block column are  as follows:

**Step 1: unblocked panel LU.** Factor the panel column $\bigl[\begin{smallmatrix} A_{22} \\ A_{32} \end{smallmatrix}\bigr]$ in-place using the standard unblocked algorithm with partial pivoting. Row swaps are applied to the **entire row** $[0:n]$ to keep the full packed matrix consistent (cf. [section 3.3](#33-full-row-swap)). After this step, the lower part of the panel holds the multipliers $L_{21}$, $L_{31}$, and the upper part holds $U_{22}$.

**Step 2: panel triangular solve.** Solve $L_{22} \cdot U_{23} = A_{23}$ for $U_{23}$ via `trsm('L','L','N','U', T(1), panel, U12)`. The extracted panel `panel(kb, kb)` is a fresh `Matrix<T,L>`, not a `MatrixView` into $A$, for the reason discussed in [3.6](#36-submatrix-copies).

**Step 3: Schur complement update.** Update the trailing submatrix:

$$A_{33} \leftarrow A_{33} - L_{31} \cdot U_{23}$$

via optimised BLAS `gemm(T(-1), L21, U12, T(1), C)`. This is the critical Level-3 operation; for large $n$, essentially all $O(n^3/3)$ arithmetic is performed here across all block iterations.

### 3.2 Pivot search

The search at panel step $j$ scans rows $[j, m)$ of column $j$ for the entry of largest absolute value:
```cpp
size_t pr = j;
auto  mx = std::abs(static_cast<T>(A(j, j)));
for (size_t i = j + 1; i < m; ++i) {
    const auto v = std::abs(static_cast<T>(A(i, j)));
    if (v > mx) { mx = v; pr = i; };
};
piv[j] = pr;
```

This is an $O(m)$ sequential scan. Parallelising it would require a parallel max-reduction with a subsequent all-thread broadcast, with synchronisation overhead dominating for all but extremely large panels (the crossover point is well beyond what typical panel widths reach at `LU_BLOCK = 64`). The [BLAS](/reference/BLAS.md#iamax--iamin-index-of-extremal-element) `iamax` routine makes the same choice for exactly the same reason.

### 3.3 Full-row swap

After identifying pivot row `pr`, the swap is applied to the entire row, not just the current block column:
```cpp
if constexpr (L == Layout::RowMajor) {
    T* LINALG_RESTRICT rj = ap + j  * lda;
    T* LINALG_RESTRICT rpr = ap + pr * lda;
    parallel_for(n, PARALLEL_THRESHOLD_SIMPLE,
        [rj, rpr](size_t s, size_t e) {
            LINALG_VECTORIZE
            for (size_t jj = s; jj < e; ++jj)
                std::swap(rj[jj], rpr[jj]);
        });
} else {
    parallel_for(n, PARALLEL_THRESHOLD_SIMPLE,
        [ap, lda, j, pr](size_t s, size_t e) {
            for (size_t jj = s; jj < e; ++jj)
                std::swap(ap[jj*lda + j], ap[jj*lda + pr]);
        });
};
```

Columns $[0, k)$ of the working matrix already hold the multipliers from previous block steps. If those multipliers were not swapped, the final packed representation would encode a different permutation for the factored part than for the unfactored part, and $PA = LU$ equality would not hold. Keeping the swap global ensures that at termination `packed` is a valid simultaneous encoding of both factors under a single consistent permutation $P$.

For `RowMajor` storage the swap is a vectorised `parallel_for` over a contiguous row. For `ColMajor` mode, the swap accesses elements at stride `lda` (one per column), which prevents auto-vectorisation; the loop is kept scalar.

### 3.4 Multiplier scaling

After the pivot swap is applied, all subdiagonal entries of column $j$ are divided by the pivot to produce multipliers:
```cpp
const T diag = static_cast<T>(A(j, j));
if (diag != T(0)) {
    const T inv = T(1) / diag;
    if constexpr (L == Layout::RowMajor) { // Strided scalar loop.
        for (size_t i = j + 1; i < m; ++i)
            ap[i*lda + j] *= inv;
    } else {
        // ColMajor: contiguous column, vectorisable.
        T* LINALG_RESTRICT col_j = ap + j * lda;
        LINALG_VECTORIZE
        for (size_t i = j + 1; i < m; ++i) col_j[i] *= inv;
    };
};
```

The single division produces `inv`; all $(m - j - 1)$ subsequent operations are multiplications. This matters particularly for complex types, where complex division costs approximately four real multiplications plus two additions plus a real division, whereas complex multiplication costs four real multiplications and two additions. Hoisting the division outside the loop is a measurable optimisation for large panels.

### 3.5 Rank-1 panel update

After scaling, the remaining panel columns (within the current block, indices $[j+1,\, k+k_b)$) are updated by subtracting the outer product of the multiplier column and the pivot row:

$$A[i,\; j{+}1 : k{+}k_b] \mathrel{-}= L_{ij} \cdot A[j,\; j{+}1 : k{+}k_b] \quad \forall\, i > j$$

In case of row-major storage, the implementation is:
```cpp
const T* LINALG_RESTRICT pivot_row = ap + j * lda;
for (size_t i = j + 1; i < m; ++i) {
    const T lij = ap[i*lda + j];
    if (lij == T(0)) continue;
    T* LINALG_RESTRICT tgt_row = ap + i * lda;
    LINALG_VECTORIZE
    for (size_t jj = j + 1; jj < right_start; ++jj) tgt_row[jj] -= lij * pivot_row[jj];
};
```

Each inner loop is a contiguous SAXPY over the current block's column range - vectorisable. For `ColMajor` mode, the outer loop is over block-columns $jj$ and the inner is a scaled column subtraction, also accessing a contiguous memory:
```cpp
const T* LINALG_RESTRICT l_col = ap + j * lda;
for (size_t jj = j + 1; jj < right_start; ++jj) {
    const T pval = ap[jj*lda + j];
    if (pval == T(0)) continue;
    T* LINALG_RESTRICT tgt_col = ap + jj * lda;
    LINALG_VECTORIZE
    for (size_t i = j + 1; i < m; ++i) tgt_col[i] -= l_col[i] * pval;
};
```

### 3.6 Submatrix copies
`sub_copy_in(dst, r0, c0)` copies the rectangular block starting at `(r0, c0)` of $A$ into the fresh `Matrix<T,L>` `dst`. `sub_copy_out(src, r0, c0)` copies it back. These lambdas exist because `trsm` and `gemm` require operands with unit-stride leading dimensions to hit the `raw_mat_info` fast path; a subview of $A$ has `lda = n` (the full matrix width) rather than the block width, which would pass a valid but oversized stride to the BLAS kernel. The copy normalises the stride and eliminates the indirection.

The parallelisation threshold is `PARALLEL_THRESHOLD_SIMPLE / nc` for `RowMajor` storage (each thread handles at least one complete row) and `PARALLEL_THRESHOLD_SIMPLE / nr` for `ColMajor` (at least one complete column), ensuring task granularity remains coarse enough to amortise spawn overhead.

---
## 4. Public API

### 4.1 `lu()`: the entry point
```cpp
template<typename T, Layout L>
LUResult<T, L> lu(const Matrix<T, L>& A);

template<typename T, Layout L, typename E>
LUResult<T, L> lu(const MatExpr<E>& e); // Materialises first.
```

The function makes a copy of $A$ into a working matrix `work`, calls `detail::lu_factor(work)` in-place, then unpacks:
```cpp
Matrix<T, L> work = A;
Vector<size_t> piv = detail::lu_factor(work);
return LUResult<T, L> {
    detail::piv_to_P<T, L>(piv, A.rows()),
    detail::unpack_L<T, L>(work),
    detail::unpack_U<T, L>(work),
    std::move(work), // Packed representation.
    std::move(piv)
};
```

The copy is unavoidable: `lu_factor` overwrites its argument with the packed representation, destroying the original values. The `LUResult` struct bundles both the ready factors ($P$, $L$, $U$) and the packed form so that callers can inspect the factors for display or debugging while still using the packed form for efficient solves, without refactoring.

The `MatExpr` overload materialises the expression into a `Matrix<T,L>` first. It cannot accept a `MatrixView` directly because the factorisation requires a contiguous mutable matrix.

### 4.2 `lu_solve()`: single and multiple RHS

**Single RHS.** `lu_solve(res, b)` solves $Ax = b$ by three sequential steps operating entirely on the packed representation:
```cpp
// Apply row permutation.
for (size_t i = 0; i < n; ++i) if (piv[i] != i) std::swap(b[i], b[piv[i]]);
// Forward substitution: L * x = b (unit lower triangular).
trsv('L', 'N', 'U', LU, b);
// Backward substitution: U * x = b (non-unit upper triangular).
trsv('U', 'N', 'N', LU, b);
```

**Multiple RHS.** `lu_solve(res, B)` handles a matrix of right-hand sides. The permutation step is parallelised: for each row swap $i \leftrightarrow \text{piv}[i]$, the range of column indices $[0, \text{nrhs})$ is distributed across threads via `parallel_for` with `LINALG_VECTORIZE`. The triangular solves, in turn, are promoted to `trsm` calls:
```cpp
for (size_t i = 0; i < n; ++i) {
    if (piv[i] != i) {
        const size_t pi = piv[i];
        parallel_for(nrhs, PARALLEL_THRESHOLD_SIMPLE,
            [&B, i, pi](size_t s, size_t e) {
                LINALG_VECTORIZE
                for (size_t j = s; j < e; ++j)
                    std::swap(B(i, j), B(pi, j));
            });
    };
};
// Forward substitution: L * X = PB (unit lower triangular).
trsm('L', 'L', 'N', 'U', T(1), LU, B);
// Backward substitution: U * X = (above) (non-unit upper triangular).
trsm('L', 'U', 'N', 'N', T(1), LU, B);
```

### 4.3 Determinant computation

**Why log-sum?** The product $\prod_{i=0}^{n-1} U_{ii}$ overflows double precision for $n \gtrsim 700$ when typical diagonal magnitudes exceed $1$. The log-magnitude representation avoids overflow:

$$|\det A| = \exp\!\left(\sum_{i=0}^{n-1} \log|U_{ii}|\right), \quad \det A = \det P \cdot \left(\prod_{i} \operatorname{sign}(U_{ii})\right) \cdot |\det A|$$

The sign, yielded by $\det P$ is $(-1)^{n - c}$ where $c$ is the number of disjoint cycles in the permutation $P$ (cf. the [discussion](#13-the-pivot-vector-and-permutation)). Nature of the entries affects the general algorithm as follows:

- **Real determinant.** The accumulation tracks log-magnitude and the sign of each diagonal entry separately:
```cpp
double log_abs = 0.0;
double diag_sign = 1.0;
for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(LU(i, i));
    if (d == 0.0) return T(0);
    log_abs += std::log(std::abs(d));
    if (d < 0.0) diag_sign = -diag_sign;
};
return static_cast<T>(perm_sign * diag_sign * std::exp(log_abs));
```

- **Complex determinant.** For complex $T$, both magnitude and argument must be tracked:
 
$$\log|\det A| = \sum_i \log|U_{ii}|, \qquad \arg(\det A) = \arg(\det P) + \sum_i \arg(U_{ii})$$
 
where $\arg(\det P) = 0$ if `perm_sign > 0` else $\pi$. The reconstruction:
 
$$\det A = e^{\sum \log|U_{ii}|} \cdot e^{i \cdot \arg(\det A)}$$
 
avoids chaining complex multiplications, which would accumulate both magnitude error and argument drift proportional to $n$.

### 4.4 Inverse
```cpp
template<typename T, Layout L>
Matrix<T, L> lu_inverse(const LUResult<T, L>& res);
```

The inverse is computed by solving $A \cdot X = I$:
```cpp
Matrix<T, L> inv = Matrix<T, L>::identity(res.packed.rows());
lu_solve(res, inv);
return inv;
```

Each column of the identity becomes the corresponding column of $A^{-1}$ after the solve. Because the argument is a `Matrix<T,L>`, the multiple-RHS `trsm` path is activated automatically, achieving [BLAS-3](/reference/BLAS.md#trsm-multiple-rhs-triangular-solve) arithmetic intensity. The function requires the input `res` to have been computed for a square matrix (enforced by a `BOUNDS_CHECK` macro).