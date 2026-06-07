# LU decomposition: reference
This document covers the implementation of LU algorithm present in, and used by the `linalg` library.
> **Source file:** `lu.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundation)
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

The Frobenius matrices $M_j$ are recomputed after each swap, ensuring $|L_{ij}| \leq 1$ for all $i > j$ — a consequence of dividing by the maximum element in the column.

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

**Sign of the permutation.** The determinant of $P$ is $\pm 1$ and equals $(-1)^s$ where $s$ is the minimum number of transpositions needed to express $P$. Every permutation decomposes uniquely (up to ordering) into disjoint cycles; an $\ell$-cycle requires exactly $\ell - 1$ transpositions, contributing $(-1)^{\ell-1}$ to the sign. Summing over all $c$ cycles:

$$\det P = (-1)^{\sum_{\text{cycles}} (\ell_i - 1)} = (-1)^{n - c}$$

since $\sum_i \ell_i = n$ (every element belongs to exactly one cycle). This is the formula used in `lu_det`.

**Why counting fixed points is wrong:** a naive approach is to count the number of entries $j$ where `piv[j] != j`, treating each as one transposition. But a 3-cycle (i.e. $a \to b \to c \to a$) has 3 non-fixed points yet decomposes into exactly 2 transpositions: $(a\,c) \circ (a\,b)$. The naive count gives 3 transpositions, flipping the sign.

### 1.4 Blocked LU: Schur complement

The blocked algorithm's correctness rests on the **Schur complement identity**. After processing the first block column of width $k_b$, suppose the working matrix has been factored so that:
 
$$A^{(0)} = \begin{bmatrix} L_{11} & 0 \\ L_{21} & I \end{bmatrix} \begin{bmatrix} U_{11} & U_{12} \\ 0 & S \end{bmatrix}$$
 
where $L_{11}$ is $k_b \times k_b$ unit lower triangular, $U_{11}$ is $k_b \times k_b$ upper triangular, and $S$ is the Schur complement:
 
$$S = A_{22} - L_{21} U_{12}$$
 
$S$ is exactly the trailing submatrix that remains after eliminating the first $k_b$ columns. The key point is that $S$ has the same LU factorisation structure as a standalone matrix: factoring $S = \hat{L} \hat{U}$ (with appropriate pivoting applied to the full trailing rows) gives the remaining blocks of $L$ and $U$. Induction over block columns proves that the blocked algorithm computes the same factorisation as the unblocked algorithm, differing only in the order of arithmetic operations.
 
In the implementation, [Step 3] (Schur complement update) computes exactly $A_{33} \leftarrow A_{33} - L_{31} U_{23}$. The subsequent block iteration then factors this updated $A_{33}$ as if it were an independent matrix, which is precisely the inductive step.

---