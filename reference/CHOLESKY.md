# Cholesky factorisation: reference
This document covers the implementation of Cholesky decomposition algorithm.
> **Source file:** `cholesky.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundations](#1-mathematical-foundations)
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

**Naming.** Rather than using the *(somewhat lengthy)* surname-based naming, the public API (`potrf`, `potrs`, `potri`) is adherent to the [LAPACK convention](https://www.netlib.org/lapack/explore-html/d2/d09/group__potrf.html) for positive-definite triangular routines.

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