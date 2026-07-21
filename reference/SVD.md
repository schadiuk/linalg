# Singular Value Decomposition: reference
> **Source files:** `bidiag.hpp`, `svd.hpp`.

> **Dependees:** `qr.hpp` (reuses [Householder infrastructure](/reference/QR.md#12-householder-reflectors) and other routines).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundations)


---
## 0. Preamble and notation

Matrices are $m \times n$ with entries in field $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. Unless stated otherwise $m \geq n$ (the "tall" case); $m < n$ is reduced to this case by transposition.

| Symbol | Meaning |
|---|---|
| $k$ | $k = \min(m,n)$, the number of singular values / reflector pairs. |
| $n_b$ | Block size `BIDIAG_BLOCK = 32`. |
| $\varepsilon$ | Machine epsilon. |
| $\kappa(A)$ | Condition number $\|A\|_2\|A^{-1}\|_2$. |
| $B$ | Real bidiagonal matrix, diagonal $d_i$, superdiagonal $e_i$ ($k-1$ entries). |
| $U, V$ | Unitary factors, $m \times k$ and $n \times k$ (reduced form) respectively. |
| $\Sigma$ | $\operatorname{diag}(s_0, \ldots, s_{k-1})$, $s_0 \geq s_1 \geq \cdots \geq 0$. |
| $\beta$ | Householder scalar in this library's own convention, $H = I - \beta v v^H$, $v$ arbitrarily scaled. |
| $\tau, v_1$ | The *same* reflector re-expressed in LAPACK convention: $v_1[0] = 1$, $\tau = \beta \lvert v[0] \rvert^2$.|

**Result convention.** `A = U * B * V^H`, with `B` real bidiagonal (`BidiagResult`) or real diagonal (`SVDResult`, after $B$'s own SVD is folded in). Singular values in `SVDResult::s` are sorted descending and non-negative.

**Two source files, one algorithm.** `bidiag.hpp` performs the exact, finite unitary reduction to bidiagonal form; `svd.hpp` performs the iterative diagonalisation of the resulting bidiagonal matrix. The two stages are logically inseparable (SVD is *defined* here as bidiagonalisation followed by Golub–Kahan–Reinsch iteration), so this document treats them as one algorithm across two files.

---
## 1. Mathematical foundations

### 1.1 Existence and uniqueness

For any $A \in \mathbb{F}^{m \times n}$ there exist unitary $U \in \mathbb{F}^{m \times m}$, $V \in \mathbb{F}^{n \times n}$, and $\Sigma \in \mathbb{R}^{m \times n}$ diagonal with non-negative entries $s_0 \geq s_1 \geq \cdots \geq s_{k-1} \geq 0$, such that $A = U \Sigma V^H$. The $s_i$ (singular values) are always unique. The corresponding columns of $U$, $V$ (singular vectors) are unique up to:
 
- a common unit-phase rotation of a paired column $(u_i, v_i)$ when $s_i$ is simple (for real $T$, this degenerates to a common sign flip);
- an arbitrary unitary mixing within any subspace spanned by columns sharing a repeated singular value.

This library performs the **reduced (thin)** form of the algorithm, returning square $\Sigma \in \mathbb{R} ^ {k \times k}$ ($U$ is $m \times k$ and $V$ is $n \times k$ accordingly).

### 1.2 The two-phase strategy

The singular values of $A$ are the square roots of the eigenvalues of $A^H A$ (or $AA^H$), and the columns of $V$ (resp. $U$) are the corresponding eigenvectors. Forming $A^H A$ explicitly and diagonalising it is mathematically valid but squares the condition number: $\kappa(A^H A) = \kappa(A)^2$, and small singular values close to $\varepsilon \|A\|$ are destroyed by the rounding error already present in forming the product. The two-stage strategy is capable of avoiding this:
 
1. **Bidiagonalisation:**  an exact, finite sequence of Householder reflectors applied alternately from the left and right reduces $A$ to a real bidiagonal $B$, $A = UBV^H$, with $U, V$ unitary. This step is backward-stable and costs $O(mn^2)$ - same order as QR.
2. **Diagonalisation of $B$**. The singular values and vectors of the *real bidiagonal* $B$ are found by an implicit-shift QR iteration that operates on $B$'s entries directly - equivalent to applying shifted QR to the tridiagonal $B^H B$, but **without ever forming $B^H B$**. Since $B$ is already bidiagonal, this iteration is $O(k)$ per sweep rather than the $O(k^3)$ a naive dense eigensolver would cost.

Combining the two above steps yields the desired decomposition: $A = U B V^H = U (U' \Sigma V'^H) V^H = (UU') \cdot \Sigma \cdot (VV')^H$.