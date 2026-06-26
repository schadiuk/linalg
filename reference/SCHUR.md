# Schur decomposition: reference
> **Source file:** `schur.hpp`.

> **Dependencies:** `qr.hpp` (inherits [Householder infrastructure](/reference/QR.md#12-householder-reflectors), `larft` and other routines).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundations](#1-mathematical-foundations)
- [Result storage](#2-result-storage-schurresult)

---
## 0. Preamble and notation
Matrices are square $n \times n$ with elements in complex field $\mathbb{C}$ (real-only matrices are promoted). Entries are denoted either as $A_{ij}$ or as $A(i,j)$ using 0-based indexing throughout, matching C++ array conventions. 

| Symbol | Meaning |
|---|---|
| $\|A\|_F$ | Frobenius norm $\sqrt{\sum_{ij} \|A_{ij}\|^2}$. |
| $\|A\|_2$ | Spectral norm (largest singular value). |
| $\|A\|_\infty$ | Maximum absolute row sum. |
| $\kappa(A)$ | Condition number $\|A\|_2\|A^{-1}\|_2$. |
| $\varepsilon$ | Machine epsilon ($\approx 2.2 \times 10^{-16}$ for `double`). |

---
## 1. Mathematical foundations

### 1.1 Existence
For every square matrix $A \in \mathbb{C}^{n \times n}$ there exists a unitary $Q \in \mathbb{C}^{n \times n}$ and an upper-triangular $T \in \mathbb{C}^{n \times n}$ such that
$$A = Q T Q^H$$

Equivalently, $Q^H A Q = T$. The columns of $Q$ are the **Schur vectors** of $A$ and the diagonal entries $\{T_{11}, \ldots, T_{nn}\} = \{\lambda_1, \ldots, \lambda_n\}$ are the eigenvalues of $A$, each counted with algebraic multiplicity. $T$ is called the **Schur form** (or **Schur triangular form**) of $A$.

**Proof of existence.** Induction on $n$. For $n = 1$ the result is trivial. For $n \ge 2$, let $\lambda_1$ be any eigenvalue of $A$ (which exists over $\mathbb{C}$ by the fundamental theorem of algebra) with unit eigenvector $q_1$. Extend $q_1$ to a unitary basis $Q_1 = [q_1 \mid \hat{Q}_1]$. Then
 
$$Q_1^H A Q_1 = \begin{bmatrix} \lambda_1 & a_{12}^T \\ 0 & A_{22} \end{bmatrix}$$
 
where the zero block follows from $A q_1 = \lambda_1 q_1$ and unitary change of basis. By the induction hypothesis applied to $A_{22} \in \mathbb{C}^{(n-1)\times(n-1)}$ there exists unitary $\hat{Q}$ with $\hat{Q}^H A_{22} \hat{Q} = T_{22}$ upper-triangular. Setting $Q = Q_1 \operatorname{diag}(1, \hat{Q})$ yields $Q^H A Q$ upper-triangular.

### 1.2 Non-uniqueness
Schur form $T$ is not unique. The $n!$ possible orderings of the diagonal eigenvalues are achievable by further unitary similarity. Similarly, $Q$ is unique only up to right-multiplication by a unitary diagonal matrix $D = \mathrm{diag}(e^{i\theta_1}, \ldots, e^{i\theta_n})$, since replacing $Q \to QD$ and $T \to D^H T D$ preserves the structure ($A = Q T Q^H$) while producing a different (but still upper-triangular) Schur form.

### 1.3 Special cases
**Normal matrices.** $A$ is normal ($A^H A = A A^H$) if and only if $T$ is diagonal. In that case the Schur vectors are orthonormal eigenvectors, and Schur form is the full spectral decomposition. 

The eigendecomposition $A = X \Lambda X^{-1}$ (with diagonal $\Lambda$ and possibly non-unitary $X$) exists only when $A$ is diagonalizable. The Schur decomposition always exists and uses a unitary $Q$, making it the numerically preferable factorisation for computing eigenvalues.

**Real matrices.** For $A \in \mathbb{R}^{n \times n}$, complex eigenvalues occur in conjugate pairs. There exists a real orthogonal $Q_{\mathbb{R}}$ reducing $A$ to **real Schur form** (quasi-upper-triangular, with $1 \times 1$ and $2 \times 2$ diagonal blocks). This library always works in `std::complex` regardless of input type, producing the **complex Schur form** with unit diagonal blocks only. Real input is promoted entry-by-entry before the computation begins.

---
## 2. Result storage: `SchurResult`
```cpp
template<Layout LL = Layout::RowMajor>
struct SchurResult {
    Matrix<DefaultScalar, LL> T;    // Upper-triangular Schur factor.
    Matrix<DefaultScalar, LL> Q;    // Unitary matrix of Schur vectors.
    Vector<DefaultScalar> eigvals;  // n eigenvalues.
    Vector<double> balance_scale;   // Per-index scaling factors.
    Vector<size_t> balance_perm;    // Permutation for balancing.
    bool balanced;                  // True iff balancing was actually performed.
};
```

`DefaultScalar` is an alias for `std::complex<double>`. All matrix fields carry this type unconditionally. Field sizes are set as follows:

| Field | `compute_vectors = true` | `compute_vectors = false` |
|---|---|---|
| `T` | $n \times n$ | $n \times n$ |
| `Q` | $n \times n$ | $0 \times 0$ (empty) |
| `eigvals` | $n$ | $n$ |
| `balance_scale` | $n$ (zeros if not balanced) | $n$ |
| `balance_perm`  | $n$ (identity if not balanced) | $n$ |
 
---
