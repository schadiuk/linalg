# Schur decomposition: reference
> **Source file:** `schur.hpp`.

> **Dependencies:** `qr.hpp` (inherits [Householder infrastructure](/reference/QR.md#12-householder-reflectors), `larft` and other routines).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundations](#1-mathematical-foundations)
- [Result storage](#2-result-storage-schurresult)
- [Balancing](#3-balancing)
- [Hessenberg reduction](#4-hessenberg-reduction)
- [Francis step](#5-francis-implicit-single-shift-qr)
- [Wilkinson shift strategy](#6-shift-strategy-and-deflation)
- [Schur vector accumulation](#7-schur-vector-accumulation)

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

`DefaultScalar` is an alias for `std::complex<double>`, and all matrix entries carry this type unconditionally. Field sizes are set as follows:

| Field | `compute_vectors = true` | `compute_vectors = false` |
|---|---|---|
| `T` | $n \times n$ | $n \times n$ |
| `Q` | $n \times n$ | $0 \times 0$ (empty) |
| `eigvals` | $n$ | $n$ |
| `balance_scale` | $n$ (zeros if not balanced) | $n$ |
| `balance_perm`  | $n$ (identity if not balanced) | $n$ |
 
---
## 3. Balancing

### 3.1 Motivation

Balancing applies an invertible similarity transformation to the working matrix to improve stability of the subsequent eigenvalue computation. The transformation preserves eigenvalues but reduces the spread of off-diagonal entry magnitudes, which decreases the number of Francis QR steps needed for convergence and improves the accuracy of the computed eigenvalues.
The precise transformation is

$$A \leftarrow D^{-1} \cdot (P A P^T) \cdot D$$

where $P$ is a permutation matrix and $D = \mathrm{diag}(2^{s_0}, \ldots, 2^{s_{n-1}})$ is a diagonal scaling with integer power-of-2 entries. The exponents $s_i$ are stored in `balance_scale[i]` and the permutation is recorded in `balance_perm[i]`.

### 3.2 Algorithm
- **Phase 1: permutation.** Here we identify indices whose corresponding row and column are both entirely zero within the active window $[\mathtt{ilo}, \mathtt{ihi}]$ (except at the diagonal). Such an index $i$ contributes only the eigenvalue $A(i,i)$ - it is decoupled from all others - and can be permuted to the boundary without affecting the rest of the problem. The procedure runs two sweeps: initial sweep moves isolated indices to the **high** boundary. The subsequent sweep is symmetric, moving isolated indices to the **low** boundary by incrementing `ilo`. After both sweeps, only the submatrix $A[\mathtt{ilo}:\mathtt{ihi}+1, \mathtt{ilo}:\mathtt{ihi}+1]$ requires further reduction.
- **Phase 2: diagonal scaling.** For each active index $i \in [\mathtt{ilo}, \mathtt{ihi}]$ define the off-diagonal row and column 1-norms restricted to the active window:

    $$r_i = \sum_{\substack{j = \mathtt{ilo} \\ j \ne i}}^{\mathtt{ihi}} |A_{ij}|, \qquad
    c_i = \sum_{\substack{j = \mathtt{ilo} \\ j \ne i}}^{\mathtt{ihi}} |A_{ji}|.$$

    The ideal scaling that equates them is $f^* = \sqrt{c_i / r_i}$. The implementation approximates $f^*$ by the nearest integer power of $\mathrm{BASE} = 2$, computed as follows.
    Let $\rho = c_i / r_i$. The algorithm brings $\rho$ into the canonical interval $[1/4, 4)$ by repeatedly multiplying or dividing by $4 = \mathrm{BASE}^2$, tracking the corresponding power $f$:
    A single additional step handles the two half-intervals: if $\rho < 1/\mathrm{BASE}$ then $f /= \mathrm{BASE}$; if $\rho > \mathrm{BASE}$ then $f *= \mathrm{BASE}$. If $f = 1$ (within $10^{-14}$) or $f = 0$, no scaling is applied. Otherwise the similarity is executed as
    
    $$A(i, j) \leftarrow f \cdot A(i, j) \quad \forall\, j \qquad \text{(row $i$ scaled by $f$)}$$
    $$A(j, i) \leftarrow A(j, i) / f \quad \forall\, j \qquad \text{(column $i$ scaled by $1/f$)}$$
    
    and the log₂ exponent is accumulated: `scale[i] += log2(f)`.
    
    **Why powers of 2?** Multiplication by an integer power of 2 is exact in IEEE 754 floating-point: it changes only the exponent field of each mantissa. No rounding error is introduced by the scaling itself, so the transformed matrix entries are representable without additional error relative to the pre-scaling values.
    
    Up to `MAXP = 10` passes are performed; early exit occurs when no index was scaled in a complete pass (convergence of the equalization).

---
## 4. Hessenberg reduction

### 4.1 Upper Hessenberg form

A matrix $H \in \mathbb{C}^{n \times n}$ is **upper Hessenberg** if $H_{ij} = 0$ for all $i > j + 1$, that is, every entry strictly more than one position below the main diagonal is zero. The Hessenberg form has some properties of practical importance:

1. An upper Hessenberg matrix remains upper Hessenberg under unitary similarity transformations applied by the QR algorithm.
2. One Francis QR step on an $n \times n$ Hessenberg matrix costs $O(n^2)$ rather than the $O(n^3)$ that would be required for a full matrix.
Every square matrix can be reduced to upper Hessenberg form by a sequence of $n - 2$ unitary similarities.

### 4.2 Householder reduction

The deicated function, `hessenberg_reduce`, transforms the active submatrix $A[\mathtt{ilo}:\mathtt{ihi}+1, \mathtt{ilo}:\mathtt{ihi}+1]$ using $K = \mathtt{ihi} - \mathtt{ilo}$ Householder reflectors. For step $k = \mathtt{ilo}, \mathtt{ilo}+1, \ldots, \mathtt{ihi}-1$:

**Step 1: Reflector construction.** Let $\mathtt{len} = \mathtt{ihi} + 1 - (k+1)$ and extract the column subvector
$$x = A[k+1 : \mathtt{ihi}+1,\; k] \in \mathbb{C}^{\mathtt{len}}.$$

Compute the Householder reflector $H_k = I - \beta_k v_k v_k^H$ (via `detail::householder_reflector` from `qr.hpp`) such that
$$H_k x = -e^{i \arg(x_0)}\, \|x\|_2\, e_1.$$
 
The reflector satisfies $H_k^H = H_k$ and $H_k^2 = I$.

**Step 2: Left application** (`hess_left_apply`). Apply $H_k$ from the left to the active row band:
$$A[k+1:\mathtt{ihi}+1,\; k:n] \leftarrow H_k \cdot A[k+1:\mathtt{ihi}+1,\; k:n].$$
 
Expanding using $H_k = I - \beta_k v_k v_k^H$:
$$\text{for } j = k, \ldots, n-1: \quad A_{:, j}[k+1:\mathtt{ihi}+1] \mathrel{-}= \beta_k v_k \bigl(\bar{v}_k^T A_{:,j}[k+1:\mathtt{ihi}+1]\bigr).$$

This is implemented as a matrix–vector product followed by a rank-1 update: compute $w[j] = \bar{v}_k^T A[k+1:\mathtt{ihi}+1, k+j]$ for each $j$, then subtract $\beta_k v_k w^T$.

**Step 3: Right application** (`apply_householder_right` from `qr.hpp`). Apply $H_k$ from the right to maintain the similarity:
$$A[0:n,\; k+1:\mathtt{ihi}+1] \leftarrow A[0:n,\; k+1:\mathtt{ihi}+1] \cdot H_k.$$
 
Since $H_k = H_k^H$, this is equivalent to $A \leftarrow A - A v_k (\beta_k v_k)^H$, computed by calling GEMV and GERC.
 
**Step 4: Enforce zeros.** Set $A(i, k) \leftarrow 0$ for $i = k+2, \ldots, \mathtt{ihi}$ explicitly to prevent floating-point residuals from accumulating.

---

After all $K$ steps, $A$ is upper Hessenberg on the active window. The full accumulated unitary factor is $Q_H = H_{\mathtt{ilo}} H_{\mathtt{ilo}+1} \cdots H_{\mathtt{ihi}-1}$ and satisfies $Q_H^H A_0 Q_H = H$.

### 4.3 Blocked WY accumulation

The Hessenberg reduction itself is always performed sequentially (one reflector at a time). This is necessary: each reflector modifies the matrix before the next one can be constructed. Thus there is no BLAS-3 analogue for the application pattern at each step of Hessenberg reduction, making it inherently Level-2.

However, the $Q_H$ accumulation can be batched. The $K$ stored reflectors are grouped into panels of width `nb = HESS_BLOCK = 64`. For each panel $[\mathtt{kb}, \mathtt{kb} + \mathtt{nb})$:

1. **Build compact WY T-factor.** Call `detail::larft` (from `qr.hpp`) on the `nb` stored reflector vectors $\{v_{k_0}, \ldots, v_{k_0+\mathtt{nb}-1}\}$ and their $\beta$-values to produce the $\mathtt{nb} \times \mathtt{nb}$ upper-triangular matrix $\mathcal{T}$ such that
   $$H_{k_0} H_{k_0+1} \cdots H_{k_0+\mathtt{nb}-1} = I - V \mathcal{T} V^H$$
    where $V \in \mathbb{C}^{(\mathtt{ihi}-k_{\text{abs}}) \times \mathtt{nb}}$ is the column-stacked matrix of reflector vectors (offset-embedded: column $j$ of $V$ has $v_{k_0+j}$ starting at row $j$).
2. **Right-apply compact WY block to $Q_H$** via `hess_wy_q_update` (cf. the [next section](#44-blocked-right-application)). This updates the column band $Q_H[0:n, k_{\text{abs}}+1 : \mathtt{ihi}+1]$ in three GEMM calls.

### 4.4 Blocked right-application

Let $W_Q = Q_H[0:n,\; c_0 : c_0 + \ell]$ where $c_0 = k_{\text{abs}} + 1$ and $\ell = \mathtt{ihi} - k_{\text{abs}}$. The compact WY right-apply computes
$$W_Q \leftarrow W_Q (I - V \mathcal{T} V^H)^H = W_Q - W_Q V \mathcal{T}^H V^H,$$

but since [`larft`](/reference/QR.md#4-larft-compact-wy-t-matrix-construction) stores $\mathcal{T}$ and the right-application uses $\mathcal{T}^H$, the three-GEMM sequence in the implementation is:
$$C = W_Q \cdot V \qquad (n \times \mathtt{nb}, \text{ GEMM 1})$$
$$C_S = C \cdot \mathcal{T} \qquad (\mathtt{nb} \times \mathtt{nb} \text{ triangular, GEMM 2})$$
$$W_Q \leftarrow W_Q - C_S \cdot V^H \qquad (\text{GEMM 3})$$

Note the **contrast with `apply_wy_left` in QR.** The QR left-apply (`apply_wy_left`) updates a trailing submatrix $W$ as $W \leftarrow (I - V\mathcal{T}V^H) W$, which expands to $W - V\mathcal{T}(V^H W)$ and uses `hermitian(T_mat)` in its second GEMM. The sign difference arises because `larft` embeds $v_j$ at row offset $j$ in $V$, introducing a conjugate-transpose structural asymmetry that manifests in opposite ($\mathcal{T}$ vs $\mathcal{T}^H$) usage depending on the side of application.

---
## 5. Francis implicit single-shift QR

### 5.1 The shifted QR step

Given an upper Hessenberg matrix $H$ and a shift $\mu \in \mathbb{C}$, the **explicit** shifted QR step computes $H - \mu I = QR$ and sets $H \leftarrow RQ + \mu I = Q^H H Q$. This preserves the Hessenberg structure and drives the (1,0) subdiagonal entry toward zero at rate determined by $|\lambda_{n-1} - \mu| / |\lambda_n - \mu|$.

The **implicit** variant avoids forming $H - \mu I$ explicitly. It uses the ***implicit Q theorem***: any two unreduced upper Hessenberg matrices that are unitarily similar and share the same first column of their unitary factor are identical. Starting from the first column of $H - \mu I$ is sufficient to determine the entire step.

### 5.2 Initial bulge

For the active window $[p, q]$, `francis_step` initialises the step from the first column of $H - \mu I$ restricted to rows $p$ and $p+1$:
$$x = \begin{bmatrix} H(p,p) - \mu \\ H(p+1,p) \end{bmatrix} \in \mathbb{C}^2.$$
 
A length-2 Householder reflector $H_0 = I - \beta_0 v_0 v_0^H$ is computed such that $H_0 x = \|x\| e_1$. Apply $H_0$ as a similarity:
- **Left-apply** to $H[p:p+2, p:n]$: `apply_householder_left(H, v, beta, p, 2, nc_Hl)` where `nc_Hl = n - p`.
- **Right-apply** to $H[0:n, p:p+2]$: inline GEMV + rank-1 update with work buffer `wH[0:n]`.
- **Right-apply to $Q_{\text{iter}}[0:n, p:p+2]$** (if `accQ`): inline GEMV call, combined with rank-1 update with work buffer `wQ[0:n]`.
After the right-apply, $H(p+2, p)$ (which was previously zero by the Hessenberg structure) may become non-zero - this is the **bulge** at position $(p+2, p)$.

### 5.3 Bulge-chase loop

The bulge must be chased down the subdiagonal to restore Hessenberg structure. For $k = p, p+1, \ldots, q-2$:

After the $k$-th chase step, the bulge has migrated to position $(k+2, k)$. The entries $H(k+1, k)$ and $H(k+2, k)$ define the vector to be annihilated:
$$x_k = \begin{bmatrix} H(k+1, k) \\ H(k+2, k) \end{bmatrix}.$$

Compute the length-2 Householder reflector $v_k$ for $x_k$, then apply it as a similarity:

- **Left-apply** to $H[k+1:k+3, k:n]$, using the pre-allocated buffer `wH[0:nc]` (where `nc = n - k`):
 
$$\mathtt{wH}[j] = \overline{v_0}\, H(k+1, k+j) + \overline{v_1}\, H(k+2, k+j), \quad j = 0, \ldots, \mathtt{nc}-1$$
$$H(k+1, k+j) \mathrel{-}= \beta\, v_0\, \mathtt{wH}[j], \quad H(k+2, k+j) \mathrel{-}= \beta\, v_1\, \mathtt{wH}[j].$$

Set $H(k+2, k) \leftarrow 0$ unconditionally.

- **Right-apply** to $H[0:\mathtt{nr\_H}, k+1:k+3]$ (where `nr_H = q + 1`), reusing `wH[0:nr_H]`:

$$\mathtt{wH}[i] = H(i, k+1)\, v_0 + H(i, k+2)\, v_1, \quad i = 0, \ldots, \mathtt{nr\_H}-1$$
$$H(i, k+1) \mathrel{-}= \beta\, \mathtt{wH}[i]\, \overline{v_0}, \quad H(i, k+2) \mathrel{-}= \beta\, \mathtt{wH}[i]\, \overline{v_1}.$$

- **Right-apply to $Q_{\text{iter}}[0:n, k+1:k+3]$** (if `accQ`), using `wQ[0:n]`:

$$\mathtt{wQ}[i] = Q(i, k+1)\, v_0 + Q(i, k+2)\, v_1$$
$$Q(i, k+1) \mathrel{-}= \beta\, \mathtt{wQ}[i]\, \overline{v_0}, \quad Q(i, k+2) \mathrel{-}= \beta\, \mathtt{wQ}[i]\, \overline{v_1}.$$

After all $q - p$ chase steps, the bulge has been eliminated and $H$ is again upper Hessenberg on $[p, q]$.

---
## 6. Shift strategy and deflation

### 6.1 Wilkinson shift

`detail::qr_iteration` maintains a scalar `q` (initially `ihi`) that tracks the index of the current **deflation front** - the last unreduced diagonal entry. The outer loop runs while `q > ilo`.

The Wilkinson shift is chosen as one of the two eigenvalues of the trailing $2 \times 2$ submatrix
$$M = \begin{bmatrix} H(q-1,q-1) & H(q-1,q) \\ H(q,q-1) & H(q,q) \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}.$$

The eigenvalues of $M$ satisfy
$$\lambda_{1,2} = \frac{a + d}{2} \pm \sqrt{\!\left(\frac{a-d}{2}\right)^{\!2} + bc}. \tag{$*$}$$
 
The **Wilkinson shift** is whichever root $\lambda_i$ satisfies $|\lambda_i - d| \le |\lambda_{3-i} - d|$, i.e. the eigenvalue of $M$ closest to the bottom-right entry $d$. This choice is optimal for the convergence of the last subdiagonal entry $H(q, q-1)$ to zero: after one step the new $|H'(q, q-1)| \approx |H(q,q-1)| \cdot |\lambda_{\text{far}} - \mu| / |\lambda_{\text{near}} - \mu|$, which is small when $\mu \approx d$ and $|d - \lambda_{\text{near}}|$ is much smaller than $|d - \lambda_{\text{far}}|$.

The discriminant in ($*$) is computed directly in `std::complex<double>`. There is no branch-cut concern: `std::sqrt` on a complex argument always returns the principal square root and is continuous except on the negative real axis; since the discriminant is complex-valued in general (not confined to $\mathbb{R}$), no special-case handling is needed.

### 6.2 Exceptional shifts

Convergence stagnation occurs when the Wilkinson shift $\mu$ is close to an eigenvalue that does not lie near $d$, causing the iteration to cycle without deflation. After `EXCEPT_FREQ = 16` steps since the last deflation (`since_deflation % EXCEPT_FREQ == 0` and `since_deflation > 0`), an **exceptional shift** is applied instead:
```cpp
const double scale = std::abs(H(q,q)) + std::abs(H(q,q-1));
mu = DefaultScalar(scale * udist(rng), scale * udist(rng));
```
The shift is a random complex number whose real and imaginary parts are drawn uniformly from $[-\mathtt{scale}, +\mathtt{scale}]$ where $\mathtt{scale} = |H(q,q)| + |H(q,q-1)|$ is a local spectral radius estimate. The random number generator is `mt19937_64` with the fixed seed set to ensure reproducible behaviour across runs.

---
## 7. Schur vector accumulation

The complete unitary matrix $Q$ satisfying $A = Q T Q^H$ is the product of two independently accumulated factors:
$$Q = Q_H \cdot Q_{\text{iter}},$$

where $Q_H$ is the accumulated [Hessenberg reduction](#4-hessenberg-reduction) factor and $Q_{\text{iter}}$ is the accumulated QR iteration factor (cf. the [discussion](#5-francis-implicit-single-shift-qr)).

`hessenberg_reduce` initialises $Q_H = I_n$ when `accumulate_q=true`, then applies each panel's WY block from the right via `hess_wy_q_update`. The invariant after stage 2 is $A_0 = Q_H H Q_H^H$.
 
At the start of stage 3, the driver initialises `Q_iter = Matrix<DefaultScalar, L>::identity(n)`. Each right-application inside `francis_step` updates two columns of `Q_iter`. The guard `accQ = (Q_iter.rows() == n)` enables this path. After stage 3, $H = Q_{\text{iter}} T Q_{\text{iter}}^H$.

The driver composes the two factors via a single GEMM call:
```cpp
Matrix<DefaultScalar, L> Q_full(n, n, DefaultScalar(0));
gemm(DefaultScalar(1), expr(Q_hess), expr(Q_iter), DefaultScalar(0), Q_full);
res.Q = std::move(Q_full);
```
This costs $O(n^3)$ multiplies - the same order as the preceding stages. For large $n$ it may be the dominant constant in the full-vectors path.

*Note:* When `compute_vectors` is set to `false`:
- `Q_iter` is left empty. The `accQ` guard suppresses all right-applications to `Q_iter` in `francis_step` (two GEMV and two AXPY pairs per bulge-chase step, totalling $\approx 4n$ multiplies per sub-step and $\approx 4n(q-p)$ per Francis step).
- The final GEMM $Q_H \cdot Q_{\text{iter}}$ is skipped entirely.
- The blocked WY right-apply in `hessenberg_reduce` (`hess_wy_q_update`) is also skipped (but the initialisation `Q_hess = identity(n)` does still execute, as `Q_hess` is always passed to `hessenberg_reduce`).

In total, passing `compute_vectors=false` eliminates the dominant $O(n^2)$ work per Francis step and the $O(n^3)$ composition GEMM, approximately halving the total wall time for the eigenvalue-only path.