# QR decomposition: reference
> **Source file:** `qr.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`), `matrix_norms.hpp`.

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundation)
- [Construction of Householder reflectors](#2-householder-reflector-stable-construction)
- [Application of the reflectors](#3-applying-reflectors)
- [`larft`: WY matrix construction](#4-larft-compact-wy-t-matrix-construction)
- [Blocked trailing update](#5-apply_wy_left-blocked-trailing-update)
- [Two factorisation paths](#6-factorisation-paths)
- [$Q$ matrix accumulation](#7-accumulation-of)
- [Public API](#8-public-api)

---
## 0. Preamble and notation

Matrices are $m \times n$ with elements in field $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. They are written either as $A_{ij}$ or as $A(i,j)$ using 0-based indexing throughout, matching C++ array conventions. Unless stated otherwise, $m \geq n$ (tall or square matrices); the factorisation is defined for $m < n$ as well, but the rank-revealing properties apply only when $m \geq n$.

| Symbol | Meaning |
|---|---|
| $\|A\|_F$ | Frobenius norm $\sqrt{\sum_{ij} \|A_{ij}\|^2}$. |
| $\|A\|_2$ | Spectral norm (largest singular value). |
| $\|A\|_\infty$ | Maximum absolute row sum. |
| $\kappa(A)$ | Condition number $\|A\|_2\|A^{-1}\|_2$. |
| $k$ | $k = \min(m, n)$ throughout .|
| $n_b$ | Block size `QR_BLOCK = 64`. |
| $\varepsilon$ | Machine epsilon ($\approx 2.2 \times 10^{-16}$ for `double`). |
| $v^H$ | Conjugate transpose of $v$ .|
| $I_p$ | Identity matrix of size $p \times p$. |

**Packed vs. unpacked representation.** [Unlike LU](/reference/LU.md#0-preamble-and-notation), QR algorithm under consideration does not use a packed single-matrix representation. The Householder vectors and their $\beta$ coefficients are stored in `std::vector<Vector<T>> vs` and `std::vector<double> betas` (local to the factorisation), and the working matrix $W$ gradually accumulates the upper triangular factor $R$ in its upper triangle while the subdiagonal positions are conceptually zeroed (explicitly set to `T(0)` after each reflector application). The public result bundles separate `Q` and `R` matrices.

**Permutation convention.** The QR pivot vector encodes a *column permutation*: `piv[j]` is the index of the original column that was moved to position $j$ after all swaps. The relationship is $AP = QR$, where $P$ is the $n \times n$ column permutation matrix with $P_{\text{piv}[j],\, j} = 1$. This is the opposite convention from LU, which encodes a row permutation satisfying $PA = LU$.

---
## 1. Mathematical foundations

### 1.1 Existence

There are several methods of performing the factorisation, such as the Gram–Schmidt process, Givens rotations, or Householder transformations. Each has a number of advantages and disadvantages.

- **Via Gram-Schmidt.** The classical proof constructs $Q$ and $R$ simultaneously. Given $A = [a_0 \mid a_1 \mid \cdots \mid a_{n-1}]$ with $m \geq n$, the modified Gram-Schmidt process computes orthonormal vectors $q_0, \ldots, q_{k-1}$ (where $k = \operatorname{rank}(A)$). At step $j$ the algorithm is as follows:

    1. Set $u_j = a_j - \sum_{i < j} \langle a_j, q_i \rangle q_i$ (subtract projections onto previously computed basis vectors).
    2. If $\|u_j\| > 0$, set $q_j = u_j / \|u_j\|$ and $R_{jj} = \|u_j\|$.
    3. Set $R_{ij} = \langle a_j, q_i \rangle$ for $i < j$.

    The upper-triangular $R$ encodes all inner products; its diagonal entries are the successive norms of the residual vectors. When $A$ has full column rank, all $R_{jj} > 0$ and $Q = [q_0 \mid \cdots \mid q_{n-1}]$ forms an orthonormal basis for the column space of $A$, giving $A = QR$ with $Q \in \mathbb{R}^{m \times n}$, $R \in \mathbb{R}^{n \times n}$.

    Though easily implemented, the Gram-Schmidt process is notoriously unstable.

- **Via Householder reflectors.** An alternative and numerically superior construction applies a sequence of unitary matrices from the left to progressively triangularise $A$:

    $$H_{k-1} \cdots H_1 H_0\, A = R$$

    Each $H_j$ is a Householder reflector chosen to zero all entries of column $j$ below the diagonal. Since each $H_j$ is unitary ($H_j^H H_j = I$), the product $Q = H_0^H H_1^H \cdots H_{k-1}^H$ is also unitary and $A = QR$. This construction is the one implemented in `qr.hpp` and is described in great detail.

**Existence for any $A$.** Neither construction requires $A$ to have full rank. If $A$ has rank $r < \min(m,n)$, at least one of the residuals $u_j$ is zero, corresponding to a zero diagonal entry $R_{jj} = 0$. The factorisation still produces valid $Q$ and $R$ (with $R$ having at least one zero on the diagonal), but the columns of $Q$ spanning the null contributions may be chosen arbitrarily - uniqueness is lost in those degrees of freedom.

### 1.2 Householder reflectors

**Definition.** A Householder reflector is a matrix of the form

$$H = I - \beta v v^H, \qquad \beta = \frac{2}{v^H v} \in \mathbb{R}_{> 0}$$

The reflector has some properties of practical importance:

- **Unitarity.** $H$ is self-adjoint ($H^H = H$, since $\beta$ is real and $v v^H$ is Hermitian) and an involution ($H^2 = I$):

    $$H^2 = (I - \beta v v^H)^2 = I - 2\beta v v^H + \beta^2 v (v^H v) v^H = I - 2\beta v v^H + \beta (v^H v) \cdot \beta (v v^H) = I - 2\beta v v^H + 2\beta v v^H = I$$

    using $\beta (v^H v) = 2$. Unitarity follows: $H^H H = H^2 = I$.

- **Norm preservation.** Since $H$ is unitary, $\|Hx\|_2 = \|x\|_2$ for all $x$. The action of $H$ on a vector $x$ can be described geometrically: $H$ reflects $x$ about the hyperplane $\{z : v^H z = 0\}$, mapping $x$ to $x - \beta (v^H x) v$.

---
**Zeroing action of the operator.** Given $x \in \mathbb{F}^n$, we want $H$ such that $Hx = \sigma e_0$ for some scalar $\sigma$, where $e_0 = (1, 0, \ldots, 0)^\top$. Setting $H = I - \beta v v^H$ and requiring $Hx = x - \beta(v^H x)v = \sigma e_0$, we need $v \parallel (x - \sigma e_0)$. Set $v = x - \sigma e_0$. Then:

$$Hx = x - \frac{2 v^H x}{v^H v} v$$

Substituting $v = x - \sigma e_0$ and $v^H x = \|x\|_2^2 - \bar{\sigma} x_0$ (expanding $v^H = x^H - \bar{\sigma} e_0^H$), the requirement $Hx = \sigma e_0$ imposes:

$$x - \frac{2(\|x\|_2^2 - \bar{\sigma} x_0)}{\|x - \sigma e_0\|_2^2}(x - \sigma e_0) = \sigma e_0$$

This is satisfied when $|\sigma| = \|x\|_2$ (norm preservation). The free choice is the **phase** of $\sigma$: any unit complex $\phi$ gives $\sigma = \phi\|x\|_2$ and a valid reflector. For real $x$, $\sigma = \pm\|x\|_2$; for complex $x$, $\sigma = -\phi\|x\|_2$ where $\phi = x_0/|x_0|$ is chosen with a view to maximise $\|v\|_2$ and avoid cancellation.

---
**Computing $\beta$ from $v$.** With $v = x - \sigma e_0$:

$$v^H v = \|x\|_2^2 - \bar{\sigma} x_0 - \sigma \bar{x}_0 + |\sigma|^2 = 2\|x\|_2^2 - 2\operatorname{Re}(\bar{\sigma} x_0)$$

For real types with $\sigma = -\|x\|_2 \cdot \operatorname{sign}(x_0)$: $v^H v = 2\|x\|_2^2 + 2|x_0|\|x\|_2 = 2\|x\|_2(\|x\|_2 + |x_0|)$, and $\beta = 1/(\|x\|_2(\|x\|_2 + |x_0|))$.

The implementation computes this as `2.0 / vHv` where `vHv = std::norm(v[0]) + sigma_sq` - equivalent that avoids the intermediate $\|x\|_2$ in the denominator.

---
### 1.3 QR algorithm's triangularisation

The factorisation proceeds by applying $k = \min(m,n)$ reflectors. After step $j$, the working matrix $W^{(j)}$ has zeros in all positions $(i, j')$ with $i > j'$ for $j' < j$. The inductive invariant is: 

$$W^{(j)} = H_{j-1} \cdots H_0\, A = \begin{bmatrix} R_{11} & R_{12} \\ 0 & A_{22}^{(j)} \end{bmatrix}$$

where $R_{11} \in \mathbb{F}^{j \times j}$ is upper triangular and $A_{22}^{(j)} \in \mathbb{F}^{(m-j) \times (n-j)}$ is the unreduced trailing submatrix.

**Step $j$.** Extract $x = W^{(j)}[j:m, j]$ (column $j$ from row $j$ downwards). Construct $H_j$ to map $x \mapsto \sigma_j e_0$, embed it as $\hat{H}_j = \operatorname{diag}(I_j, H_j) \in \mathbb{F}^{m \times m}$ (identity on the first $j$ rows, $H_j$ on the rest). Then:

$$W^{(j+1)} = \hat{H}_j W^{(j)}$$

The first $j$ rows are unchanged (multiplied by $I_j$). In the remaining $(m-j)$ rows, column $j$ becomes $(\sigma_j, 0, \ldots, 0)^\top$ and columns $j+1, \ldots, n-1$ receive the trailing update. After $k$ steps, $W^{(k)}$ is upper triangular (or upper trapezoidal for $m > n$), giving $R = W^{(k)}$ and:

$$Q^H = H_{k-1} \cdots H_0, \qquad Q = H_0^H \cdots H_{k-1}^H = H_0 \cdots H_{k-1}$$

(using $H_j^H = H_j$ since each reflector is self-adjoint).

### 1.4 Operation count. 

At step $j$, the trailing update $\hat{H}_j W^{(j)}$ affects an $(m-j) \times (n-j)$ submatrix. The GEMV computing $w = W_\text{sub}^\top \bar{v}$ costs $2(m-j)(n-j)$ flops and the GER update costs the same, giving $4(m-j)(n-j)$ flops per step. Summing:

$$\text{Total} = 4\sum_{j=0}^{k-1} (m-j)(n-j) \approx 2mn^2 - \frac{2n^3}{3} \quad (m \geq n)$$

The $Q$ accumulation costs an additional $\approx 4m^2 n - \frac{4mn^2}{3} + \frac{2n^3}{3}$ flops (reduced QR). Both counts are $O(mn^2)$, roughly twice the cost of the LU factorisation of the same matrix.

---
## 2. `householder reflector`: stable construction

The three output modes differ in the shapes of $Q$ and $R$:
| Mode | `QRMode` value | $Q$ shape | $R$ shape | Use case |
|---|---|---|---|---|
| Reduced | `Reduced` | $m \times k$, $k = \min(m,n)$ | $k \times n$ | Economy; standard for least-squares. |
| Complete | `Complete` | $m \times m$ | $m \times n$ | Full orthonormal basis of $\mathbb{F}^m$. |
| R-only | `R` | Not formed. | $k \times n$ | Triangular factor only; fastest route. |

For the column-pivoted variant the factorisation satisfies $A \cdot P = QR$ and the diagonal of $R$ is (at least) non-increasing in absolute value: $|R_{00}| \geq |R_{11}| \geq \cdots \geq |R_{k-1,k-1}|$.

---
The dedicated function signature is as follows:
```cpp
template<typename T>
std::pair<Vector<T>, double> householder_reflector(const Vector<T>& x);
```

Returns `(v, beta)` such that $(I - \beta v v^H) x = \alpha e_0$ where $\alpha$ is [discussed below](#21-sign-convention).

### 2.1 Sign convention

- **Real $T$:** $\alpha = -\operatorname{sign}(x[0]) \cdot \|x\|_2$. The sign is chosen to **maximise** $|x[0] - \alpha|$, which maximises $\|v\|_2$, which minimises $\beta$ in turn. A small $v[0]$ would cause $\beta$ to be large and introduce rounding error when $\beta v v^H$ is formed. Choosing $\alpha$ opposite in sign to $x[0]$ gives $v[0] = x[0] - \alpha$ with magnitude $|x[0]| + \|x\|_2$ - the maximum possible - guaranteeing $v[0] \neq 0$.
 
- **Complex $T$:** $\alpha = -\phi\|x\|_2$ where $\phi = x[0]/|x[0]|$ is the unit-magnitude phase of $x[0]$. This ensures $Hx = \|x\|_2 e_0$ is real and positive result, consistent with the LAPACK convention for complex Householder reflectors.

### 2.2 Computing $\|x\|_2$ without overflow
```cpp
double sigma = 0.0;
for (size_t i = 1; i < n; ++i) sigma += std::norm(x[i]);
const double xnorm = std::sqrt(std::norm(x[0]) + sigma);
```

`std::norm(z)` returns $|z|^2$ for complex $z$ and $z^2$ for real $z$ - it does **not** call `std::abs`. Squaring directly (rather than the two-pass scaled algorithm used in `nrm2`) is acceptable here because the input $x$ is a column of the working matrix $W$, whose entries are bounded in magnitude by the initial matrix norm times the accumulated Householder growth factor, which for well-conditioned inputs does not cause overflow. The scaled algorithm would add an unnecessary full pass over the column inside an already $O(m)$ construction.

---
## 3. Applying reflectors

The responsible function is:
```cpp
template<typename T, Layout L>
void apply_householder_left(Matrix<T, L>& W, const Vector<T>& v, double beta,
                            size_t k, size_t len, size_t ncols);
```

It applies $H_k = I - \beta v v^H$ to the submatrix $W[k:k+\text{len},\; k:k+\text{ncols}]$ from the left:

$$W_\text{sub} \leftarrow H_k W_\text{sub} = W_\text{sub} - \beta v \underbrace{(v^H W_\text{sub})}_{w^\top}$$

In BLAS terminology, this is a rank-1 update $W_\text{sub} \mathrel{-}= \beta v w^\top$ where $w = W_\text{sub}^\top \bar{v}$ (for real types $\bar{v} = v$, so $w = W_\text{sub}^\top v$).

### 3.1 Forming of $w = W_\text{sub}^H \bar{v}$
The conjugated reflector vector is materialised first:
```cpp
Vector<T> cv(len);
LINALG_VECTORIZE
for (size_t i = 0; i < len; ++i) cp[i] = linalg::conj(vp[i]);
```

Then a GEMV is called with a **layout-flipped** kernel to compute $w = W_\text{sub}^H \bar{v}$:
```cpp
if constexpr (L == Layout::RowMajor)
    kernels::gemv_kernel_col(T(1), W_base, lda, cv.data(), 1, T(0), w.data(), ncols, len);
else
    kernels::gemv_kernel_row(T(1), W_base, lda, cv.data(), 1, T(0), w.data(), ncols, len);
```

The layout flip is **justified by the following argument.** Let `W_base` point to element $W[k,k]$, and let the logical submatrix $W_\text{sub}$ have shape $\text{len} \times \text{ncols}$ with leading dimension `lda`. For `RowMajor` mode of storage, element $(i,j)$ is at offset $i \cdot \text{lda} + j$. Interpreting the **same** memory block as a `ColMajor` matrix $M$ of shape $\text{ncols} \times \text{len}$ with the **same** `lda`, element $M(j,i)$ is at offset $j \cdot \text{lda} + i$ - the identical address. Therefore $M = W_\text{sub}^\top$ exactly: the reinterpreted `ColMajor` matrix is the transpose of the original `RowMajor` submatrix.

`gemv_kernel_col` computes $M u$ for a `ColMajor` matrix $M$ of shape $\text{rows\_M} \times \text{cols\_M}$, where here $\text{rows\_M} = \text{ncols}$ and $\text{cols\_M} = \text{len}$, and the input is `cv` $= \bar{v}$. The result is:

$$w = M\,\bar{v} = W_\text{sub}^\top \bar{v}, \qquad w_j = \sum_{i=0}^{\text{len}-1} W_\text{sub}[i,j]\,\overline{v_i}$$

The subsequent `ger` step applies $W_\text{sub} \mathrel{-}= \beta\, v\, w^\top$, i.e. adds $-\beta\, v_i\, w_j$ to $W_\text{sub}[i,j]$.

Substituting:

$$-\beta\, v_i\, w_j = -\beta\, v_i \sum_{i'} W_\text{sub}[i',j]\,\overline{v_{i'}}$$

Summing over the `ger` outer product, the net update to $W_\text{sub}[i,j]$ is $-\beta\, v_i \,(W_\text{sub}^\top \bar{v})_j$, so:

$$W_\text{sub} \leftarrow W_\text{sub} - \beta\, v\,(W_\text{sub}^\top \bar{v})^\top = W_\text{sub} - \beta\, v\, v^H W_\text{sub}$$

The last equality uses $(W_\text{sub}^\top \bar{v})^\top = \bar{v}^\top W_\text{sub} = v^H W_\text{sub}$. This matches $(I - \beta v v^H)W_\text{sub}$ exactly, confirming correctness for both real and complex $T$.

### 3.2 `ger` update
```cpp
MatrixView<T, L, false, false, true> W_sub(W_base, len, ncols, lda);
ger(-b, v, w, W_sub);
```

This calls the unconjugated `ger` (not `gerc`), applying $W_\text{sub} \mathrel{-}= \beta v w^\top$. The conjugation required for the Hermitian update was already absorbed into $w$ via the $\bar{v}$ input to `gemv` [earlier](#31-forming-of).

**Why not `gemm` here?** For a single reflector, the rank-1 `ger` cost is $O(\text{len} \cdot \text{ncols})$ - the same asymptotic cost as GEMM. However, `ger` avoids the temporary allocation required for GEMM and is lower-latency for the narrow panels (small `ncols`, up to `QR_BLOCK = 64`) typical in the unblocked panel factorisation. The blocked WY path promotes to `gemm` exactly when it matters: for the wide trailing update where `ncols = n - k - n_b` is large.

---
## 4. `larft`: compact WY T-matrix construction
```cpp
template<typename T, Layout L>
Matrix<T, L> larft(const std::vector<Vector<T>>& vs,
                   const std::vector<double>& betas,
                   size_t k, size_t n);
```

The compact WY representation expresses the product of $n_b$ consecutive Householder reflectors starting at block-column $k$ as a single rank-$n_b$ update:

$$H_k H_{k+1} \cdots H_{k+n_b-1} = I - V T V^H$$

where $V \in \mathbb{F}^{(m-k) \times n_b}$ has the reflector vectors as columns (with $V_{ij} = \texttt{vs}[k+j][i]$ for $i \geq j$, zero above) and $T \in \mathbb{F}^{n_b \times n_b}$ is upper triangular.
 
`larft` builds $T$ column by column.
 
**Column $j = 0$:** $T[0,0] = \beta_k$. The first reflector has no predecessors.

**Column $j > 0$:** The recurrence:
 
$$T[0:j,\; j] = -\beta_{k+j} \cdot T[0:j,\; 0:j] \cdot (V_{0:j}^H\, v_j)$$
 
Implemented in two sub-steps.
 
* *Sub-step 1: inner products.* For each $l \in \{0, \ldots, j-1\}$, compute
 
    $$z_l = -\beta_{k+j} \sum_{ii=0}^{\mathrm{len}_j - 1} \overline{v_l[j - l + ii]}\; v_j[ii]$$
 
    where $v_l = \texttt{vs}[k+l]$ has length $m - (k+l)$ and $v_j = \texttt{vs}[k+j]$ has length $m - (k+j) = \text{len}_j$. The offset $j - l$ aligns the two vectors: $v_l$ is zero in its first $j - l$ entries (the reflector was applied at column $k+l$, so rows $0$ through $k+l-1$ are untouched), while $v_j$ starts at row $k+j$. The inner product therefore begins at the row where $v_j$ starts.

* *Sub-step 2: triangular matrix-vector product.* Apply the already-computed upper-triangular $T[0:j, 0:j]$ to $z$:
 
$$T[0:j,\; j] = T[0:j,\; 0:j] \cdot z$$
 
```cpp
for (size_t l = 0; l < j; ++l) {
    T acc = T(0);
    for (size_t ll = l; ll < j; ++ll) acc += T_mat(l, ll) * z[ll];
    T_mat(l, j) = acc;
};
```
---
**The $T^H$ convention in `apply_wy_left`.** Despite `larft` building the mathematically correct $T$ (satisfying $H_k \cdots H_{k+n_b-1} = I - VTV^H$), `apply_wy_left` passes `hermitian(T_mat)` to `gemm`, computing $T^H C$ instead of $TC$. The `hermitian` call compensates for a certain embedding convention, recovering the correct trailing update. Cf. the detailed treatise [here](#51-implementation).
 
*Note:* `larft` runs serially and costs $O(n_b^2 m)$. For `QR_BLOCK = 64` and typical $m$, this is $O(4096m)$ - negligible compared to the $O(n_b m (n - k - n_b))$ cost of the `apply_wy_left` trailing update that follows.

---
## 5. `apply_wy_left`: blocked trailing update
```cpp
template<typename T, Layout L>
void apply_wy_left(Matrix<T, L>& W, const std::vector<Vector<T>>& vs,
                   const Matrix<T, L>& T_mat,
                   size_t k, size_t nb, size_t j0);
```

Applies the compact WY block to the trailing submatrix $W[k:m,\; j_0:n]$ where $j_0 = k + n_b$. The intended compact WY identity is:

$$W_\text{trail} \leftarrow (I - VTV^H)\,W_\text{trail} = W_\text{trail} - V\bigl(T\,(V^H W_\text{trail})\bigr)$$

### 5.1 Implementation
The $V$ matrix built inside `apply_wy_left` uses an offset-row embedding: column $j$ of $V$ has `vj[0]` placed at row $j$ of $V$ (not row 0), making $V$ lower trapezoidal. This embedding means taht the effective matrix applied via the three `gemm` calls is $I - V S V^H$ where $S = T^H$ - the conjugate transpose of `larft`'s output. The `hermitian(T_mat)` call therefore recovers the correct $T = S^H$ and makes the three-GEMM sequence apply precisely $I - VTV^H$.

This is confirmed empirically: the reconstruction error $\|A - QR\|_\infty$ is $O(\varepsilon)$ with `hermitian(T_mat)` and $O(1)$ with `expr(T_mat)`.
 
The three multiplications as executed:

$$C = V^H W_\text{trail}, \qquad TC = T^H C, \qquad W_\text{trail} \mathrel{-}= V \cdot TC$$

which computes $W_\text{trail} - V T^H V^H W_\text{trail} = (I - VT^HV^H) W_\text{trail}$. Since the effective factor applied by the offset-embedded $V$ is $I - VS V^H$ with $S = T^H$, this is $(I - VTV^H)W_\text{trail}$ as required.
 
```cpp
// GEMM 1: C = V^H * W_trail  (nb * ncols_trail).
gemm(T(1), hermitian(V), expr(W_trail), T(0), C);
// GEMM 2: TC = T^H * C  (nb * ncols_trail).
gemm(T(1), hermitian(T_mat), expr(C), T(0), TC);
// GEMM 3: W_trail -= V * TC.
gemm(-T(1), expr(V), expr(TC), T(1), W_trail);
```

The intermediate matrices $C$ (shape $n_b \times \text{ncols\_trail}$) and $TC = T^H C$ (same shape) fit comfortably in L2 cache for `QR_BLOCK = 64` and typical $n$. Reusing them across the two subsequent GEMMs achieves Level-3 arithmetic intensity: each GEMM's ratio of floating-point operations to memory transfers is $O(n_b)$, vs. $O(1)$ for a sequence of $n_b$ individual rank-1 `ger` calls that would re-read $W_\text{trail}$ for every reflector separately.

### 5.2 The $V$ matrix layout

$V$ is built from the panel reflectors:

```cpp
Matrix<T, L> V(len, nb, T(0)); // len = m - k, nb = n_b.
for (size_t j = 0; j < nb; ++j) {
    const Vector<T>& vj = vs[k + j];
    for (size_t i = 0; i < vj.size(); ++i) V(i + j, j) = vj[i];
};
```

Column $j$ of $V$ holds reflector $v_{k+j}$, zero-padded above row $j$ (reflecting the fact that $H_{k+j}$ acts only on rows $k+j$ and below). The offset `i + j` places the first non-zero entry at row $j$ of $V$, so $V$ is lower-trapezoidal.
 
### 5.3 Extract-and-writeback pattern

$W_\text{trail}$ is extracted into a temporary `Matrix<T,L>` before the `gemm` calls and written back afterwards:
```cpp
Matrix<T, L> W_trail(len, ncols_trail);
parallel_for(len, copy_thresh, [&](size_t rs, size_t re) {
    for (size_t i = rs; i < re; ++i)
        for (size_t jj = 0; jj < ncols_trail; ++jj)
            W_trail(i, jj) = W(k + i, j0 + jj);
});
// ...Multiplications on on W_trail...
parallel_for(len, copy_thresh, [&](size_t rs, size_t re) {
    for (size_t i = rs; i < re; ++i)
        for (size_t jj = 0; jj < ncols_trail; ++jj)
            W(k + i, j0 + jj) = W_trail(i, jj);
});
```

The copy is necessary because the `gemm` accumulation target must have a unit leading dimension for the `raw_mat_info` fast path in `gemm`. A `MatrixView` subblock of $W$ has `lda = W.cols()` (the full working matrix width), which is valid but wider than `ncols_trail`; the `gemm` blocked tiling relies on stride exactly equalling the column count for optimal cache behaviour. The extract step normalises the stride and allows the blocked GEMM kernel to operate without stride indirection.

The extract and writeback are parallelised with threshold `max(1, PARALLEL_THRESHOLD_SIMPLE / (ncols_trail + 1))`, ensuring each thread handles at least one row.

---
## 6. Factorisation paths

The `qr()` master function dispatches to one of two paths depending on whether pivoting is requested and whether $K > n_b$.

### 6.1 Blocked WY path (non-pivoted, $K > 64$)
```
for k = 0, n_b, 2*n_b, etc.:
    nb = min(QR_BLOCK, K - k)
    Panel factorisation: apply n_b reflectors to W[k:m, k:k+nb]  [Level 2]
    larft: build T from the nb panel reflectors                  [O(nb^2 * m)]
    apply_wy_left: one blocked BLAS-3 trailing update
```

**Asymptotic efficiency.** Level-2 panel work per block is $O(n_b \cdot (m - k) \cdot n_b) = O(n_b^2 m)$. The BLAS-3 trailing update is $O(n_b \cdot (m - k) \cdot (n - k - n_b))$. For large $m$, $n$ and small $n_b$, the fraction of work in `gemm` calls approaches $1 - O(n_b / n)$. With `QR_BLOCK = 64` and $n \gg 64$, essentially all arithmetic is concentrated in the multiplication.

**Panel-only application.** Within the panel factorisation loop, `apply_householder_left` is called with `ncols = nb - j` (only the remaining panel columns), not the full trailing width. This defers the trailing update to the single BLAS-3 WY call, minimising redundant work.

### 6.2 Unblocked Level-2 path (pivoted, or $K \leq 64$)

Each step $k$ performs:

1. **Pivot selection** (pivoted only): scan `col_norms[k:n]`, swap the maximum-norm column into position $k$.
2. **Rank check** (pivoted only): if $\sqrt{\texttt{col\_norms}[k]} \leq \tau$, break.
3. **Reflector construction**: `householder_reflector` on `W[k:m, k]`.
4. **Application to trailing submatrix**: `apply_householder_left(W, v, beta, k, m-k, n-k)`.
5. **Zero sub-diagonal**: `W(i, k) = T(0)` for $i > k$.
6. **Incremental norm update** (pivoted only): update `col_norms[k+1:n]`.

The compact WY approach defers the trailing update: it accumulates $n_b$ reflectors and applies them all at once. With column pivoting, the selection at step $k+1$ depends on the norms of columns after $H_k$ has been applied. It is therefore impossible to defer the application of $H_k$ - the norm update must be computed immediately. **Pivoting is fundamentally incompatible with the blocked WY approach.**

### 6.3 Incremental norm update

After applying $H_k$ at step $k$, the squared residual norm of column $j > k$ decreases by $|W[k,j]|^2$:

$$\|r_j^{(k+1)}\|^2 = \|r_j^{(k)}\|^2 - |W[k,\,j]|^2$$

This is the Pythagorean identity: $H_k$ is an isometry, so $\|W[k:m,\,j]\|^2 = |W[k,j]|^2 + \|W[k+1:m,\,j]\|^2$ before the update, and after the update the first entry is zero, leaving $\|W[k+1:m,\,j]\|^2 = \|r_j^{(k)}\|^2 - |W[k,j]|^2$.

The incremental update is susceptible to catastrophic cancellation when $|W[k,j]|^2 \approx \|r_j^{(k)}\|^2$ (the column norm is almost entirely in row $k$). The recompute guard follows the LAPACK `DLAQP2` strategy:
```cpp
// ~sqrt(std::numeric_limits<double>::epsilon()):
static constexpr double RECOMPUTE_THRESH = 1.49e-8;
const double one_minus = 1.0 - rkj_sq / col_norms[j];
col_norms[j] = (one_minus > RECOMPUTE_THRESH)
    ? col_norms[j] * one_minus
    : detail::col_norm_sq(W, j, k + 1);
```

When `one_minus` is close to 0 (near-cancellation: the subtracted quantity nearly equals the current norm), a full recomputation from `W[k+1:m, j]` is performed. The threshold $\sqrt{\varepsilon}$ is the standard choice: it corresponds to the level at which the incremental update would lose more than half its significant bits.

---
## 7. Accumulation of $Q$

After `lu_factor`-style factorisation of $W$, the reflectors $H_0, H_1, \ldots, H_{K-1}$ are stored in `vs` and `betas`. The $Q$ matrix satisfies

$$Q = H_0 H_1 \cdots H_{K-1}$$
 
It is built by initialising $Q = I_{m \times q}$ and applying the reflectors in **reverse order**:
 
```cpp
Q = Matrix<T, L>(m, q_cols, T(0));
for (size_t i = 0; i < q_cols; ++i) Q(i, i) = T(1);
 
for (int ki = static_cast<int>(K) - 1; ki >= 0; --ki) {
    const size_t k = static_cast<size_t>(ki);
    if (betas[k] == 0.0) continue;
    const size_t len   = m - k;
    const size_t ncols = q_cols - k;
    detail::apply_householder_left(Q, vs[k], betas[k], k, len, ncols);
};
```

The invariant is that after applying $H_{j}, H_{j+1}, \ldots, H_{K-1}$ in reverse order (i.e., $H_{K-1}$ first, then $H_{K-2}$, and so on), the accumulated result equals $H_j H_{j+1} \cdots H_{K-1}$. Pre-multiplying by $H_{j-1}$ gives $H_{j-1} H_j \cdots H_{K-1}$. Applying $H_0$ last (in the iteration `ki = 0`) finalises $Q = H_0 H_1 \cdots H_{K-1}$ as required.
 
**The `ncols = q_cols - k` argument.** At step `ki = k`, $H_k$ acts on rows $[k, m)$ and therefore can only affect columns $[k, q\_\text{cols})$ of $Q$ (columns $0$ through $k-1$ of $Q$ have zeros in rows $[k, m)$ and are therefore unchanged by $H_k$). Passing `ncols = q_cols - k` avoids redundant work on those columns.

---
## 8. Public API
### 8.1 `qr()`: the entry point
```cpp
template<typename T, Layout L = Layout::RowMajor>
QRResult<T, L> qr(const Matrix<T, L>& A,
                  QRMode mode = QRMode::Reduced,
                  bool pivoting = false,
                  double tol = -1.0);
```

| Parameter | Type | Default | Description |
|---|------|---|---|
| `A` | `Matrix<T,L>` |  | Input matrix, copied into working storage. |
| `mode` | `QRMode` | `Reduced` | Controls shape of $Q$ and $R$. |
| `pivoting` | `bool` | `false` | Enables column pivoting for rank-revealing factorisation. |
| `tol` | `double` | `-1.0` | Rank-detection tolerance; negative falls back auto-tolerance. |

When `tol < 0`, the tolerance is defined as:

$$\tau = \max(m, n) \cdot \varepsilon \cdot \|A\|_\infty$$

This is the standard LAPACK default. Round-off errors in the Householder transformations scale as $\varepsilon \|A\|$ per step and accumulate over $O(\max(m,n))$ steps, giving this combined bound.

**`QRResult` object structure** is as follows:
```cpp
template<typename T, Layout LL>
struct QRResult {
    Matrix<T, LL> P; // Column permutation matrix (A * P = Q * R).
    Matrix<T, LL> Q; // Orthonormal factor (empty for R-only mode).
    Matrix<T, LL> R; // Upper triangular factor.
    Vector<size_t> piv; // Pivot vector (length n).
    int rank; // Estimated numerical rank (-1 if non-pivoted).
    bool pivoted; // Whether column pivoting was used.
};
```

For non-pivoted factorisations, `piv` is set to identity permutation, `P` is the identity matrix, and `rank = -1`.

### 8.2 Convenience wrappers
```cpp
// Economy QR, no pivoting.
template<typename T, Layout L>
QRResult<T, L> qr_reduced(const Matrix<T, L>& A);

// Full QR (Q is m * m), no pivoting.
template<typename T, Layout L>
QRResult<T, L> qr_complete(const Matrix<T, L>& A);

// R factor only, no Q, no pivoting - fastest.
template<typename T, Layout L>
QRResult<T, L> qr_r(const Matrix<T, L>& A);

// Column-pivoted rank-revealing QR.
template<typename T, Layout L>
QRResult<T, L> qr_pivoted(const Matrix<T, L>& A, double tol = -1.0);

// MatExpr overload.
template<typename T, Layout L, typename E>
QRResult<T, L> qr(const MatExpr<E>& e);
```
 
*Note:* the `MatExpr` overload materialises the expression into a `Matrix<T,L>` before calling the main entry point. It cannot accept a `MatrixView` directly because the factorisation is performed in-place on the working copy.