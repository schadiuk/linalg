# QR decomposition: reference
> **Source file:** `qr.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`), `matrix_norms.hpp`.

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundation)
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

