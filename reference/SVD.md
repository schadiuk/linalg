# Singular Value Decomposition: reference
> **Source files:** `bidiag.hpp`, `svd.hpp`.

> **Dependees:** `qr.hpp` (reuses [Householder infrastructure](/reference/QR.md#12-householder-reflectors) and other routines).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundations)
- [Unblocked bidiagonalisation](#2-bidiagonalisation)
- [The problem of real diagonal](#3-realifying-the-diagonal)

---
## 0. Preamble and notation

Matrices are $m \times n$ with entries in field $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. Unless stated otherwise $m \geq n$ (the "tall" case); $m < n$ is reduced to this case by transposition.

| Symbol | Meaning |
|---|---|
| $k$ | $k = \min(m,n)$, the number of singular values / reflector pairs. |
| $n_b$ | Block size `BIDIAG_BLOCK = 64`. |
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

---
## 2. Bidiagonalisation

The dedicated driver (`bidiag_tall_unblocked`) builds $A = U B V^H$ by applying $k$ left reflectors $H_0, \ldots, H_{k-1}$ (zeroing successive sub-diagonal column tails) interleaved with $k-1$ right reflectors $G_0, \ldots, G_{k-2}$ (zeroing successive super-superdiagonal row tails):

$$H_{k-1} \cdots H_0 \; A \; G_0 \cdots G_{k-2} = B, \qquad Q = H_0 \cdots H_{k-1}, \quad P = G_0 \cdots G_{k-2}, \qquad A = Q\,B\,P^H$$

### 2.1 Column step
With the working matrix $W^{(\text{col})}$ already reduced in its leading `col` rows/columns:

```cpp
const size_t len = m - col;
Vector<T> x(len);
for (size_t i = 0; i < len; ++i) x[i] = W(col + i, col);
auto [u, ubet] = householder_reflector(x);
us[col] = u; ubeta[col] = ubet;
apply_householder_left(W, u, ubet, col, len, n - col);
for (size_t i = col + 1; i < m; ++i) W(i, col) = T(0);
// Right reflector: zero W[col, col+2:n], acting on cols [col+1, n).
if (col + 1 < n) {
    const size_t rlen = n - (col + 1);
    Vector<T> y(rlen);
    for (size_t j = 0; j < rlen; ++j) y[j] = conj(W(col, col + 1 + j));
    auto [v, vbet] = householder_reflector(y);
    vs[col] = v; vbeta[col] = vbet;
    apply_householder_right(W, v, vbet, col + 1, rlen);
    for (size_t j = col + 2; j < n; ++j) W(col, j) = T(0);
};
```

The left half is exactly QR's [step $j$](/reference/QR.md#13-qr-algorithms-triangularisation): construct a reflector zeroing $W[\text{col}{+}1{:}m,\,\text{col}]$, apply it to the whole trailing width. The right half is new: it must zero the *row* $W[\text{col},\, \text{col}{+}2{:}n]$ instead of a column, while leaving $W[\text{col},\text{col}{+}1]$ untouched (that entry becomes $e_{\text{col}}$, the superdiagonal).

---
`householder_reflector` and `apply_householder_left` are built around *column* reflectors: given $x$, they find $H$ with $Hx = \alpha e_0$, a left-multiplication. To zero a **row** tail with the same machinery, note that right-multiplication by a reflector $G = I - \beta v v^H$ satisfies
 
$$(\text{row}) \cdot G = \bigl(G^H\, \text{row}^H\bigr)^H = \bigl(G\, \text{row}^H\bigr)^H \qquad (G \text{ is Hermitian})$$
 
so treating $y = \text{row}^H = \overline{\text{row}}$ (a column vector) and building $G$ from `householder_reflector(y)` gives $Gy = \alpha e_0$, hence $\text{row} \cdot G = (\alpha e_0)^H = \bar\alpha\, e_0^H$ - a row with $\bar\alpha$ in the leading position and zero elsewhere. This is exactly why `y[j] = conj(W(col, col+1+j))` is built before reflecting, and it has an immediate, easy-to-miss consequence, as `apply_householder_right`'s in-place update leaves $\bar\alpha$ at position $W(\text{col}, \text{col}{+}1)$. The unblocked code above never explicitly assigns that entry.

### 2.2 Finalisation

Both the unblocked and blocked paths converge on `bidiag_finalize_tall`, which:

1. Extracts the raw (possibly complex) diagonal/superdiagonal `draw`, `eraw` from the fully-reduced $W$.
2. Realifies them via `real_bidiag` ([cf. the next section](#3-realifying-the-diagonal)).
3. Accumulates $U$ by replaying the stored left reflectors **in reverse order** onto $I_{m \times k}$ - identical in structure to QR's [$Q$ accumulation](/reference/QR.md#7-accumulation-of-q):
```cpp
   for (ci = k; ci-- > 0; ) apply_householder_left(res.U, us[ci], ubeta[ci], ci, m - ci, k - ci);
```
4. Accumulates $V$ by replaying the stored right reflectors **in forward order**:
```cpp
   for (ci = 0; ci < vs.size(); ++ci) apply_householder_right(res.V, vs[ci], vbeta[ci], ci + 1, n - (ci + 1));
```
5. Folds in the phase correction from `real_bidiag` (complex `T` only).

---
## 3. Realifying the diagonal

### 3.1 Implementation
For complex `T`, the raw reduction above produces a complex bidiagonal $W_{\text{raw}}$: unlike LAPACK's `ZLARFG`, this library's [`householder_reflector`](/reference/QR.md#21-sign-convention) does not force $\alpha$ to be real, so $A = U_{\text{raw}} W_{\text{raw}} V_{\text{raw}}^H$ holds exactly, but $W_{\text{raw}}$'s diagonal/superdiagonal entries are, in general, complex numbers of the correct magnitude and an arbitrary phase. `real_bidiag` computes unit-modulus corrections that rotate $W_{\text{raw}}$ into a real, non-negative bidiagonal $B$, and `bidiag_finalize_tall` folds the compensating rotations into $U$, $V$ so that $A = U B V^H$ continues to hold exactly.
```cpp
using R = real_type_t<T>;
// Select unit-modulus factor x such that x * z is real nonnegative.
auto conj_phase = [](T z) -> T {
    const double m = std::abs(z);
    return (m == 0.0) ? T(1) : conj(z) / static_cast<R>(m);
};
for (size_t i = 0; i < k; ++i) {
    // dr[0] = 1 by convention; dl[i] must satisfy dl[i] * dr[i] * diag_original[i] real >= 0.
    const T val = dr[i] * diag[i];
    dl[i] = conj_phase(val);
    diag[i] = T(std::abs(val));
    if (i + 1 < k) {
        const T eval = dl[i] * super[i];
        dr[i + 1] = conj_phase(eval);
        super[i] = T(std::abs(eval));
    };
};
```

### 3.2 Compensating $U$, $V$ transforms

Let $D_L = \operatorname{diag}(d_{l,i})$, $D_R = \operatorname{diag}(d_{r,i})$ (both unitary diagonal), and let $B$ be the realified bidiagonal obtained via the procedure above, so that entrywise: $$B_{ii} = d_{l,i} d_{r,i} \, (W_{\text{raw}})_{ii}$$ and $$B_{i,i+1} = d_{r,i+1} d_{l,i} \, (W_{\text{raw}})_{i,i+1}$$ 

I.e. $B = D_L W_{\text{raw}} D_R^H$. Seek diagonal unitary $X = \operatorname{diag}(x_i)$, $Y = \operatorname{diag}(y_i)$ with $U_{\text{new}} = U_{\text{raw}} X$, $V_{\text{new}} = V_{\text{raw}} Y$ such that $A = U_{\text{new}} B V_{\text{new}}^H$ still holds. Since $A = U_{\text{raw}} W_{\text{raw}} V_{\text{raw}}^H$ exactly, this reduces to requiring $X B Y^H = W_{\text{raw}}$.
$$\text{diagonal:} \quad x_i \bar y_i \, B_{ii} = (W_{\text{raw}})_{ii} \;\Longrightarrow\; x_i \bar y_i = \frac{1}{d_{l,i} d_{r,i}} \qquad\qquad \text{off-diagonal:} \quad x_i \bar y_{i+1}\, B_{i,i+1} = (W_{\text{raw}})_{i,i+1} \;\Longrightarrow\; x_i \bar y_{i+1} = \frac{1}{d_{l,i} d_{r,i+1}}$$

Trying $x_i = \overline{d_{l,i}}$: the diagonal condition gives $\bar y_i = 1/(d_{l,i} d_{r,i} \overline{d_{l,i}}) = 1/(|d_{l,i}|^2 d_{r,i}) = 1/d_{r,i}$ (unit modulus), i.e. $y_i = d_{r,i}$. Substituting into the off-diagonal condition yields: $x_i \bar y_{i+1} = \overline{d_{l,i}}\,\overline{d_{r,i+1}} = 1/(d_{l,i} d_{r,i+1})$ (again using unit modulus) - **matches exactly**, confirming $X = D_L^H$, $Y = D_R$ solves both conditions simultaneously. Hence:
 
$$U_{\text{new}} = U_{\text{raw}} \cdot D_L^H \qquad V_{\text{new}} = V_{\text{raw}} \cdot D_R$$

*Note:* the derivation is only valid if `super[i]` fed into `real_bidiag` is exactly $\bar\alpha$ from earlier.

---