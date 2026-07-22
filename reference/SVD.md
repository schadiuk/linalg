# Singular Value Decomposition: reference
> **Source files:** `bidiag.hpp`, `svd.hpp`.

> **Dependees:** `qr.hpp` (reuses [Householder infrastructure](/reference/QR.md#12-householder-reflectors) and other routines).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundations)
- [Unblocked bidiagonalisation](#2-bidiagonalisation)
- [The problem of real diagonal](#3-realifying-the-diagonal)
- [Blocked bidiagonalisation](#4-blocked-bidiagonalisation)
- [Wide matrices](#5-wide-matrices)
- [GKR algorithm](#6-golub-kahan-reinsch-iteration)

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
## 4. Blocked bidiagonalisation

`labrd` (panel reduction) and `bidiag_tall_blocked` (driver) are the LAPACK `ZLABRD`/`ZGEBRD` pair equivalent blocked path, used whenever $n > n_b = 64$. The structure resembles that present in QR's [`larft`](/reference/QR.md#4-larft-compact-wy-t-matrix-construction) / [`apply_wy_left`](/reference/QR.md#5-apply_wy_left-blocked-trailing-update) split: a Level-2 panel factorisation produces auxiliary matrices, which a single pair of Level-3 GEMMs then applies to the untouched trailing submatrix. However, QR's compact $T$-matrix becomes **two** auxiliary matrices $X$, $Y$ here with each panel column carryiong *two* reflectors (left and right) instead of one.

### 4.1 Deferred-update identity

After a panel of width $n_b$ is factored, the trailing $(M - n_b) \times (N - n_b)$ submatrix satisfies

$$A_{\text{trail}} \;\leftarrow\; A_{\text{trail}} - V\,Y^H - X\,U$$

where $V \in \mathbb{F}^{(M-n_b) \times n_b}$, $X \in \mathbb{F}^{(M-n_b) \times n_b}$ hold *(embedded slices of)* the left reflectors and their derived correction vectors, and $Y \in \mathbb{F}^{(N-n_b) \times n_b}$, $U \in \mathbb{F}^{n_b \times (N-n_b)}$ the corresponding right-side quantities. 

This is the announced earlier two-sided analogue of QR's $W_{\text{trail}} \leftarrow (I - VTV^H)W_{\text{trail}}$: a rank-$n_b$ correction from the left reflectors ($VY^H$) plus a second rank-$n_b$ correction from the right reflectors ($XU$), applied in two GEMMs instead of $n_b$ individual reflector applications.

### 4.2 Reflector renormalisation

`householder_reflector` returns an arbitrarily-scaled valid pair $(v, \beta)$: for any nonzero scalar $c$, $(cv,\, \beta/|c|^2)$ represents the *identical* operator $H = I - \beta v v^H$, since

$$\left(\frac{\beta}{|c|^2}\right)(cv)(cv)^H = \frac{\beta}{|c|^2}\, |c|^2\, v v^H = \beta\, v v^H$$
 
`apply_householder_left`/`apply_householder_right` apply $H$ correctly for *any such pair* - they were never restricted to a particular scaling (this is exactly what lets QR's own reflectors be stored and replayed without normalisation). The panel routine exploits this to locally renormalise every reflector to **LAPACK convention** by setting $v_1[0] = 1$, $\tau = \beta |v[0]|^2$: `ZLABRD` conjugates row $i$ in place before the pair of GEMVs that correct $\text{Wp}(i,\, i{+}1{:}N{-}1)$, performs them, and un-conjugates the *whole* row again only after the right reflector and $X(:,i)$ have been built - i.e. the row spends most of step $i$ in a conjugated state.

### 4.3 $X$, $Y$ construction

For panel-local step $i$ (global column $k{+}i$), with `len_u` $= M-i$, `ncol_trail` $= N-i-1$, `len_v` $=$ `ncol_trail`:
 
| Term | Formula | Shape |
|---|---|---|
| Column correction | $\text{Wp}(i{:}M{-}1, i) \mathrel{-}= \text{Wp}(i{:}M{-}1, 0{:}i) \cdot \overline{Y(i, 0{:}i)}^\top + X(i{:}M{-}1, 0{:}i) \cdot \text{Wp}(0{:}i, i)$ | $O(\text{len\_u} \cdot i)$ |
| $Y$ main term | $Y(i{+}1{:}N{-}1, i) = \overline{\text{Wp}(i{:}M{-}1,\, i{+}1{:}N{-}1)}^\top v_1$ | $O(\text{len\_u} \cdot \text{ncol\_trail})$ |
| $Y$ correction (×2) | Subtract cross-terms through `ybuf`/`ybuf2` (inner products against previously-built $Y$, $X$ columns). | $O(i \cdot \text{len\_u})$ each |
| Row correction | Cf. the discussion earlier. | $O(i \cdot \text{ncol\_trail})$ |
| $X$ main term | $X(i{+}1{:}M{-}1, i) = \text{Wp}(i{+}1{:}M{-}1,\, i{+}1{:}N{-}1) \cdot u_{1v}$ | $O(\text{mrow\_trail} \cdot \text{len\_v})$ |
| $X$ correction (×2) | Subtract cross-terms through `xbuf`/`xbuf2` | $O(i \cdot \text{len\_v})$ each. |

The dominant terms - $Y$'s main term and $X$'s main term, each $O(MN)$ per panel step, $O(MNn_b)$ per panel - mirror QR's $O((m{-}j)(n{-}j))$ per-step trailing-update cost; the correction terms are all $O(i)$-bounded (never exceeding $n_b$) and therefore asymptotically dominated for large $M, N$.

### 4.4 Deferred trailing update (driver)

`bidiag_tall_blocked` extracts $V_{\text{trail}}$, $Y_{\text{trail}}$, $X_{\text{trail}}$, $U_{\text{trail}}$ — offset-embedded slices restricted to the trailing rows/columns, directly analogous to QR's [$V$ matrix construction](/reference/QR.md#52-the-v-matrix-layout):

```cpp
for (r = 0; r < Mtrail; ++r) Vtrail(r, i) = u1s[i][nb + r - i];
for (r = 0; r < Mtrail; ++r) Xtrail(r, i) = X(nb + r, i);
for (c = 0; c < Ntrail; ++c) Ytrail(c, i) = Y(nb + c, i);
for (c = 0; c < Ntrail; ++c) Utrail(i, c) = conj(v1s[i][nb + c - i - 1]);
```

then applies [the known identity](#41-deferred-update-identity) as two GEMMs:

```cpp
gemm(T(-1), expr(Vtrail), hermitian(Ytrail), T(1), Atrail);   // A_trail -= V * Y^H
gemm(T(-1), expr(Xtrail), expr(Utrail), T(1), Atrail);        // A_trail -= X * U
```

$A_{\text{trail}}$ itself is extract-and-written-back through a tight temporary, for the same stride-normalisation reason as QR's [$W_{\text{trail}}$ pattern](/reference/QR.md#53-extract-and-writeback-pattern).

After the panel's own diagonal/superdiagonal are restored, the driver writes them into the global working matrix $W$, explicitly zeroing every other entry of that row/column (matching the unblocked convention so that `bidiag_finalize_tall` can read `d`/`e` back the same way regardless of which path produced them), then copies the GEMM-corrected trailing block back into $W$ for the next panel iteration to consume as its own fresh `Wp`.

---
## 5. Wide matrices

`bidiag()` handles $m < n$ by bidiagonalising $A^H$ (which is $n \times m$, tall) and re-deriving the result for $A$ itself:

```cpp
Matrix<T, L> AH = hermitian(A);
auto sub = detail::bidiag_tall(AH, accumulate_uv); // A^H = U' * B' * V'^H, B' upper bidiagonal.
const size_t k = sub.d.size();
BidiagResult<T, L> res;
res.d = Vector<double>(k);
for (size_t i = 0; i < k; ++i) res.d[i] = sub.d[k - 1 - i];
res.e = Vector<double>(k > 0 ? k - 1 : 0);
for (size_t i = 0; i + 1 < k; ++i) res.e[i] = sub.e[k - 2 - i];
```

Since $A^H = U' B' V'^H$, transposing gives $A = V' (B')^\top U'^H$ (real $B'$, so $(B')^H = (B')^\top$). $(B')^\top$ is **lower**.

Reversing both the row and column order of a lower-bidiagonal matrix (index $i \mapsto k{-}1{-}i$) maps its subdiagonal onto a superdiagonal, turning it back into upper-bidiagonal form; reversing the columns of $U'$, $V'$ to match undoes the same index flip on the singular-vector side. The net effect - reverse `d`, reverse `e`, take $U = $ reversed $V'$ and $V = $ reversed $U'$ - reconstructs $A = U B V^H$ with $B$ upper bidiagonal, without a second Householder pass needed.

---
## 6. Golub-Kahan-Reinsch iteration

### 6.1 Overview

`gkr_iteration` diagonalises the real bidiagonal $B$ in place by applying the implicit-shift QR algorithm to the (never formed) symmetric tridiagonal $T = B^H B$. Each sweep is a sequence of Givens rotations, alternately applied from the right (to $B$'s columns, equivalently to $V$) and the left (to $B$'s rows, equivalently to $U$), that "chase a bulge" from the top-left to the bottom-right of the active window, mirroring the chasing pattern known as [Francis step](/reference/SCHUR.md#5-francis-implicit-single-shift-qr) used for the general eigenvalue problem, specialised to a single real shift because $T$ is symmetric.

### 6.2 Wilkinson shift

```cpp
LINALG_INLINE double wilkinson_shift_bidiag(double dm1, double em1, double d0) {
    const double a = dm1 * dm1;
    const double b = dm1 * em1;
    const double d = em1 * em1 + d0 * d0;
    const double tr = a + d;
    const double disc = std::sqrt(std::max(0.0, (a - d) * (a - d) / 4.0 + b * b));
    const double l1 = tr / 2.0 + disc;
    const double l2 = tr / 2.0 - disc;
    return (std::abs(l1 - d) <= std::abs(l2 - d)) ? l1 : l2; // Eigenvalue closer to d.
};
```

The trailing $2\times2$ principal submatrix of $T = B^H B$, expressed directly in $B$'s own entries $(d_{m-1}, e_{m-1}, d_0)$ (bottom-right corner of the active window), has eigenvalues $l_1, l_2$ from the usual $2\times2$ symmetric-eigenvalue formula. Choosing the root **closer to** $d = T_{qq}$ (rather than always the larger or smaller root) guarantees the shift is a good local approximation to the eigenvalue the iteration is about to converge to, giving asymptotically cubic convergence and avoiding spurious large shifts that would slow convergence on well-separated singular values.

### 6.3 The bulge chase

```cpp
template<typename T, Layout L>
void gkr_step(std::vector<double>& d, std::vector<double>& e, size_t p, size_t q,  double mu, Matrix<T, L>& U, Matrix<T, L>& V, bool accU, bool accV) {
    double f = d[p] * d[p] - mu;
    double g = d[p] * e[p];
    for (size_t k = p; k < q; ++k) {
        // Right rotation on columns (k, k+1): zero g into f.
        double c, s, r = f, gg = g;
        rotg(r, gg, c, s);
        if (k > p) e[k - 1] = r;
        if (accV) rot_col(V, k, k + 1, c, s);
        f = c * d[k] + s * e[k];
        e[k] = c * e[k] - s * d[k];
        g = s * d[k + 1];
        d[k + 1] = c * d[k + 1];
        // Left rotation on rows (k, k+1): zero g into f.
        double c2, s2, r2 = f, gg2 = g;
        rotg(r2, gg2, c2, s2);
        d[k] = r2;
        if (accU) rot_col(U, k, k + 1, c2, s2);
        f = c2 * e[k] + s2 * d[k + 1];
        d[k + 1] = c2 * d[k + 1] - s2 * e[k];
        if (k + 1 < q) {
            g = s2 * e[k + 1];
            e[k + 1] = c2 * e[k + 1];
        };
    };
    e[q - 1] = f;
};
```

Each iteration of the $k$-loop performs **one right rotation** (columns $k, k{+}1$ of $B$, accumulated into $V$) immediately followed by **one left rotation** (rows $k, k{+}1$, accumulated into $U$). The right rotation zeros the "bulge" element $g$ introduced by the previous step (or, at $k=p$, by the shift itself) into the superdiagonal position $f$; this necessarily creates a *new* bulge one position further along ($g \leftarrow s \cdot d_{k+1}$), which the following left rotation similarly absorbs while creating the next bulge for the next iteration. `rotg` is the standard (real or complex-$s$/real-$c$) Givens rotation constructor shared with [`schur.hpp`](/reference/SCHUR.md)'s own bulge chases. After $q - p$ iterations the bulge has been walked entirely out of the active window and `e[q-1] = f` records the final superdiagonal entry.

---