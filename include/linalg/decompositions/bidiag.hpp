#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Bidiagonalisation package: `A = U * B * V^H`, `B` real bidiagonal.
    template<typename T, Layout LL>
    struct BidiagResult {
        Matrix<T, LL> U;    // Left Householder product.
        Vector<double> d;   // Main diagonal of `B`, length `k = min(m, n)`.
        Vector<double> e;   // Superdiagonal of `B`, length `k - 1`.
        Matrix<T, LL> V;    // Right Householder product (non-transposed).
    };

    namespace detail {
        // Multiplies column `j` of `M` by `phase[j]` for complex `T`.
        template<typename T, Layout L>
        void apply_col_phase(Matrix<T, L>& M, const std::vector<T>& phase) {
            if constexpr(!is_complex_v<T>) return;
            const size_t m = M.rows(), k = phase.size();
            if (k == 0 || m == 0) return;
            parallel_for(m, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (k + 1)),
                [&](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < k; ++j) M(i, j) *= phase[j];
                });
        };

        template<typename T>
        LINALG_INLINE void real_bidiag(std::vector<T>& diag, std::vector<T>& super, std::vector<T>& dl, std::vector<T>& dr) {
            const size_t k = diag.size();
            dl.assign(k, T(1));
            dr.assign(k, T(1));
            if constexpr(!is_complex_v<T>) return;

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
                if (i + 1 < k) {
                    const T eval = dl[i] * super[i];
                    dr[i + 1] = conj_phase(eval);
                    super[i] = T(std::abs(eval));
                };
            };
        };

        template<typename T, Layout L>
        BidiagResult<T, L> bidiag_tall(const Matrix<T, L>& A, bool accumulate_uv) {
            const size_t m = A.rows(), n = A.cols();
            const size_t k = n;
            Matrix<T, L> W = A;
            std::vector<Vector<T>> us(k);
            std::vector<double> ubeta(k, 0.0);
            std::vector<Vector<T>> vs(k > 0 ? k - 1 : 0);
            std::vector<double> vbeta(k > 0 ? k - 1 : 0, 0.0);

            for(size_t col = 0; col < k; ++col) {
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
            };

            BidiagResult<T, L> res;
            std::vector<T> draw(k), eraw(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i < k; ++i) draw[i] = W(i, i);
            for (size_t i = 0; i + 1 < k; ++i) eraw[i] = W(i, i + 1);

            std::vector<T> dl, dr;
            real_bidiag(draw, eraw, dl, dr);
            res.d = Vector<double>(k);
            for (size_t i = 0; i < k; ++i) res.d[i] = std::real(static_cast<T>(draw[i]));
            res.e = Vector<double>(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i + 1 < k; ++i) res.e[i] = std::real(static_cast<T>(eraw[i]));

            if (!accumulate_uv) return res;
            // U accumulation: apply left reflectors in reverse order onto I_m (first k columns).
            res.U = Matrix<T, L>(m, k, T(0));
            for (size_t i = 0; i < k; ++i) res.U(i, i) = T(1);
            for (size_t ci = k; ci-- > 0; ) {
                if (ubeta[ci] == 0.0) continue;
                const size_t len = m - ci;
                const size_t ncols = k - ci;
                apply_householder_left(res.U, us[ci], ubeta[ci], ci, len, ncols);
            };
            // V accumulation:
            res.V = Matrix<T, L>(n, k, T(0));
            for (size_t i = 0; i < k; ++i) res.V(i, i) = T(1);
            for (size_t ci = 0; ci < vs.size(); ++ci) {
                if (vbeta[ci] == 0.0) continue;
                const size_t rlen = n - (ci + 1);
                apply_householder_right(res.V, vs[ci], vbeta[ci], ci + 1, rlen);
            };

            // Fold the realizing phases in: `U <- U * D_L^H` (column `i` scaled by `conj(dl[i])`), `V <- V * D_R` (column `i` scaled by `dr[i]`).
            if constexpr (is_complex_v<T>) {
                std::vector<T> dl_conj(k);
                for (size_t i = 0; i < k; ++i) dl_conj[i] = conj(dl[i]);
                apply_col_phase(res.U, dl_conj);
                apply_col_phase(res.V, dr);
            };
            return res;
        };
    };

    /// @brief Unblocked Golub-Kahan bidiagonalisation: `A = U * B * V^H`, `B` real bidiagonal.
    /// @param A Matrix to be decomposed.
    /// @param accumulate_uv Householder products accumulation flag.
    /// @return Corresponding `BidiagResult` structure.
    /// @note Wide matrices (`n > m`) are handled by bidiagonalising `A^H` (tall) and swapping the resulting `U`/`V`.
    template<typename T, Layout L>
    BidiagResult<T, L> bidiag(const Matrix<T, L>& A, bool accumulate_uv = true) {
        const size_t m = A.rows(), n = A.cols();
        if (m < n) {
            Matrix<T, L> AH = hermitian(A);
            auto sub = detail::bidiag_tall(AH, accumulate_uv);
            const size_t k = sub.d.size();
            BidiagResult<T, L> res;
            res.d = Vector<double>(k);
            for (size_t i = 0; i < k; ++i) res.d[i] = sub.d[k - 1 - i];
            res.e = Vector<double>(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i + 1 < k; ++i) res.e[i] = sub.e[k - 2 - i];
            if (accumulate_uv) {
                res.U = Matrix<T, L>(sub.V.rows(), k);
                for (size_t i = 0; i < sub.V.rows(); ++i)
                    for (size_t j = 0; j < k; ++j) res.U(i, j) = sub.V(i, k - 1 - j);
                res.V = Matrix<T, L>(sub.U.rows(), k);
                for (size_t i = 0; i < sub.U.rows(); ++i)
                    for (size_t j = 0; j < k; ++j) res.V(i, j) = sub.U(i, k - 1 - j);
            };
            return res;
        };
        return detail::bidiag_tall(A, accumulate_uv);
    };

    template<typename T, Layout L, typename E>
    BidiagResult<T, L> bidiag(const MatExpr<E>& e, bool accumulate_uv = true) {
        return bidiag(Matrix<T, L>(e), accumulate_uv);
    };
};