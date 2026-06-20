#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Publicly-visible Schur decomposition package.
    template<Layout LL = Layout::RowMajor>
    struct SchurResult {
        Matrix<DefaultScalar, LL> T; // Upper-triangular Schur factor.
        Matrix<DefaultScalar, LL> Q; // Unitary matrix of Schur vectors.
        Vector<DefaultScalar> eigvals;
        Vector<double> balance_scale; // Per-index scaling factors.
        Vector<size_t> balance_perm; // Permutation for balancing.
        bool balanced{false};
    };

    namespace detail {
        struct BalanceInfo {
            Vector<double> scale; // log-base scaling exponent per index.
            Vector<size_t> perm;
            size_t ilo, ihi; // Only [ilo, ihi] is non-trivially redused.
        };

        /// @brief Parlett-Reinsch balancing.
        /// @param A Input matrix.
        /// @return `scale[i] = s_i` such that the balancing transformation is`D = diag(base^{s_0}, ..., base^{s_{n-1}})` and `perm[i]` is source row that was swapped into position `i`.
        /// @note  Applies a similarity `D^{-1}*A*D` where `D = diag(scale)` and a permutation that isolates eigenvalues in the top-left and bottom-right corners.
        template<Layout L>
        BalanceInfo balance(Matrix<DefaultScalar, L>& A) {
            const size_t n = A.rows();
            Vector<double> scale(n, 0.0);
            Vector<size_t> perm(n);
            for (size_t i = 0; i < n; ++i) perm[i] = i;
            // Phase 1: permute rows/columns to isolate decoupled eigenvalues at corners.
            size_t ilo = 0, ihi = n - 1;
            bool changed = true;
            while (changed && ihi > ilo) {
                changed = false;
                for (size_t i = ilo; i <= ihi; ++i) {
                    // Check if row i has all zeros in [ilo, ihi] columns except column i.
                    bool row_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(i, j)) != 0.0) { row_ok = false; break; };
                    };
                    bool col_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(j, i)) != 0.0) { col_ok = false; break; };
                    };

                    if (row_ok && col_ok && i != ihi) {
                        // Swap row/col i with ihi:
                        for (size_t k = 0; k < n; ++k) std::swap(A(i, k), A(ihi, k));
                        for (size_t k = 0; k < n; ++k) std::swap(A(k, i), A(k, ihi));
                        std::swap(perm[i], perm[ihi]);
                        scale[ihi] = 0.0; // No scaling needed for isolated eigenvalue.
                        --ihi;
                        changed = true;
                        break;
                    };
                };
            };

            changed = true;
            while (changed && ilo < ihi) {
                changed = false;
                for (size_t i = ilo; i <= ihi; ++i) {
                    bool row_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(i, j)) != 0.0) { row_ok = false; break; }
                    };
                    bool col_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(j, i)) != 0.0) { col_ok = false; break; }
                    };
                    if (row_ok && col_ok && i != ilo) {
                        for (size_t k = 0; k < n; ++k) std::swap(A(i, k), A(ilo, k));
                        for (size_t k = 0; k < n; ++k) std::swap(A(k, i), A(k, ilo));
                        std::swap(perm[i], perm[ilo]);
                        scale[ilo] = 0.0;
                        ++ilo;
                        changed = true;
                        break;
                    };
                };
            };

            // Phase 2: diagonal scaling on the [ilo, ihi] submatrix.
            static constexpr double BASE = 2.0;
            static constexpr double BETA = 0.95; // Convergence threshold.
            static constexpr int MAXP = 10; // Maximum for repeated passes computing row-norm and col-norm.
            if (ilo < ihi) {
                for (int pass = 0; pass < MAXP; ++pass) {
                    bool any_scaled = false;
                    for (size_t i = ilo; i <= ihi; ++i) {
                        double rn = 0.0, cn = 0.0;
                        for (size_t j = ilo; j <= ihi; ++j) {
                            if (j != i) {
                                rn += std::abs(A(i, j));
                                cn += std::abs(A(j, i));
                            };
                        };
                        if (rn == 0.0 || cn == 0.0) continue;
                        // Compute integer power of BASE so that rn*s ~ cn/s, i.e. s = sqrt(cn/rn), rounded to nearest power of BASE.
                        double s = 1.0;
                        double f = 1.0;
                        double ratio = cn / rn;
                        // Bring ratio into [1/BASE^2, BASE^2] using integer steps.
                        while (ratio < 1.0 / (BASE * BASE)) { ratio *= BASE * BASE; f /= BASE; }
                        while (ratio > BASE * BASE) { ratio /= BASE * BASE; f *= BASE; }
                        // Adjustment: single BASE step.
                        if (ratio < 1.0 / BASE) f /= BASE;
                        else if (ratio > BASE) f *= BASE;
                        if (std::abs(f - 1.0) < 1e-14) continue;
                        if (f == 0.0) continue;
                        any_scaled = true;
                        // Apply similarity:
                        for (size_t j = 0; j < n; ++j) A(i, j) *= f;
                        for (size_t j = 0; j < n; ++j) A(j, i) /= f;
                        scale[i] += std::log2(f);
                    };
                    if (!any_scaled) break;
                };
            };

            return BalanceInfo{ std::move(scale), std::move(perm), ilo, ihi };
        };

        /// @brief Balancing revert.
        /// @param Q Schur vector matrix.
        /// @param bi `BalanceInfo` instance.
        /// @note `Q_balanced = D * Q * D^{-1}` (column scaling).
        template<Layout L>
        void unbalance(Matrix<DefaultScalar, L>& Q, const BalanceInfo& bi) {
            const size_t n = Q.rows();
            const size_t np = bi.perm.size();
            Vector<size_t> inv_perm(np);
            for (size_t i = 0; i < np; ++i) inv_perm[bi.perm[i]] = i;
            for (size_t j = 0; j < n; ++j) {
                const double f = std::exp2(bi.scale[j]);
                for (size_t i = 0; i < n; ++i) Q(i, j) *= f;
            };

            std::vector<size_t> p(n);
            for (size_t i = 0; i < n; ++i) p[i] = i;
            for (size_t i = 0; i < n; ++i) {
                if (bi.perm[i] != i) {
                    for (size_t j = i+1; j < n; ++j) {
                        if (p[j] == bi.perm[i]) { std::swap(p[i], p[j]); break; };
                    };
                };
            };

            std::vector<size_t> inv_p(n);
            for (size_t i = 0; i < n; ++i) inv_p[p[i]] = i;
            Matrix<DefaultScalar, L> Q_tmp(n, n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) Q_tmp(i, j) = Q(inv_p[i], j);
            
            Q = Q_tmp;
        };

        // Compact-WY trailing update.
        template<typename T, Layout L>
        void apply_wy_right(Matrix<T, L>& W, const std::vector<Vector<T>>& vs, const Matrix<T, L>& T_mat, size_t k, size_t nb, size_t i0) {
            const size_t m = W.rows();
            const size_t n = W.cols();
            const size_t len = m - k;
            const size_t nrows_trail = (i0 < n) ? n - i0 : 0;
            if (nrows_trail == 0 || len == 0 || nb == 0) return;

            Matrix<T, L> V(len, nb, T(0));
            for (size_t j = 0; j < nb; ++j) {
                const Vector<T>& vj = vs[k + j];
                for (size_t i = 0; i < vj.size(); ++i) V(i + j, j) = vj[i];
            };

            const size_t copy_thresh = std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (len + 1));
            Matrix<T, L> W_trail(nrows_trail, len);
            parallel_for(nrows_trail, copy_thresh, [&](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i)
                    for (size_t jj = 0; jj < len; ++jj) W_trail(i, jj) = W(i0 + i, k + jj);
            });
            // GEMM 1: C = W_trail * V (nrows_trail * nb).
            Matrix<T, L> C(nrows_trail, nb, T(0));
            gemm(T(1), expr(W_trail), expr(V), T(0), C);
            // GEMM 2: CS = C * T_mat^H (nrows_trail * nb).
            Matrix<T, L> CS(nrows_trail, nb, T(0));
            gemm(T(1), expr(C), hermitian(T_mat), T(0), CS);
            // GEMM 3: W_trail -= CS * V^H.
            gemm(-T(1), expr(CS), hermitian(V), T(1), W_trail);
            // Write W_trail back into W[i0:nrows, k:m].
            parallel_for(nrows_trail, copy_thresh, [&](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i)
                    for (size_t jj = 0; jj < len; ++jj) W(i0 + i, k + jj) = W_trail(i, jj);
            });
        };

        /// @brief In-place reduction to upper Hessenberg form.
        template<typename T, Layout L>
        std::pair<std::vector<Vector<DefaultScalar>>, std::vector<double>>
        hessenberg_reduce(Matrix<DefaultScalar>& A, size_t ilo, size_t ihi, bool accumulate_q, Matrix<DefaultScalar, L>& Q) {
            const size_t n = A.rows();
            const size_t nsub = (ihi >= ilo) ? ihi - ilo + 1 : 0;
            const size_t K = (nsub > 1) ? nsub - 1 : 0;

            std::vector<Vector<DefaultScalar>> vs(K);
            std::vector<double> betas(K, 0.0);
            Q = Matrix<DefaultScalar, L>::identity(n);
            if (K == 0) return {vs, betas};

            auto hess_left_apply = [&](const Vector<DefaultScalar>& v, double beta, size_t r0, size_t c0, size_t len, size_t ncols) {
                if (beta == 0.0 || len == 0 || ncols == 0) return;
                const DefaultScalar b = static_cast<DefaultScalar>(beta);
                Vector<DefaultScalar> w(ncols, DefaultScalar(0));
                for (size_t j = 0; j < ncols; ++j)
                    for (size_t i = 0; i < len; ++i) w[j] += conj(v[i]) * A(r0 + i, c0 + j);
                // A[r0+i, c0+j] -= beta * v[i] * w[j]:
                for (size_t i = 0; i < len; ++i)
                    for (size_t j = 0; j < ncols; ++j) A(r0 + i, c0 + j) -= b * v[i] * w[j];
            };
        
            auto unblocked_step = [&](size_t k_abs) {
                const size_t len = ihi + 1 - (k_abs + 1);
                if (len < 1) return;
                // Extract column segment A[k_abs+1 : ihi+1, k_abs]:
                Vector<DefaultScalar> x(len);
                for (size_t i = 0; i < len; ++i) x[i] = A(k_abs + 1 + i, k_abs);
                auto [v, beta] = householder_reflector(x);
                const size_t kidx = k_abs - ilo;
                vs[kidx] = v;
                betas[kidx] = beta;
        
                if (beta == 0.0) return;
                hess_left_apply(v, beta, k_abs + 1, k_abs, len, n - k_abs);
                // Enforce exact zeros below subdiagonal in this column:
                for (size_t i = k_abs + 2; i <= ihi; ++i) A(i, k_abs) = DefaultScalar(0);
                apply_householder_right(A, v, beta, k_abs + 1, len);
            };

            for (size_t k = 0; k < K; ++k) unblocked_step(ilo + K);
            if (accumulate_q) {
                for (size_t k = 0; k < K; ++k) {
                    if (betas[k] == 0.0) continue;
                    const size_t k_abs = ilo + k;
                    const size_t len   = ihi + 1 - (k_abs + 1);
                    // Right-apply to Q: updates columns k_abs+1 through ihi.
                    apply_householder_right(Q, vs[k], betas[k], k_abs + 1, len);
                };
            };
        
            return {vs, betas};
        };
    };
};