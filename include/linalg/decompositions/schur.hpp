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
        constexpr int EXCEPT_FREQ = 16; // Steps between exceptional shifts.
        constexpr int MAX_ITER_PER_EIG = 32;

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
            if (accumulate_q) Q = Matrix<DefaultScalar, L>::identity(n);
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

            for (size_t k = 0; k < K; ++k) unblocked_step(ilo + k);
            if (accumulate_q) {
                for (size_t k = 0; k < K; ++k) {
                    if (betas[k] == 0.0) continue;
                    const size_t k_abs = ilo + k;
                    const size_t len = ihi + 1 - (k_abs + 1);
                    // Right-apply to Q: updates columns k_abs+1 through ihi.
                    apply_householder_right(Q, vs[k], betas[k], k_abs + 1, len);
                };
            };
        
            return {vs, betas};
        };

        LINALG_INLINE
        DefaultScalar wilkinson_shift(DefaultScalar a, DefaultScalar b, DefaultScalar c, DefaultScalar d) {
            // Eigenvalues of [[a,b],[c,d]]:  (a+d)/2 +- sqrt(((a-d)/2)^2 + b*c).
            const DefaultScalar tr = a + d;
            const DefaultScalar disc  = (a - d) * (a - d) / 4.0 + b * c;
            const DefaultScalar sqrtd = std::sqrt(disc);
            const DefaultScalar l1 = tr / 2.0 + sqrtd;
            const DefaultScalar l2 = tr / 2.0 - sqrtd;
            // Return the one closest to d:
            return (std::abs(l1 - d) <= std::abs(l2 - d)) ? l1 : l2;
        };

        /// @brief Applies one Francis step to the active submatrix window `H[p:q+1, p:q+1]`.
        /// @param H Hessenberg matrix input.
        /// @param Q_iter `n * n` Schur vector accumulator.
        /// @param p 
        /// @param q 
        /// @param mu Shift value.
        template<Layout L>
        void francis_step(Matrix<DefaultScalar, L>& H, Matrix<DefaultScalar, L>& Q_iter, size_t p, size_t q, DefaultScalar mu) {
            const size_t n = H.rows();
            const bool accQ = (Q_iter.rows() == n);
            // First Householder application: reflect the 2-vector [H[p,p]-mu, H[p+1,p]] to [r, 0].
            DefaultScalar x0 = H(p, p) - mu;
            DefaultScalar x1 = H(p + 1, p);
            Vector<DefaultScalar> x2(2); x2[0] = x0; x2[1] = x1;
            auto [v, beta] = householder_reflector(x2);
    
            if (beta != 0.0) {
                const size_t left_ncols = n - p;
                apply_householder_left(H, v, beta, p, 2, left_ncols);
                // Right application:
                const DefaultScalar b = static_cast<DefaultScalar>(beta);
                const size_t nrows_right = q + 1;
                Vector<DefaultScalar> w(nrows_right, DefaultScalar(0));
                for (size_t i = 0; i < nrows_right; ++i)
                    for (size_t jj = 0; jj < 2; ++jj) w[i] += H(i, p + jj) * v[jj];
                for (size_t i = 0; i < nrows_right; ++i)
                    for (size_t jj = 0; jj < 2; ++jj) H(i, p + jj) -= b * w[i] * conj(v[jj]);
                
                if (accQ) apply_householder_right(Q_iter, v, beta, p, 2);
            };
            
            for (size_t k = p; k + 2 <= q; ++k) {
                DefaultScalar x0 = H(k + 1, k);
                DefaultScalar x1 = H(k + 2, k);
                Vector<DefaultScalar> x2(2); x2[0] = x0; x2[1] = x1;
                auto [v, beta] = householder_reflector(x2);
                
                if (beta == 0.0) { H(k + 2, k) = DefaultScalar(0); continue; };
        
                // Left-apply to H[k+1:k+3, k:n] (row start k+1, col start k).
                const DefaultScalar b = static_cast<DefaultScalar>(beta);
                const size_t ncols_left = n - k;
                Vector<DefaultScalar> wl(ncols_left, DefaultScalar(0));
                for (size_t j = 0; j < ncols_left; ++j) wl[j] = conj(v[0]) * H(k+1, k+j) + conj(v[1]) * H(k+2, k+j);
                for (size_t j = 0; j < ncols_left; ++j) {
                    H(k+1, k+j) -= b * v[0] * wl[j];
                    H(k+2, k+j) -= b * v[1] * wl[j];
                };

                H(k + 2, k) = DefaultScalar(0); // Enforce zero.
                // Right-apply to H[0:q+1, k+1:k+3] (rows 0...q):
                    //const DefaultScalar b = static_cast<DefaultScalar>(beta);
                    const size_t nrows_right = q + 1;
                    Vector<DefaultScalar> w(nrows_right, DefaultScalar(0));
                    for (size_t i = 0; i < nrows_right; ++i)
                        for (size_t jj = 0; jj < 2; ++jj) w[i] += H(i, k + 1 + jj) * v[jj];
                    for (size_t i = 0; i < nrows_right; ++i)
                        for (size_t jj = 0; jj < 2; ++jj) H(i, k + 1 + jj) -= b * w[i] * conj(v[jj]);
                if (accQ) apply_householder_right(Q_iter, v, beta, k + 1, 2);
            };
        };

        template<Layout L>
        bool qr_iteration(Matrix<DefaultScalar, L>& H, Matrix<DefaultScalar, L>& Q_iter, size_t ilo, size_t ihi) {
            if (ihi <= ilo) return true;
            const double eps = std::numeric_limits<double>::epsilon();
            // Thread-local RNG for exceptional shifts.
            std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
            std::uniform_real_distribution<double> udist(-1.0, 1.0);
            size_t q = ihi; // Top of active window deflates one step at a time.
            int since_deflation = 0; // Steps since last deflation (for exceptional shifts).
            while (q > ilo) {
                // Find the largest p such that H[p:q+1, p:q+1] is unreduced (no small subdiagonals).
                size_t p = q;
                while (p > ilo) {
                    const double tol = eps * (std::abs(H(p - 1, p - 1)) + std::abs(H(p, p)));
                    if (std::abs(H(p, p - 1)) <= tol) {
                        H(p, p - 1) = DefaultScalar(0);
                        break;
                    };
                    --p;
                };

                if (p == q) { --q; since_deflation = 0; continue; };

                DefaultScalar mu;
                if (since_deflation > 0 && since_deflation % EXCEPT_FREQ == 0) {
                    // Exceptional shift: random perturbation scaled by spectral radius estimate.
                    const double scale = std::abs(H(q, q)) + std::abs(H(q, q - 1));
                    mu = DefaultScalar(scale * udist(rng), scale * udist(rng));
                } else {
                    mu = wilkinson_shift(H(q - 1, q - 1), H(q - 1, q), H(q, q - 1), H(q, q));
                };
                // Apply one Francis step to H[p:q+1, p:q+1] window.
                francis_step(H, Q_iter, p, q, mu);
                ++since_deflation;
        
                // Check for new deflation at bottom of active window
                const double tol = eps * (std::abs(H(q - 1, q - 1)) + std::abs(H(q, q)));
                if (std::abs(H(q, q - 1)) <= tol) {
                    H(q, q - 1) = DefaultScalar(0);
                    --q;
                    since_deflation = 0;
                };
                
                // Iteration limit guard (per eigenvalue, not per step)
                if (since_deflation > static_cast<int>(q - ilo + 1) * MAX_ITER_PER_EIG)
                    return false;  // failed to converge
            };
            return true;
        };
    };

    /// @brief General Schur decomposition.
    /// @param A Input matrix (complex-valued).
    /// @param compute_vectors
    /// @param balance Balancing flag.
    /// @return `SchurResult` structure.
    template<Layout L = Layout::RowMajor>
    SchurResult<L> schur(const Matrix<DefaultScalar, L>& A, bool compute_vectors = true, bool balance = true) {
        const size_t n = A.rows();
        BOUNDS_CHECK(A.cols() == n);
        SchurResult<L> res;
        Matrix<DefaultScalar, L> H = A; // Working copy.
        Matrix<DefaultScalar, L> Q(0, 0); // Hessenberg Q (accumulation on request).

        const bool do_balance = balance && n > 1 && !compute_vectors;
        size_t ilo = 0, ihi = (n > 0) ? n - 1 : 0;
        detail::BalanceInfo bi;
        if (do_balance) {
            bi = detail::balance(H);
            ilo = bi.ilo;
            ihi = bi.ihi;
            res.balanced = true;
            res.balance_scale = bi.scale;
            res.balance_perm = bi.perm;
        };

        Matrix<DefaultScalar, L> Q_hess(0, 0);
        auto [vs, betas] = detail::hessenberg_reduce<DefaultScalar, L>(H, ilo, ihi, compute_vectors, Q_hess);
    
        Matrix<DefaultScalar, L> Q_iter(0, 0);
        if (compute_vectors) Q_iter = Matrix<DefaultScalar, L>::identity(n);
    
        if (n > 1) detail::qr_iteration(H, Q_iter, ilo, ihi);

        if (compute_vectors) {
            // Q = Q_hess * Q_iter.
            Matrix<DefaultScalar, L> Q_full(n, n, DefaultScalar(0));
            gemm(DefaultScalar(1), expr(Q_hess), expr(Q_iter), DefaultScalar(0), Q_full);
            res.Q = std::move(Q_full);
        };

        res.T = std::move(H);
        res.eigvals = Vector<DefaultScalar>(n);
        for (size_t i = 0; i < n; ++i) res.eigvals[i] = res.T(i, i);
        return res;
    };

    template<Layout L = Layout::RowMajor>
    SchurResult<L> schur(const Matrix<double, L>& A, bool compute_vectors = true, bool balance = true) {
        const size_t n = A.rows();
        Matrix<std::complex<double>, L> Ac(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) Ac(i, j) = std::complex<double>(A(i, j), 0.0);
        return schur(Ac, compute_vectors, balance);
    };

    template<Layout L = Layout::RowMajor, typename E>
    SchurResult<L> schur(const MatExpr<E>& A_expr, bool compute_vectors = true, bool balance = true) {
        using raw_t = std::remove_cvref_t<decltype(A_expr.self()(0, 0))>;
        // If the expression already yields complex<double>, materialise directly:
        if constexpr (std::is_same_v<raw_t, std::complex<double>>) {
            return schur(Matrix<std::complex<double>, L>(A_expr), compute_vectors, balance);
        } else {
            const size_t m = A_expr.self().rows(), nc = A_expr.self().cols();
            Matrix<std::complex<double>, L> Ac(m, nc);
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < nc; ++j)
                    Ac(i, j) = std::complex<double>(
                        static_cast<double>(std::real(A_expr.self()(i, j))),
                        static_cast<double>(std::imag(A_expr.self()(i, j))));
            return schur(Ac, compute_vectors, balance);
        };
    };

    template<Layout L = Layout::RowMajor, typename E>
    Vector<std::complex<double>> eigenvalues(const MatExpr<E>& A) {
        return schur<L>(A, /*compute_vectors=*/false, /*balance=*/true).eigvals;
    };
    
    template<Layout L = Layout::RowMajor>
    Vector<std::complex<double>> eigenvalues(const Matrix<double, L>& A) {
        return schur<L>(A, false, true).eigvals;
    };
    
    template<Layout L = Layout::RowMajor>
    Vector<std::complex<double>> eigenvalues(const Matrix<std::complex<double>, L>& A) {
        return schur<L>(A, false, true).eigvals;
    };
};