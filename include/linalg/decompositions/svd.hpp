#pragma once

#include <linalg/decompositions/bidiag.hpp>

namespace linalg {
    // Publicly-visible SVD package: `A = U * diag(s) * V^H`.
    template<typename T, Layout LL>
    struct SVDResult {
        Matrix<T, LL> U;    // Left singular vectors as columns.
        Vector<double> s;   // Vector of singular values, sorted in descending order.
        Matrix<T, LL> V;    // Right singular vectors as columns (non-transposed).
    };

    namespace detail {
        constexpr int MAX_ITER_PER_SVAL = 32;

        // Rotates columns `(i, j)` of `M` in place: `col_i <- c*col_i + s*col_j`, `col_j <- -s*col_i + c*col_j`.
        template<typename T, Layout L>
        void rot_col(Matrix<T, L>& M, size_t ci, size_t cj, double c, double s) {
            const size_t m = M.rows();
            if (m == 0) return;
            const T cc = static_cast<T>(c), ss = static_cast<T>(s);
            parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
                LINALG_VECTORIZE
                for (size_t r = rs; r < re; ++r) {
                    const T a = M(r, ci), b = M(r, cj);
                    M(r, ci) = cc * a + ss * b;
                    M(r, cj) = -ss * a + cc * b;
                };
            });
        };

        //  Wilkinson shift for the implicit-QR bidiagonal step.
        LINALG_INLINE double wilkinson_shift_bidiag(double dm1, double em1, double d0) {
            const double a = dm1 * dm1;
            const double b = dm1 * em1;
            const double d = em1 * em1 + d0 * d0;
            const double tr = a + d;
            const double disc = std::sqrt(std::max(0.0, (a - d) * (a - d) / 4.0 + b * b));
            const double l1 = tr / 2.0 + disc;
            const double l2 = tr / 2.0 - disc;
            return (std::abs(l1 - d) <= std::abs(l2 - d)) ? l1 : l2;
        };

        template<typename T, Layout L>
        void chase_zero_diag(std::vector<double>& d, std::vector<double>& e, size_t zi, size_t q, Matrix<T, L>& U, bool accQ) {
            double f = e[zi];
            e[zi] = 0.0;
            for (size_t k = zi + 1; k <= q; ++k) {
                double c, s, r = d[k];
                double ff = f;
                rotg(r, ff, c, s);
                d[k] = r;
                if (accQ) rot_col(U, k, zi, c, s);
                if (k < q) {
                    f = -s * e[k];
                    e[k] = c * e[k];
                };
            };
        };

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

        template<typename T, Layout L>
        bool gkr_iteration(std::vector<double>& d, std::vector<double>& e, Matrix<T, L>& U, Matrix<T, L>& V, bool accumulate) {
            const size_t n = d.size();
            if (n <= 1) return true;
            const double eps = std::numeric_limits<double>::epsilon();
            size_t q = n - 1;
            int since_deflation = 0;
            while (q > 0) {
                // Find largest p such that [p, q] is unreduced (no negligible superdiagonal).
                size_t p = q;
                while (p > 0) {
                    const double tol = eps * (std::abs(d[p - 1]) + std::abs(d[p]));
                    if (std::abs(e[p - 1]) <= tol) { e[p - 1] = 0.0; break; };
                    --p;
                };
                if (p == q) { --q; since_deflation = 0; continue; };
                // Zero-diagonal special case anywhere in the active window.
                bool handled_zero = false;
                for (size_t i = p; i < q; ++i) {
                    const double dtol = eps * (std::abs(d[p]) + std::abs(d[q]) + 1.0);
                    if (std::abs(d[i]) <= dtol) {
                        chase_zero_diag(d, e, i, q, U, accumulate);
                        handled_zero = true;
                        break;
                    };
                };
                if (handled_zero) { ++since_deflation; continue; };

                const double mu = wilkinson_shift_bidiag((p + 1 <= q) ? d[q - 1] : d[p], e[q - 1], d[q]);
                gkr_step(d, e, p, q, mu, U, V, accumulate, accumulate);
                ++since_deflation;
                const double tol = eps * (std::abs(d[q - 1]) + std::abs(d[q]));
                if (std::abs(e[q - 1]) <= tol) { e[q - 1] = 0.0; --q; since_deflation = 0; };
                if (since_deflation > static_cast<int>(q - p + 1) * MAX_ITER_PER_SVAL) return false;
            };
            return true;
        };

        template<typename T, Layout L>
        void post_svd(std::vector<double>& d, Matrix<T, L>& U, Matrix<T, L>& V, bool accumulate) {
            const size_t k = d.size();
            for (size_t i = 0; i < k; ++i) {
                if (d[i] < 0.0) {
                    d[i] = -d[i];
                    if (accumulate) {
                        parallel_for(U.rows(), PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
                            LINALG_VECTORIZE
                            for (size_t r = rs; r < re; ++r) U(r, i) = -U(r, i);
                        });
                    };
                };
            };
            std::vector<size_t> idx(k);
            for (size_t i = 0; i < k; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return d[a] > d[b]; });
            std::vector<double> d_sorted(k);
            for (size_t i = 0; i < k; ++i) d_sorted[i] = d[idx[i]];
            d = std::move(d_sorted);
            if (!accumulate) return;
            // Column permutation: independent per output column `i`.
            Matrix<T, L> U_sorted(U.rows(), k), V_sorted(V.rows(), k);
            const size_t urows = U.rows(), vrows = V.rows();
            parallel_for(k, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (urows + vrows + 1)),
                [&](size_t is, size_t ie) {
                    for (size_t i = is; i < ie; ++i) {
                        const size_t src = idx[i];
                        LINALG_VECTORIZE
                        for (size_t r = 0; r < urows; ++r) U_sorted(r, i) = U(r, src);
                        LINALG_VECTORIZE
                        for (size_t r = 0; r < vrows; ++r) V_sorted(r, i) = V(r, src);
                    };
                });
            U = std::move(U_sorted);
            V = std::move(V_sorted);
        };

        struct DqdsBlock { size_t lo, hi; double shift; };

        LINALG_INLINE bool dqds_sweep(std::vector<double>& qv, std::vector<double>& ev, size_t lo, size_t hi, double tau) {
            double dd = qv[lo] - tau;
            if (dd < 0.0) return false;
            for (size_t i = lo; i < hi; ++i) {
                const double qp = dd + ev[i];
                if (qp == 0.0) return false;
                const double t = qv[i + 1] / qp;
                ev[i] = ev[i] * t;
                qv[i] = qp;
                dd = dd * t - tau;
                if (dd < 0.0) return false;
            };
            qv[hi] = dd;
            return true;
        };

        LINALG_INLINE double dqds_shift(const std::vector<double>& qv, const std::vector<double>& ev, size_t lo, size_t hi) {
            if (hi == lo) return 0.0;
            const double a = qv[hi - 1] + (hi >= lo + 2 ? ev[hi - 2] : 0.0);
            const double d = qv[hi] + ev[hi - 1];
            const double c_sq = qv[hi - 1] * ev[hi - 1];
            const double tr = a + d;
            const double disc = std::sqrt(std::max(0.0, (a - d) * (a - d) / 4.0 + c_sq));
            const double l2 = tr / 2.0 - disc; // Smaller root yields safe (non-negative by construction) shift.
            return std::max(0.0, l2);
        };
    };

    /// @brief Values-only singular value computation via dqds.
    /// @param d Diagonal of the real bidiagonal `B` (post-bidiagonalization).
    /// @param e Superdiagonal of `B`.
    /// @return Singular values, descending.
    /// @throw `std::runtime_error` on convergence failure.
    LINALG_INLINE Vector<double> dqds(const Vector<double>& d, const Vector<double>& e) {
        const size_t n = d.size();
        if (n == 0) return Vector<double>(0);
        if (n == 1) { Vector<double> s(1); s[0] = std::abs(d[0]); return s; };

        std::vector<double> qv(n), ev(n - 1);
        for (size_t i = 0; i < n; ++i) qv[i] = d[i] * d[i];
        for (size_t i = 0; i + 1 < n; ++i) ev[i] = e[i] * e[i];

        const double eps = std::numeric_limits<double>::epsilon();
        std::vector<detail::DqdsBlock> stack{ {0, n - 1, 0.0} };
        std::vector<double> result;
        result.reserve(n);
        int global_iter = 0;
        const int iter_cap = static_cast<int>(n) * detail::MAX_ITER_PER_SVAL * 4;

        while (!stack.empty()) {
            detail::DqdsBlock blk = stack.back(); stack.pop_back();
            const size_t lo = blk.lo, hi = blk.hi;
            if (lo == hi) { result.push_back(qv[lo] + blk.shift); continue; };
            // Split on negligible ev entries within [lo, hi); both children inherit the
            // parent's already-accumulated shift unchanged (splitting doesn't shift anything).
            bool split = false;
            for (size_t i = lo; i < hi; ++i) {
                const double tol = eps * eps * (qv[i] + qv[i + 1] + 1.0); // ev entries are squared, hence eps^2 scale.
                if (ev[i] <= tol) {
                    stack.push_back({ i + 1, hi, blk.shift });
                    stack.push_back({ lo, i, blk.shift });
                    split = true;
                    break;
                };
            };
            if (split) continue;

            const double tau = detail::dqds_shift(qv, ev, lo, hi);
            double applied_shift = tau;
            if (tau != 0.0) {
                // Snapshot before attempting the shifted sweep: a failure leaves qv/ev partially transformed under the (rejected) tau, which must not leak into the tau=0 retry.
                std::vector<double> qv_snap(qv.begin() + lo, qv.begin() + hi + 1);
                std::vector<double> ev_snap(ev.begin() + lo, ev.begin() + hi);
                if (!detail::dqds_sweep(qv, ev, lo, hi, tau)) {
                    std::copy(qv_snap.begin(), qv_snap.end(), qv.begin() + lo);
                    std::copy(ev_snap.begin(), ev_snap.end(), ev.begin() + lo);
                    detail::dqds_sweep(qv, ev, lo, hi, 0.0); // Always succeeds: see dqds_sweep's tau=0 non-negativity argument.
                    applied_shift = 0.0;
                };
            } else {
                detail::dqds_sweep(qv, ev, lo, hi, 0.0);
            };
            stack.push_back({ lo, hi, blk.shift + applied_shift });

            if (++global_iter > iter_cap) throw std::runtime_error("dqds: iteration cap exceeded.");
            // Re-check for a fresh split created by this sweep before iterating again.
            for (size_t i = lo; i < hi; ++i) {
                const double tol = eps * eps * (qv[i] + qv[i + 1] + 1.0);
                if (ev[i] <= tol) {
                    stack.pop_back();
                    stack.push_back({ i + 1, hi, blk.shift + applied_shift });
                    stack.push_back({ lo, i, blk.shift + applied_shift });
                    break;
                };
            };
        };

        std::sort(result.begin(), result.end(), std::greater<double>());
        Vector<double> s(result.size());
        for (size_t i = 0; i < result.size(); ++i) s[i] = std::sqrt(std::max(0.0, result[i]));
        return s;
    };

    /// @brief Singular Value Decomposition: `A = U * diag(S) * V^H`.
    /// @param A Matrix to be decomposed.
    /// @return `SVDResult` structure.
    /// @throw `std::runtime_error` when GKR iteration fails to converge.
    template<typename T, Layout L>
    SVDResult<T, L> svd(const Matrix<T, L>& A) {
        SVDResult<T, L> res;
        BidiagResult<T, L> bd = bidiag(A, /*accumulate_uv=*/true);
        std::vector<double> d(bd.d.size()), e(bd.e.size());
        for (size_t i = 0; i < d.size(); ++i) d[i] = bd.d[i];
        for (size_t i = 0; i < e.size(); ++i) e[i] = bd.e[i];

        if (!detail::gkr_iteration(d, e, bd.U, bd.V, true))
            throw std::runtime_error("svd: Golub-Kahan-Reinsch iteration failed to converge.");
        detail::post_svd(d, bd.U, bd.V, true);

        res.U = std::move(bd.U);
        res.V = std::move(bd.V);
        res.s = Vector<double>(d.size());
        for (size_t i = 0; i < d.size(); ++i) res.s[i] = d[i];
        return res;
    };

    template<typename T, Layout L, typename E>
    SVDResult<T, L> svd(const MatExpr<E>& e) {
        return svd(Matrix<T, L>(e));
    };
};