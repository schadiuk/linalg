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
            parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
                for (size_t r = rs; r < re; ++r) {
                    const T a = M(r, ci), b = M(r, cj);
                    M(r, ci) = static_cast<T>(c) * a + static_cast<T>(s) * b;
                    M(r, cj) = static_cast<T>(-s) * a + static_cast<T>(c) * b;
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
    };
};