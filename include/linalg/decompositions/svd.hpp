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
    };
};