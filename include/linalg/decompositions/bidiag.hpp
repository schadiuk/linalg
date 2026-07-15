#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Bidiagonalisation package: `A = U * B * V^H`, `B` real bidiagonal.
    template<typename T, Layout LL>
    struct BidiagResult {
        Matrix<T, LL> U;    // Left Householder product.
        Vector<double> d;   // Main diagonal of `B`, length `k = (m, n)`.
        Vector<double> e;   // Superdiagonal of `B`, length `k - 1`.
        Matrix<T, LL> V;    // Right Householder product.
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
    };
};