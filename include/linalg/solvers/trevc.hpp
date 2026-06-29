#pragma once

#include <linalg/blas/level3.hpp>

namespace linalg {
    // Result of trevc: right and/or left eigenvectors of `A = Q*T*Q^H`.
    template<typename T, Layout LL>
    struct TrevcResult {
        Vector<T> eigenvalues;
        Matrix<T, LL> VR; // Right eigenvectors as columns.
        Matrix<T, LL> VL; // Left eigenvectors.
    };

    template<typename T>
    LINALG_INLINE double smin(const T* LINALG_RESTRICT tp, size_t n) noexcept {
        using R = real_type_t<T>;
        constexpr double ulp = std::numeric_limits<R>::epsilon();
        constexpr double safmin = std::numeric_limits<R>::min();
        double mx = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = std::abs(tp[i * n + i]);
            if (v > mx) mx = v;
        };
        return std::max(ulp * mx, safmin);
    };

    template<typename T>
    LINALG_INLINE void normalise_inf(T* LINALG_RESTRICT p, size_t len) noexcept {
        using R = real_type_t<T>;
        double mx = 0.0;
        for (size_t i = 0; i < len; ++i) {
            const double a = std::abs(p[i]);
            if (a > mx) mx = a;
        };
        if (mx == 0.0) return;
        const R inv = static_cast<R>(1.0 / mx);
        LINALG_VECTORIZE
        for (size_t i = 0; i < len; ++i) p[i] *= inv;
    };

    template<typename T>
    void normalise_cols(T* LINALG_RESTRICT mp, size_t n) {
        using R = real_type_t<T>;
        const size_t thresh = std::max(size_t(1), PARALLEL_THRESHOLD_COMPUTE / (n + 1));
        parallel_for(n, thresh, [mp, n](size_t ks, size_t ke) {
            for (size_t k = ks; k < ke; ++k) {
                double nrm = 0.0;
                for (size_t i = 0; i < n; ++i) nrm += std::norm(mp[i * n + k]);
                nrm = std::sqrt(nrm);
                if (nrm > 0.0) {
                    const R inv = static_cast<R>(1.0 / nrm);
                    for (size_t i = 0; i < n; ++i) mp[i * n + k] *= inv;
                };
            };
        });
    };
};