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

    namespace detail {
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

        template<typename T>
        Matrix<T, Layout::RowMajor> trevc_right(const T* LINALG_RESTRICT tp, const T* LINALG_RESTRICT qp, size_t n, double smin) {
            // Phase 1: substitution into ColMajor big-buffer:
            std::vector<T, AlignedAllocator<T>> big(n * n, T(0));
            T* LINALG_RESTRICT bp = detail::assume_aligned<64>(big.data());
            auto& pool = ThreadPool::instance();
            const size_t NT = pool.thread_count();
            std::vector<std::future<void>> futures;
            futures.reserve(NT);
            for (size_t t = 0; t < NT; ++t) {
                futures.push_back(pool.enqueue([=, &big]() noexcept {
                    // Private y-buffer for this thread's columns.
                    std::vector<T, AlignedAllocator<T>> y_store(n, T(0));
                    T* LINALG_RESTRICT yp = detail::assume_aligned<64>(y_store.data());
                    for (size_t k = t; k < n; k += NT) {
                        std::fill(yp, yp + n, T(0));
                        const T lam = tp[k * n + k];
                        yp[k] = T(1);
                        // Back-substitute rows k-1...0.
                        for (size_t ii = 0; ii < k; ++ii) {
                            const size_t i = k - 1 - ii;
                            const T* LINALG_RESTRICT row_i = tp + i * n;
                            T sum = T(0);
                            LINALG_VECTORIZE
                            for (size_t j = i + 1; j <= k; ++j)
                                sum += row_i[j] * yp[j];
                            T pivot = tp[i * n + i] - lam;
                            if (std::abs(pivot) < smin) pivot = T(smin);
                            yp[i] = -sum / pivot;
                        }
                        normalise_inf(yp, k + 1);
                        T* LINALG_RESTRICT col_k = bp + k * n;
                        for (size_t i = 0; i <= k; ++i) col_k[i] = yp[i];
                    };
                }));
            };
            for (auto& f : futures) f.get();

            // Phase 2: transpose buffer into RowMajor VT.
            Matrix<T, Layout::RowMajor> VT(n, n, T(0));
            T* LINALG_RESTRICT vp = detail::assume_aligned<64>(VT.data());
            parallel_for(n, 1, [bp, vp, n](size_t ks, size_t ke) noexcept {
                for (size_t k = ks; k < ke; ++k) {
                    const T* LINALG_RESTRICT col = bp + k * n;
                    for (size_t i = 0; i <= k; ++i) vp[i * n + k] = col[i];
                };
            });

            // Phase 3: blocked lower-triangular GEMM: VR = Q * VT.
            Matrix<T, Layout::RowMajor> VR(n, n, T(0));
            T* LINALG_RESTRICT rp = detail::assume_aligned<64>(VR.data());
            for (size_t j0 = 0; j0 < n; j0 += L2_BLOCK) {
                const size_t j1 = std::min(j0 + L2_BLOCK, n);
                const size_t bs = j1 - j0; // Number of columns in this block..
                const size_t K  = j1; // Only first j1 rows of VT nonzero
                detail::gemm_direct<T, Layout::RowMajor>(T(1), qp, n, vp + j0, n, rp + j0, n, n, bs, K);
            };

            normalise_cols(rp, n);
            return VR;
        };
    };
};