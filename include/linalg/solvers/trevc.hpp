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
                const size_t K = j1; // Only first j1 rows of VT nonzero
                detail::gemm_direct<T, Layout::RowMajor>(T(1), qp, n, vp + j0, n, rp + j0, n, n, bs, K);
            };

            normalise_cols(rp, n);
            return VR;
        };

        template<typename T>
        Matrix<T, Layout::RowMajor> trevc_left(const T* LINALG_RESTRICT tp, const T* LINALG_RESTRICT qp, size_t n, double smin) {
            std::vector<T, AlignedAllocator<T>> big(n * n, T(0));
            T* LINALG_RESTRICT bp = detail::assume_aligned<64>(big.data());

            auto& pool = ThreadPool::instance();
            const size_t NT = pool.thread_count();
            
            std::vector<std::future<void>> futures;
            futures.reserve(NT);
            for (size_t t = 0; t < NT; ++t) {
                futures.push_back(pool.enqueue([=, &big]() noexcept {
                    std::vector<T, AlignedAllocator<T>> z_store(n, T(0));
                    T* LINALG_RESTRICT zp = detail::assume_aligned<64>(z_store.data());
                    for (size_t k = t; k < n; k += NT) {
                        std::fill(zp, zp + n, T(0));
                        const T lam = tp[k * n + k];
                        zp[k] = T(1);
                        // Forward-substitute rows k+1 ... n-1:
                        for (size_t i = k + 1; i < n; ++i) {
                            T sum = T(0);
                            for (size_t j = k; j < i; ++j) sum += linalg::conj(tp[j * n + i]) * zp[j];
                            T pivot = linalg::conj(tp[i * n + i] - lam);
                            if (std::abs(pivot) < smin) pivot = T(smin);
                            zp[i] = -sum / pivot;
                        };
                        normalise_inf(zp + k, n - k);
                        T* LINALG_RESTRICT col_k = bp + k * n;
                        for (size_t i = k; i < n; ++i) col_k[i] = zp[i];
                    };
                }));
            };
            for (auto& f : futures) f.get();

            Matrix<T, Layout::RowMajor> ZT(n, n, T(0));
            T* LINALG_RESTRICT zp_out = detail::assume_aligned<64>(ZT.data());
            parallel_for(n, 1, [bp, zp_out, n](size_t ks, size_t ke) noexcept {
                for (size_t k = ks; k < ke; ++k) {
                    const T* LINALG_RESTRICT col = bp + k * n;
                    for (size_t i = k; i < n; ++i) zp_out[i * n + k] = col[i];
                };
            });

            Matrix<T, Layout::RowMajor> VL(n, n, T(0));
            T* LINALG_RESTRICT lp = detail::assume_aligned<64>(VL.data());
            for (size_t j0 = 0; j0 < n; j0 += L2_BLOCK) {
                const size_t j1 = std::min(j0 + L2_BLOCK, n);
                const size_t bs = j1 - j0;
                const size_t K = n - j0; // Rows j0...n-1 of ZT are nonzero.
                detail::gemm_direct<T, Layout::RowMajor>(T(1), qp + j0, n, zp_out + j0 * n + j0, n, lp + j0, n, n, bs, K);
            };

            normalise_cols(lp, n);
            return VL;
        };
    };
    
    /// @brief Computes eigenvectors of `A` given its complex Schur decomposition `A = Q*T*Q^H`.
    /// @param T_schur Complex upper-triangular Schur factor.
    /// @param Q_schur Complex unitary matrix of Schur vectors.
    /// @param right Compute right eigenvectors.
    /// @param left Compute left eigenvectors
    /// @return Corresponding `TrevcResult` structure.
    template<typename T, Layout LL>
    TrevcResult<T, LL>
    trevc(const Matrix<T, LL>& T_schur, const Matrix<T, LL>& Q_schur, bool right = true, bool left  = false) {
        static_assert(detail::is_complex_v<T>, "trevc: T must be std::complex<float> or std::complex<double>. Use the unified Schur path");
        const size_t N = T_schur.rows();
        BOUNDS_CHECK(T_schur.cols() == N);
        if (right || left) BOUNDS_CHECK(Q_schur.rows() == N && Q_schur.cols() == N);

        // Work matrices:
        Matrix<T, Layout::RowMajor> T_rm(N, N);
        Matrix<T, Layout::RowMajor> Q_rm(N, N);
        if constexpr (LL == Layout::RowMajor) {
            T_rm = T_schur;
            if (right || left) Q_rm = Q_schur;
        } else {
            // Parallel element-wise copy: cheaper than the element-by-element loop.
            parallel_for(N, 1, [&](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i)
                    for (size_t j = 0; j < N; ++j) {
                        T_rm(i, j) = T_schur(i, j);
                        if (right || left) Q_rm(i, j) = Q_schur(i, j);
                    };
            });
        };

        Vector<T> eigs(N);
        const T* LINALG_RESTRICT tp = detail::assume_aligned<64>(T_rm.data());
        for (size_t i = 0; i < N; ++i) eigs[i] = tp[i * N + i];
        if (N == 0) return { std::move(eigs), {}, {} };
    
        const double smin = detail::smin(tp, N);
        const T* LINALG_RESTRICT qp = detail::assume_aligned<64>(Q_rm.data());
        TrevcResult<T, LL> res;
        res.eigenvalues = std::move(eigs);

        if (right) {
            Matrix<T, Layout::RowMajor> VR_rm = detail::trevc_right(tp, qp, N, smin);
    
            if constexpr (LL == Layout::RowMajor) {
                res.VR = std::move(VR_rm);
            } else {
                res.VR = Matrix<T, LL>(N, N);
                parallel_for(N, 1, [&](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < N; ++j) res.VR(i, j) = VR_rm(i, j);
                });
            };
        };

        if (left) {
            Matrix<T, Layout::RowMajor> VL_rm = detail::trevc_left(tp, qp, N, smin);
            if constexpr (LL == Layout::RowMajor) {
                res.VL = std::move(VL_rm);
            } else {
                res.VL = Matrix<T, LL>(N, N);
                parallel_for(N, 1, [&](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < N; ++j) res.VL(i, j) = VL_rm(i, j);
                });
            };
        };
        return res;
    };
};