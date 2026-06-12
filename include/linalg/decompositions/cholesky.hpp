#pragma once

#include <linalg/blas/level3.hpp>

namespace linalg {
    // The structure bundles the Cholesky factor with its triangular flag:
    // `uplo` == 'L': `A = L * L^H`, `L` is lower-triangular matrix.
    // `uplo` == 'U': `A = U^H * U`, `U` upper-triangular.
    template<typename T, Layout LL>
    struct CholeskyResult {
        Matrix<T, LL> factor;
        char uplo;
    };

    namespace detail {
        template<typename T, Layout L>
        bool chol_unblocked_lower(Matrix<T, L>& W, size_t k, size_t nb) {
            T* const wp = W.data();
            const size_t lda = W.stride();

            std::vector<T, AlignedAllocator<T>> cv(nb);
            T* LINALG_RESTRICT cvp = detail::assume_aligned<64>(cv.data());
            for (size_t j = 0; j < nb; ++j) {
                const size_t gj = k + j;
                for (size_t p = 0; p < j; ++p) {
                    if constexpr (L == Layout::RowMajor) cvp[p] = conj(wp[gj * lda + (k + p)]);
                    else cvp[p] = conj(wp[(k + p) * lda + gj]);
                };

                real_type_t<T> d = static_cast<real_type_t<T>>(std::real(static_cast<T>(wp[gj * lda + gj])));

                for (size_t p = 0; p < j; ++p)
                    d -= static_cast<real_type_t<T>>(std::norm(L == Layout::RowMajor ? 
                    wp[gj * lda + (k+p)] : wp[(k+p) * lda + gj]));
                if (d <= real_type_t<T>(0)) return false;

                const real_type_t<T> diag_val = std::sqrt(d);
                wp[gj * lda + gj] = static_cast<T>(diag_val);
                const T inv_diag = T(1) / static_cast<T>(diag_val);

                for (size_t i = gj + 1; i < k + nb; ++i) {
                    T s;
                    if constexpr (L == Layout::RowMajor) s = wp[i * lda + gj];
                    else s = wp[gj * lda + i];

                    LINALG_VECTORIZE
                    for (size_t p = 0; p < j; ++p) {
                        T wpi;
                        if constexpr (L == Layout::RowMajor) wpi = wp[i * lda + (k + p)];
                        else wpi = wp[(k + p) * lda + i];
                        s -= wpi * cvp[p];
                    };

                    if constexpr (L == Layout::RowMajor) wp[i * lda + gj] = s * inv_diag;
                    else wp[gj * lda + i] = s * inv_diag;
                };
            };
            return true;
        };

        template<typename T, Layout L>
        bool chol_unblocked_upper(Matrix<T, L>& W, size_t k, size_t nb) {
            T* const wp = W.data();
            const size_t lda = W.stride();

            std::vector<T, AlignedAllocator<T>> cv(nb);
            T* LINALG_RESTRICT cvp = detail::assume_aligned<64>(cv.data());
            for (size_t j = 0; j < nb; ++j) {
                const size_t gj = k + j;
                for (size_t p = 0; p < j; ++p) {
                    if constexpr (L == Layout::RowMajor) cvp[p] = conj(wp[(k + p) * lda + gj]);
                    else cvp[p] = conj(wp[gj * lda + (k + p)]);
                };

                real_type_t<T> d = static_cast<real_type_t<T>>(std::real(static_cast<T>(wp[gj * lda + gj])));

                for (size_t p = 0; p < j; ++p)
                    d -= static_cast<real_type_t<T>>(std::norm(L == Layout::RowMajor ? 
                    wp[(k+p) * lda + gj] : wp[gj * lda + (k+p)]));
                if (d <= real_type_t<T>(0)) return false;

                const real_type_t<T> diag_val = std::sqrt(d);
                wp[gj * lda + gj] = static_cast<T>(diag_val);
                const T inv_diag = T(1) / static_cast<T>(diag_val);

                for (size_t i = gj + 1; i < k + nb; ++i) {
                    T s;
                    if constexpr (L == Layout::RowMajor) s = wp[gj * lda + i];
                    else s = wp[i * lda + gj];

                    LINALG_VECTORIZE
                    for (size_t p = 0; p < j; ++p) {
                        T wpi;
                        if constexpr (L == Layout::RowMajor) wpi = wp[(k + p) * lda + i];
                        else wpi = wp[i * lda + (k + p)];
                        s -= wpi * cvp[p];
                    };

                    if constexpr (L == Layout::RowMajor) wp[i * lda + i] = s * inv_diag;
                    else wp[gj * lda + gj] = s * inv_diag;
                };
            };
            return true;
        };

        template<typename T, Layout L>
        void zero_off_triangle(Matrix<T, L>& A, char uplo) {
            const size_t n = A.rows();
            const bool lower = (uplo == "L" || uplo == "l");
            parallel_for(n, std::max(1, PARALLEL_THRESHOLD_SIMPLE / (n + 1)),
            [&A, n, lower](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i) {
                    if (lower) for (size_t j = i + 1; j < n; ++j) A(i, j) = T(0);
                    else for (size_t j = 0; j < i; ++j) A(i, j) = T(0);;
                };
            });
        };

        template<typename T, Layout L>
        LINALG_INLINE void sub_copy_in(const Matrix<T,L>& A, Matrix<T,L>& dst, size_t r0, size_t c0) {
            const size_t nr = dst.rows(), nc = dst.cols();
            const size_t thr = std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (nc + 1));
            if constexpr (L == Layout::RowMajor) {
                parallel_for(nr, thr, [&](size_t s, size_t e) {
                    for (size_t i = s; i < e; ++i) {
                        T* LINALG_RESTRICT dp = dst.data() + i * dst.stride();
                        const T* LINALG_RESTRICT sp = A.data() + (r0+i)*A.stride() + c0;
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < nc; ++j) dp[j] = sp[j];
                    };
                });
            } else {
                parallel_for(nc, thr, [&](size_t s, size_t e) {
                    for (size_t j = s; j < e; ++j) {
                        T* LINALG_RESTRICT dp = dst.data() + j * dst.stride();
                        const T* LINALG_RESTRICT sp = A.data() + (c0+j)*A.stride() + r0;
                        LINALG_VECTORIZE
                        for (size_t i = 0; i < nr; ++i) dp[i] = sp[i];
                    };
                });
            };
        };
        
        template<typename T, Layout L>
        LINALG_INLINE void sub_copy_out(Matrix<T,L>& A, const Matrix<T,L>& src, size_t r0, size_t c0) {
            const size_t nr = src.rows(), nc = src.cols();
            const size_t thr = std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (nc + 1));
            if constexpr (L == Layout::RowMajor) {
                parallel_for(nr, thr, [&](size_t s, size_t e) {
                    for (size_t i = s; i < e; ++i) {
                        const T* LINALG_RESTRICT sp = src.data() + i * src.stride();
                        T* LINALG_RESTRICT dp = A.data() + (r0+i)*A.stride() + c0;
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < nc; ++j) dp[j] = sp[j];
                    };
                });
            } else {
                parallel_for(nc, thr, [&](size_t s, size_t e) {
                    for (size_t j = s; j < e; ++j) {
                        const T* LINALG_RESTRICT sp = src.data() + j * src.stride();
                        T* LINALG_RESTRICT dp = A.data() + (c0+j)*A.stride() + r0;
                        LINALG_VECTORIZE
                        for (size_t i = 0; i < nr; ++i) dp[i] = sp[i];
                    };
                });
            };
        };

        template<typename T, Layout L>
        bool chol_factor_lower(Matrix<T, L>& A) {
            const size_t n = A.rows();
            for (size_t k = 0; k < n; k += L2_BLOCK) {
                const size_t kb = std::min(L2_BLOCK, n - k);
                // Step 1: unblocked panel
                if (!chol_unblocked_lower(A, k, kb)) return false;
                if (k + kb >= n) break;
                const size_t trail = n - (k + kb);
                // Step 2: trsm  L21 = A21 * L11^{-H}
                // L11 is at A[k:k+kb, k:k+kb] - wrap as a non-owning view - trsm's MatExpr overload accepts this directly.
                T* const ap = A.data();
                const size_t lda = A.stride();
                const T* l11_ptr = ap + k*lda+k;
                MatrixView<T, L, false, false, false> L11_view(l11_ptr, kb, kb, lda);
                Matrix<T, L> A21(trail, kb);
                sub_copy_in(A, A21, k + kb, k);
                trsm('R', 'L', 'C', 'N', T(1), expr(L11_view), A21);
                sub_copy_out(A, A21, k + kb, k);
                // Step 3: herk  A22 -= L21 * L21^H.
                Matrix<T, L> A22(trail, trail);
                sub_copy_in(A, A22, k + kb, k + kb);
                herk('L', 'N', real_type_t<T>(-1), expr(A21), real_type_t<T>(1), A22);
                sub_copy_out(A, A22, k + kb, k + kb);
            };
            return true;
        };

        template<typename T, Layout L>
        bool chol_factor_upper(Matrix<T, L>& A) {
            const size_t n = A.rows();
            for (size_t k = 0; k < n; k += L2_BLOCK) {
                const size_t kb = std::min(L2_BLOCK, n - k);
                if (!chol_unblocked_upper(A, k, kb)) return false;
                if (k + kb >= n) break;
                const size_t trail = n - (k + kb);

                T* const ap = A.data();
                const size_t lda = A.stride();
                const T* u11_ptr = ap + k * lda + k;
                MatrixView<T, L, false, false, false> U11_view(u11_ptr, kb, kb, lda);
                Matrix<T, L> A12(kb, trail);
                sub_copy_in(A, A12, k, k + kb);
                trsm('L', 'U', 'C', 'N', T(1), expr(U11_view), A12);
                sub_copy_out(A, A12, k, k + kb);
        
                Matrix<T, L> A22(trail, trail);
                sub_copy_in(A, A22, k + kb, k + kb);
                herk('U', 'C', real_type_t<T>(-1), expr(A12), real_type_t<T>(1), A22);
                sub_copy_out(A, A22, k + kb, k + kb);
            };
            return true;
        };
    };
};