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

                    if constexpr (L == Layout::RowMajor) wp[gj * lda + i] = s * inv_diag;
                    else wp[i * lda + gj] = s * inv_diag;
                };
            };
            return true;
        };

        template<typename T, Layout L>
        void zero_off_triangle(Matrix<T, L>& A, char uplo) {
            const size_t n = A.rows();
            const bool lower = (uplo == 'L' || uplo == 'l');
            parallel_for(n, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (n + 1)),
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

        // Triangular inversion.
        constexpr size_t TRTRI_BLOCK = 64;
        
        template<typename T, Layout L>
        LINALG_INLINE
        void trtri_unblocked_lower(Matrix<T, L>& F, size_t k0, size_t nb) {
            for (size_t j = 0; j < nb; ++j) {
                const size_t gj = k0 + j;
                F(gj, gj) = T(1) / F(gj, gj);
                
                for (size_t i = j + 1; i < nb; ++i) {
                    const size_t gi = k0 + i;
                    T s = T(0);
                    for (size_t p = j; p < i; ++p) s += F(gi, k0 + p) * F(k0 + p, gj);
                    F(gi, gj) = -s / F(gi, gi);
                };
            };
        };

        template<typename T, Layout L>
        LINALG_INLINE
        void trtri_unblocked_upper(Matrix<T, L>& F, size_t k0, size_t nb) {
            for (size_t jj = 0; jj < nb; ++jj) {
                const size_t j = nb - 1 - jj;
                const size_t gj = k0 + j;
                F(gj, gj) = T(1) / F(gj, gj);
                
                for (size_t ii = 0; ii < j; ++ii) {
                    const size_t i = j - 1 - ii;
                    const size_t gi = k0 + i;
                    T s = T(0);
                    for (size_t p = i + 1; p <= j; ++p) s += F(gi, k0 + p) * F(k0 + p, gj);
                    F(gi, gj) = -s / F(gi, gi);
                };
            };
        };

        template<typename T, Layout L>
        void trtri_lower(Matrix<T, L>& F) {
            const size_t n = F.rows();
            for (size_t k = 0; k < n; k += TRTRI_BLOCK) {
                const size_t kb = std::min(TRTRI_BLOCK, n - k);
                // Invert the diagonal block F[k:k+kb, k:k+kb] in place (unblocked).
                trtri_unblocked_lower(F, k, kb);
                if (k + kb >= n) break;
                const size_t trail = n - (k + kb);
                // F21_new = -F22^{-1} * F21 * F11^{-1}.
                // Step A: tmp = F21 * F11^{-1}.
                // Solved as a right-side lower trsm: X * F11 = F21, i.e. trsm('R','L','N','N', 1, F11, F21).
                Matrix<T, L> F21(trail, kb);
                sub_copy_in(F, F21, k + kb, k);

                T* const fp = F.data();
                const size_t lda = F.stride();
                const T* f11_ptr = fp + k * lda + k;
                MatrixView<T, L, false, false, false> F11_view(f11_ptr, kb, kb, lda);
        
                trsm('R', 'L', 'N', 'N', T(1), expr(F11_view), F21);
                // F21 now holds F21_orig * F11^{-1}
                // Step B: F21_new = -F22^{-1} * (F21_orig * F11^{-1})
                // This is a left-side lower trsm against F22 (which has not yet been inverted: it holds the original lower factor of F).
                const T* f22_ptr = fp + (k + kb) * lda + ( k + kb);
                MatrixView<T, L, false, false, false> F22_view(f22_ptr, trail, trail, lda);
        
                trsm('L', 'L', 'N', 'N', T(-1), expr(F22_view), F21);
                sub_copy_out(F, F21, k + kb, k);
            };
        };
        
        template<typename T, Layout L>
        void trtri_upper(Matrix<T, L>& F) {
            const size_t n = F.rows();
            // Process right-to-left: invert the last diagonal block, then propagate left.
            const size_t nblocks = (n + TRTRI_BLOCK - 1) / TRTRI_BLOCK;
            for (size_t bb = 0; bb < nblocks; ++bb) {
                const size_t b = nblocks - 1 - bb;
                const size_t k = b * TRTRI_BLOCK;
                const size_t kb = std::min(TRTRI_BLOCK, n - k);

                trtri_unblocked_upper(F, k, kb);
                if (k == 0) break;
        
                // U12_new = -U11^{-1} * U12 * U22^{-1}.
                // Step A: tmp = U12 * U22^{-1}  (right-side upper trsm).
                Matrix<T, L> U12(k, kb);
                sub_copy_in(F, U12, 0, k);
        
                T* const fp = F.data();
                const size_t lda = F.stride();
                const T* u22_ptr = fp +  k * lda + k;
                MatrixView<T, L, false, false, false> U22_view(u22_ptr, kb, kb, lda);
        
                trsm('R', 'U', 'N', 'N', T(1), expr(U22_view), U12);
        
                // Step B: U12_new = -U11^{-1} * tmp.
                const T* u11_ptr = fp;
                MatrixView<T, L, false, false, false> U11_view(u11_ptr, k, k, lda);
        
                trsm('L', 'U', 'N', 'N', T(-1), expr(U11_view), U12);
                sub_copy_out(F, U12, 0, k);
            };
        };
    };
};