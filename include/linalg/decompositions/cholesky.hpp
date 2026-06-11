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
                    d -= static_cast<chol_real_t<T>>(std::norm(L == Layout::RowMajor ? 
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
                    d -= static_cast<chol_real_t<T>>(std::norm(L == Layout::RowMajor ? 
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
    };
};