#pragma once

#include <linalg/blas/level3.hpp>
#include <linalg/norms/matrix_norms.hpp>

namespace linalg {
    enum class QRMode {Reduced, Complete, R};

    // Publicly-visible decomposition package
    template<typename T, Layout LL>
    struct QRResult {
        Matrix<T, LL> P;
        Matrix<T, LL> Q;
        Matrix<T, LL> R;
        Vector<size_t> piv;
        size_t rank;
        bool pivoted;
    };

    namespace detail {
        // Returns pair (v, beta): (I - beta*v*v^H)*x = norm(x)*sign(x[0])*e_0
        template<typename T>
        std::pair<Vector<T>, double> householder_reflector(const Vector<T>& x) {
            const size_t n = x.size();
            Vector<T> v = x;
            double sigma = 0.0;
            for (size_t i = 1; i < n; ++i) sigma += std::norm(x[i]);
            const double xnorm = std::sqrt(std::norm(x[0]) + sigma);
            if (xnorm == 0.0) return {v, 0.0};
            T alpha;
            if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
                using R = typename T::value_type;
                const double x0_abs = std::abs(x[0]);
                const T phase = (x0_abs == 0.0) ? T(1) : x[0] / static_cast<R>(x0_abs);
                alpha = -phase * static_cast<R>(xnorm);
            } else {
                alpha = (x[0] >= T(0)) ? -static_cast<T>(xnorm) : static_cast<T>(xnorm);
            };
            v[0] = x[0] - alpha;
            const double vHv  = std::norm(v[0]) + sigma;
            const double beta = (vHv == 0.0) ? 0.0 : 2.0 / vHv;
            return {v, beta};
        };

        template<typename T, Layout L>
        void apply_householder_left(Matrix<T, L>& W, const Vector<T>& v, double beta, size_t k, size_t len, size_t ncols) {
            if (beta == 0.0 || ncols == 0 || len == 0) return;
            const T b = static_cast<T>(beta);
            const size_t lda = W.stride();
            T* const W_base = W.data() + k * lda + k;
            // Materialise conj(v) for the transposed GEMV
            Vector<T> cv(len);
            for (size_t i = 0; i < len; ++i) cv[i] = linalg::conj(v[i]);
            // Step 1: w = W_sub^T * cv (layout-flipped raw GEMV)
            Vector<T> w(ncols, T(0));
            if constexpr (L == Layout::RowMajor)
                kernels::gemv_kernel_col(T(1), W_base, lda, cv.data(), size_t(1), T(0), w.data(), ncols, len);
            else
                kernels::gemv_kernel_row(T(1), W_base, lda, cv.data(), size_t(1), T(0), w.data(), ncols, len);
            // Step 2: W_sub -= beta * v * w^T  (GER; conj already absorbed in w via cv)
            MatrixView<T, L, false, false, true> W_sub(W_base, len, ncols, lda);
            ger(-b, v, w, W_sub);
        };

        template<typename T, Layout L>
        void apply_householder_right(Matrix<T, L>& Q, const Vector<T>& v, double beta, size_t k, size_t len) {
            if (beta == 0.0 || len == 0) return;
            const size_t m = Q.rows();
            if (m == 0) return;
            const T b = static_cast<T>(beta);
            const size_t lda = Q.stride();
            T* const Q_base = Q.data() + (L == Layout::RowMajor ? k : k * lda);
            // Step 1: w = Q_sub * v
            Vector<T> w(m, T(0));
            if constexpr (L == Layout::RowMajor)
                kernels::gemv_kernel_row(T(1), Q_base, lda, v.data(), size_t(1), T(0), w.data(), m, len);
            else
                kernels::gemv_kernel_col(T(1), Q_base, lda, v.data(), size_t(1), T(0), w.data(), m, len);
            // Step 2: Q_sub -= beta * w * v^H  (GERC)
            MatrixView<T, L, false, false, true> Q_sub(Q_base, m, len, lda);
            gerc(-b, w, v, Q_sub);
        };
    };
};