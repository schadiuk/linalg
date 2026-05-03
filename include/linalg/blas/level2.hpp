# pragma once

#include <linalg/blas/level1.hpp>
#include <optional>

namespace linalg {
    namespace detail {
        // Resolving a VecExpr object to a const T* pointer for use in pointer-based loops
        // Returns nullopt when the expression is not a contiguous Vector or unit-stride VectorView, forcing the caller to materialise first
        template<typename T>
        struct VecInfo { const T* data; size_t stride; };

        template<typename T>
        std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<Vector<T>>& e) {
            return VecInfo<T>(e.self().data(), 1);
        };

        template<typename T>
        std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<VectorView<T, false>>& e) {
            return VecInfo<T>(e.self().data(), e.self().stride());
        };

        template<typename T>
        std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<VectorView<T, true>>& e) {
            return VecInfo<T>(e.self().data(), e.self().stride());
        };

        template<typename T, typename E>
        std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<E>&) {
            return std::nullopt;
        };

        // Internal materialisation helper
        template<typename T, typename EV>
        Vector<T> materialise(const VecExpr<EV>& v) {
            const auto& vv = v.self();
            const size_t n = vv.size();
            Vector<T> out(n);
            parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&vv, &out](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i)
                    out[i] = static_cast<T>(vv(i));
            });
            return out;
        };

        // Namespace containing optimised kernels
        namespace kernels {
            // Fused pointer-level gemv kernel for RowMajor layout
            template<typename T>
            void gemv_kernel_row(T alpha, const T* A, size_t lda, const T* x, size_t incx, T beta, T* y, size_t M, size_t N) {
                parallel_for(M, PARALLEL_THRESHOLD_COMPUTE,
                        [=](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i) {
                        const T* a_row = A + i * lda;
                        T acc = T(0);
                        for (size_t j = 0; j < N; ++j) acc += a_row[j] * x[j * incx];
                        y[i] = alpha * acc + beta * y[i];
                    };
                });
            };

            // ColMajor kernel
            template<typename T>
            void gemv_kernel_col(T alpha, const T* A, size_t lda, const T* x, size_t incx, T beta, T* y, size_t M, size_t N) {
                // Scale y first, then accumulate column contributions
                parallel_for(M, PARALLEL_THRESHOLD_SIMPLE, [y, beta, M](size_t s, size_t e) {
                    for (size_t i = s; i < e; ++i) y[i] *= beta;
                });
                // Serial (alas!) over columns to avoid concurrent y[i] updates
                for (size_t j = 0; j < N; ++j) {
                    const T* a_col = A + j * lda;
                    const T  xj   = alpha * x[j * incx];
                    for (size_t i = 0; i < M; ++i) y[i] += a_col[i] * xj;
                };
            };
        };

        // Dispatch: extracts raw pointers whenever possible, else materialises
        template<typename T, Layout L, typename EM, typename EV>
        void gemv_impl(T alpha, const MatExpr<EM>& a_expr, const VecExpr<EV>& x_expr, T beta, T* y_ptr, size_t M, size_t N) {
            // Materialise x if needed
            auto x_info = detail::raw_vec_ptr<T>(x_expr);
            Vector<T> x_tmp;
            const T* x_ptr; size_t incx;
            if (x_info) { x_ptr = x_info->data; incx = x_info->stride; }
            else { x_tmp = materialise<T>(x_expr); x_ptr = x_tmp.data(); incx = 1; };
            // Try to get raw A pointer
            const T* a_ptr = nullptr;
            size_t   lda   = 0;
            if constexpr (std::is_same_v<EM, Matrix<T, L>>) {
                a_ptr = a_expr.self().data();
                lda = a_expr.self().stride();
            } else if constexpr (std::is_same_v<EM, MatRef<T, L>>) {
                a_ptr = a_expr.self().mat.data();
                lda = a_expr.self().mat.stride();
            };
    
            if (a_ptr) {
                if constexpr (L == Layout::RowMajor)
                    kernels::gemv_kernel_row(alpha, a_ptr, lda, x_ptr, incx, beta, y_ptr, M, N);
                else kernels::gemv_kernel_col(alpha, a_ptr, lda, x_ptr, incx, beta, y_ptr, M, N);
            } else {
                // Materialise A then recurse with concrete type
                Matrix<T, L> A_tmp(a_expr.self().rows(), a_expr.self().cols());
                A_tmp = a_expr;
                if constexpr (L == Layout::RowMajor)
                    kernels::gemv_kernel_row(alpha, A_tmp.data(), A_tmp.stride(), x_ptr, incx, beta, y_ptr, M, N);
                else kernels::gemv_kernel_col(alpha, A_tmp.data(), A_tmp.stride(), x_ptr, incx, beta, y_ptr, M, N);
            };
        };
    };

    // y = alpha * A * x + beta * y. Public interface output: Vector<T>
    template<typename T, Layout L = Layout::RowMajor, typename EM, typename EV>
    void gemv(T alpha, const MatExpr<EM>& A, const VecExpr<EV>& x, T beta,  Vector<T>& y) {
        const size_t M = A.self().rows();
        const size_t N = A.self().cols();
        BOUNDS_CHECK(M == y.size() && N == x.self().size());
        if (M == 0 || N == 0) return;
        detail::gemv_impl<T, L>(alpha, A, x, beta, y.data(), M, N);
    };
    
    // Public interface output: VectorView<T,true>
    template<typename T, Layout L = Layout::RowMajor, typename EM, typename EV>
    void gemv(T alpha, const MatExpr<EM>& A, const VecExpr<EV>& x, T beta,  VectorView<T, true>& y) {
        const size_t M = A.self().rows();
        const size_t N = A.self().cols();
        BOUNDS_CHECK(M == y.size() && N == x.self().size() && y.stride() == 1);
        if (M == 0 || N == 0) return;
        // VectorView with unit stride: data() gives a contiguous T*
        detail::gemv_impl<T, L>(alpha, A, x, beta, const_cast<T*>(y.data()), M, N);
    };
};