#pragma once

#include <linalg/blas/level1.hpp>
#include <linalg/blas/level2.hpp>

namespace linalg {
    namespace detail {
        namespace kernels {
            // Microkernels: accumulate alpha * A[i_block,k_block] * B[k_block,j_block] into C[i_block,j_block].  4-way unrolled loop
            // These operate on contiguous raw pointers only (thus enabling __restrict)
            template<typename T>
            inline void gemm_microkernel_row(
                const T* __restrict A, const T* __restrict B,
                T* __restrict C, T alpha,
                size_t lda, size_t ldb, size_t ldc,
                size_t i0, size_t i1,
                size_t j0, size_t j1,
                size_t k0, size_t k1) {

                for (size_t i = i0; i < i1; ++i) {
                    const T* a_row = A + i * lda;
                    T* c_row = C + i * ldc;
                    for (size_t k = k0; k < k1; ++k) {
                        const T a_ik  = alpha * a_row[k];
                        const T* b_row = B + k * ldb;
                        size_t j = j0;
                        for (; j + 4 <= j1; j += 4) {
                            c_row[j] += a_ik * b_row[j];
                            c_row[j+1] += a_ik * b_row[j+1];
                            c_row[j+2] += a_ik * b_row[j+2];
                            c_row[j+3] += a_ik * b_row[j+3];
                        };
                        for (; j < j1; ++j) c_row[j] += a_ik * b_row[j];
                    };
                };
            };
        
            template<typename T>
            inline void gemm_microkernel_col(
                const T* __restrict A, const T* __restrict B,
                T* __restrict C, T alpha,
                size_t lda, size_t ldb, size_t ldc,
                size_t i0, size_t i1,
                size_t j0, size_t j1,
                size_t k0, size_t k1) {
                for (size_t j = j0; j < j1; ++j) {
                    const T* b_col = B + j * ldb;
                    T* c_col = C + j * ldc;
                    for (size_t k = k0; k < k1; ++k) {
                        const T b_kj  = alpha * b_col[k];
                        const T* a_col = A + k * lda;
                        size_t i = i0;
                        for (; i + 4 <= i1; i += 4) {
                            c_col[i] += a_col[i] * b_kj;
                            c_col[i+1] += a_col[i+1] * b_kj;
                            c_col[i+2] += a_col[i+2] * b_kj;
                            c_col[i+3] += a_col[i+3] * b_kj;
                        }
                        for (; i < i1; ++i) c_col[i] += a_col[i] * b_kj;
                    };
                };
            };
        };

        template<typename T, Layout L>
        void gemm_blocked(T alpha, const T* a_ptr, size_t lda, const T* b_ptr, size_t ldb,
                T* c_ptr, size_t ldc, size_t M, size_t N, size_t K) {
            if (M == 0 || N == 0 || K == 0) return;
            const size_t bs = L1_BLOCK;
            const size_t ni_blocks = (M + bs - 1) / bs;
            const size_t nj_blocks = (N + bs - 1) / bs;
            const size_t nk_blocks = (K + bs - 1) / bs;
            const size_t total_ij = ni_blocks * nj_blocks;
            auto& pool = ThreadPool::instance();
            const size_t nthrd = std::min(pool.thread_count(), total_ij);
            std::vector<std::future<void>> futures;
            futures.reserve(nthrd);
            for (size_t t = 0; t < nthrd; ++t) {
                futures.push_back(pool.enqueue([=]() {
                    for (size_t blk = t; blk < total_ij; blk += nthrd) {
                        const size_t ib = blk / nj_blocks;
                        const size_t jb = blk % nj_blocks;
                        const size_t i0 = ib * bs, i1 = std::min(i0 + bs, M);
                        const size_t j0 = jb * bs, j1 = std::min(j0 + bs, N);
                        for (size_t kb = 0; kb < nk_blocks; ++kb) {
                            const size_t k0 = kb * bs, k1 = std::min(k0 + bs, K);
                            if constexpr (L == Layout::RowMajor)
                                kernels::gemm_microkernel_row(a_ptr, b_ptr, c_ptr, alpha,
                                                            lda, ldb, ldc,
                                                            i0, i1, j0, j1, k0, k1);
                            else
                                kernels::gemm_microkernel_col(a_ptr, b_ptr, c_ptr, alpha,
                                                            lda, ldb, ldc,
                                                            i0, i1, j0, j1, k0, k1);
                        };
                    };
                }));
            };
            for (auto& f : futures) f.get();
        };

        // Parallelised materialisation
        template<typename T, Layout L, typename E>
        Matrix<T, L> materialise(const MatExpr<E>& e) {
            const auto& src = e.self();
            const size_t M  = src.rows(), N = src.cols();
            Matrix<T, L> dst(M, N);
            if constexpr (L == Layout::RowMajor) {
                parallel_for(M, 1, [&src, &dst, N](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < N; ++j)
                            dst(i, j) = static_cast<T>(src(i, j));
                });
            } else {
                parallel_for(N, 1, [&src, &dst, M](size_t cs, size_t ce) {
                    for (size_t j = cs; j < ce; ++j)
                        for (size_t i = 0; i < M; ++i)
                            dst(i, j) = static_cast<T>(src(i, j));
                });
            };
            return dst;
        };

        template<typename T, Layout L>
        void gemm_direct(T alpha, const T* ap, size_t lda, const T* bp, size_t ldb, T* cp, size_t ldc, size_t M, size_t N, size_t K) {
            if (M * N * K < static_cast<size_t>(PARALLEL_THRESHOLD_COMPUTE) * 10) {
                if constexpr (L == Layout::RowMajor)
                    kernels::gemm_microkernel_row(ap, bp, cp, alpha, lda, ldb, ldc, 0, M, 0, N, 0, K);
                else
                    kernels::gemm_microkernel_col(ap, bp, cp, alpha, lda, ldb, ldc, 0, M, 0, N, 0, K);
                return;
            };
            gemm_blocked<T, L>(alpha, ap, lda, bp, ldb, cp, ldc, M, N, K);
        };
    };

    

    // GEneral Matrix Multiplication: C = alpha * A * B + beta * C
    template<typename T, Layout L, typename EA, typename EB>
    void gemm(T alpha, const MatExpr<EA>& A_expr, const MatExpr<EB>& B_expr, T beta, Matrix<T, L>& C) {
        const size_t M = A_expr.self().rows();
        const size_t N = B_expr.self().cols();
        const size_t K = A_expr.self().cols();
        BOUNDS_CHECK(M == C.rows() && N == C.cols() && K == B_expr.self().rows());
        if (M == 0 || N == 0 || K == 0) return;
        T* cp = C.data();
        // Apply beta: C is always tight (Matrix is flat-packed)
        if (beta == T(0)) std::fill(cp, cp + M * N, T(0));
        else if (beta != T(1)) {
            parallel_for(M * N, PARALLEL_THRESHOLD_SIMPLE, [cp, beta](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) cp[i] *= beta;
            });
        }
        if (alpha == T(0)) return;
        // Fast path: extract pointer
        auto a_info = detail::raw_mat_info<T>(A_expr);
        auto b_info = detail::raw_mat_info<T>(B_expr);
        // Materialise only the operand(s) that could not yield a raw pointer
        Matrix<T, L> A_tmp, B_tmp;
        if (!a_info) A_tmp = detail::materialise<T, L>(A_expr);
        if (!b_info) B_tmp = detail::materialise<T, L>(B_expr);
        const T* ap = a_info ? a_info->data : A_tmp.data();
        size_t lda = a_info ? a_info->lda : A_tmp.stride();
        const T* bp = b_info ? b_info->data : B_tmp.data();
        size_t ldb = b_info ? b_info->lda : B_tmp.stride();
        detail::gemm_direct<T, L>(alpha, ap, lda, bp, ldb, cp, C.stride(), M, N, K);
    };
};