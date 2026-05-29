#pragma once

#include <linalg/blas/level1.hpp>
#include <linalg/blas/level2.hpp>

namespace linalg {
    namespace detail {
        // Real type traits
		template<typename T> struct real_type_impl { using type = T; };
        template<typename T> struct real_type_impl<std::complex<T>> { using type = T; };
        template<typename T>
        using real_type_t = typename real_type_impl<std::remove_cvref_t<T>>::type;
        namespace kernels {
            template<typename T>
            LINALG_INLINE void gemm_microkernel_row(
                const T* LINALG_RESTRICT A, const T* LINALG_RESTRICT B,
                T* LINALG_RESTRICT C, T alpha,
                size_t lda, size_t ldb, size_t ldc,
                size_t i0, size_t i1,
                size_t j0, size_t j1,
                size_t k0, size_t k1) {

                for (size_t i = i0; i < i1; ++i) {
                    const T* LINALG_RESTRICT a_row = A + i * lda;
                    T* LINALG_RESTRICT c_row = C + i * ldc;
                    /*
                    for (size_t k = k0; k < k1; ++k) {
                        const T a_ik  = alpha * a_row[k];
                        const T* LINALG_RESTRICT b_row = B + k * ldb;
                        LINALG_PREFETCH(b_row + ldb, 0, 2);
                        LINALG_VECTORIZE
                        for (size_t j = j0; j < j1; ++j) c_row[j] += a_ik * b_row[j];
                    };
                    */
                    
                    for (size_t k = k0; k < k1; ++k) {
                        const T a_ik  = alpha * a_row[k];
                        const T* b_row = B + k * ldb;
                        size_t j = j0;
                        for (; j + 8 <= j1; j += 8) {
                            c_row[j] += a_ik * b_row[j];
                            c_row[j+1] += a_ik * b_row[j+1];
                            c_row[j+2] += a_ik * b_row[j+2];
                            c_row[j+3] += a_ik * b_row[j+3];
                            c_row[j+4] += a_ik * b_row[j+4];
                            c_row[j+5] += a_ik * b_row[j+5];
                            c_row[j+6] += a_ik * b_row[j+6];
                            c_row[j+7] += a_ik * b_row[j+7];
                        };
                        for (; j < j1; ++j) c_row[j] += a_ik * b_row[j];
                    };
                    
                };
            };
        
            template<typename T>
            LINALG_INLINE void gemm_microkernel_col(
                const T* LINALG_RESTRICT A, const T* LINALG_RESTRICT B,
                T* LINALG_RESTRICT C, T alpha,
                size_t lda, size_t ldb, size_t ldc,
                size_t i0, size_t i1,
                size_t j0, size_t j1,
                size_t k0, size_t k1) {
                for (size_t j = j0; j < j1; ++j) {
                    const T* LINALG_RESTRICT b_col = B + j * ldb;
                    T* LINALG_RESTRICT c_col = C + j * ldc;
                    for (size_t k = k0; k < k1; ++k) {
                        /*
                        const T b_kj  = alpha * b_col[k];
                        const T* LINALG_RESTRICT a_col = A + k * lda;
                        LINALG_PREFETCH(a_col + lda, 0, 2);  // prefetch next a_col
                        LINALG_VECTORIZE
                        for (size_t i = i0; i < i1; ++i) c_col[i] += a_col[i] * b_kj;
                        */

                        // Auto-vectorizer does not pay off well, with 8-wide unroll outperforming for every dtype.
                        
                        const T b_kj  = alpha * b_col[k];
                        const T* a_col = A + k * lda;
                        size_t i = i0;
                        for (; i + 8 <= i1; i += 8) {
                            c_col[i] += a_col[i] * b_kj;
                            c_col[i+1] += a_col[i+1] * b_kj;
                            c_col[i+2] += a_col[i+2] * b_kj;
                            c_col[i+3] += a_col[i+3] * b_kj;
                            c_col[i+4] += a_col[i+4] * b_kj;
                            c_col[i+5] += a_col[i+5] * b_kj;
                            c_col[i+6] += a_col[i+6] * b_kj;
                            c_col[i+7] += a_col[i+7] * b_kj;
                        };
                        for (; i < i1; ++i) c_col[i] += a_col[i] * b_kj;
                        
                    };
                };
            };
        };

        template<typename T, Layout L>
        void gemm_blocked(T alpha, const T* LINALG_RESTRICT a_ptr, size_t lda, const T* LINALG_RESTRICT b_ptr, size_t ldb,
                T* LINALG_RESTRICT c_ptr, size_t ldc, size_t M, size_t N, size_t K) {
            if (M == 0 || N == 0 || K == 0) return;
            const size_t bs = L1_BLOCK * 2;
            const size_t ni_blocks = (M + bs - 1) / bs;
            const size_t nj_blocks = (N + bs - 1) / bs;
            const size_t nk_blocks = (K + bs - 1) / bs;
            const size_t total_ij = ni_blocks * nj_blocks;
            auto& pool = ThreadPool::instance();
            const size_t num_threads = std::min(pool.thread_count(), total_ij);
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            for (size_t t = 0; t < num_threads; ++t) {
                futures.push_back(pool.enqueue([=]() {
                    for (size_t blk = t; blk < total_ij; blk += num_threads) {
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
                    for (size_t i = rs; i < re; ++i) {
                        T* LINALG_RESTRICT dp = dst.data() + i * dst.stride();
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < N; ++j) dp[j] = static_cast<T>(src(i, j));
                    };
                });
            } else {
                parallel_for(N, 1, [&src, &dst, M](size_t cs, size_t ce) {
                    for (size_t j = cs; j < ce; ++j) {
                        T* LINALG_RESTRICT dp = dst.data() + j * dst.stride();
                        LINALG_VECTORIZE
                        for (size_t i = 0; i < M; ++i) dp[i] = static_cast<T>(src(i, j));
                    };
                });
            };
            return dst;
        };

        template<typename T, Layout L>
        LINALG_INLINE void gemm_direct(T alpha, const T* LINALG_RESTRICT ap, size_t lda, const T* LINALG_RESTRICT bp, size_t ldb, T* LINALG_RESTRICT cp, size_t ldc, size_t M, size_t N, size_t K) {
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
    LINALG_INLINE void gemm(T alpha, const MatExpr<EA>& A_expr, const MatExpr<EB>& B_expr, T beta, Matrix<T, L>& C) {
        const size_t M = A_expr.self().rows();
        const size_t N = B_expr.self().cols();
        const size_t K = A_expr.self().cols();
        BOUNDS_CHECK(M == C.rows() && N == C.cols() && K == B_expr.self().rows());
        if (M == 0 || N == 0 || K == 0) return;
        T* LINALG_RESTRICT cp = detail::assume_aligned<64>(C.data());
        // Apply beta: C is always tight (Matrix is flat-packed)
        if (beta == T(0)) std::fill(cp, cp + M * N, T(0));
        else if (beta != T(1)) {
            parallel_for(M * N, PARALLEL_THRESHOLD_SIMPLE, [cp, beta](size_t s, size_t e) {
                LINALG_VECTORIZE for (size_t i = s; i < e; ++i) cp[i] *= beta;
            });
        };
        if (alpha == T(0)) return;
        // Fast path: extract pointer
        auto a_info = detail::raw_mat_info<T>(A_expr);
        auto b_info = detail::raw_mat_info<T>(B_expr);
        // Materialise only the operand(s) that could not yield a raw pointer
        Matrix<T, L> A_tmp, B_tmp;
        if (!a_info) A_tmp = detail::materialise<T, L>(A_expr);
        if (!b_info) B_tmp = detail::materialise<T, L>(B_expr);
        const T* LINALG_RESTRICT ap = detail::assume_aligned<64>(a_info ? a_info->data : A_tmp.data());
        const T* LINALG_RESTRICT bp = detail::assume_aligned<64>(b_info ? b_info->data : B_tmp.data());
        const size_t lda = a_info ? a_info->lda : A_tmp.stride();
        const size_t ldb = b_info ? b_info->lda : B_tmp.stride();
        detail::gemm_direct<T, L>(alpha, ap, lda, bp, ldb, cp, C.stride(), M, N, K);
    };

    namespace detail {
        // In-place triangular solve on a contiguous vector
        template<typename T, typename AM>
        LINALG_INLINE void trsm_col_solve(char uplo, char trans, char diag, const AM& A, T* xp, size_t N) {
            const bool upper = (uplo  == 'U' || uplo  == 'u');
            const bool unit = (diag  == 'U' || diag  == 'u');
            const bool do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
            const bool do_conj = (trans == 'C' || trans == 'c');
            auto Aij = [&](size_t i, size_t j) -> T {
                T v = static_cast<T>(A(i, j));
                return do_conj ? conj(v) : v;
            };
            if (!do_trans) {
                if (upper) {
                    for (size_t ii = 0; ii < N; ++ii) {
                        const size_t i = N - 1 - ii;
                        T s = xp[i];
                        LINALG_VECTORIZE
                        for (size_t j = i + 1; j < N; ++j) s -= static_cast<T>(A(i, j)) * xp[j];
                        xp[i] = unit ? s : s / static_cast<T>(A(i, i));
                    }
                } else {
                    for (size_t i = 0; i < N; ++i) {
                        T s = xp[i];
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < i; ++j) s -= static_cast<T>(A(i, j)) * xp[j];
                        xp[i] = unit ? s : s / static_cast<T>(A(i, i));
                    }
                }
            } else {
                // A^T or A^H: access A(j,i) instead of A(i,j), flips triangle direction
                if (upper) {
                    // Upper^T acts as lower -> forward substitution
                    for (size_t i = 0; i < N; ++i) {
                        T s = xp[i];
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < i; ++j) s -= Aij(j, i) * xp[j];
                        T d = unit ? T(1) : (do_conj ? conj(static_cast<T>(A(i, i))) : static_cast<T>(A(i, i)));
                        xp[i] = unit ? s : s / d;
                    }
                } else {
                    // Lower^T acts as upper -> backward substitution
                    for (size_t ii = 0; ii < N; ++ii) {
                        const size_t i = N - 1 - ii;
                        T s = xp[i];
                        LINALG_VECTORIZE
                        for (size_t j = i + 1; j < N; ++j) s -= Aij(j, i) * xp[j];
                        T d = unit ? T(1) : (do_conj ? conj(static_cast<T>(A(i, i))) : static_cast<T>(A(i, i)));
                        xp[i] = unit ? s : s / d;
                    };
                };
            };
        };

        // Block width for the RHS panel in the side='L' path.
        static constexpr size_t TRSM_RHS_BLOCK = 64;

        template<typename T, Layout L, typename AM>
        void trsm_impl(char side, char uplo, char trans, char diag, T alpha,  const AM& A, T* B_ptr, size_t ldb, size_t M, size_t N_rhs) {
            if (M == 0 || N_rhs == 0) return;
            // Scale B by alpha (beta=0 fills unconditionally)
            if (alpha == T(0)) {
                if constexpr (L == Layout::RowMajor)
                    parallel_for(M, PARALLEL_THRESHOLD_SIMPLE, [B_ptr, ldb, N_rhs](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i)
                            std::fill(B_ptr + i*ldb, B_ptr + i*ldb + N_rhs, T(0));
                    });
                else
                    parallel_for(N_rhs, PARALLEL_THRESHOLD_SIMPLE, [B_ptr, ldb, M](size_t cs, size_t ce) {
                        for (size_t j = cs; j < ce; ++j)
                            std::fill(B_ptr + j*ldb, B_ptr + j*ldb + M, T(0));
                    });
                return;
            };
            if (alpha != T(1)) {
                if constexpr (L == Layout::RowMajor) {
                    parallel_for(M, PARALLEL_THRESHOLD_SIMPLE, [B_ptr, ldb, N_rhs, alpha](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i) {
                            T* LINALG_RESTRICT row = B_ptr + i * ldb;
                            LINALG_VECTORIZE
                            for (size_t j = 0; j < N_rhs; ++j) row[j] *= alpha;
                        };
                    });
                } else {
                    parallel_for(N_rhs, PARALLEL_THRESHOLD_SIMPLE, [B_ptr, ldb, M, alpha](size_t cs, size_t ce) {
                        for (size_t j = cs; j < ce; ++j) {
                            T* LINALG_RESTRICT col = B_ptr + j * ldb;
                            LINALG_VECTORIZE
                            for (size_t i = 0; i < M; ++i) col[i] *= alpha;
                        };
                    });
                };
            };
            const bool left = (side == 'L' || side == 'l');
            auto& pool = ThreadPool::instance();
            if (left) {
                /*
                // Each column of B is an independent trsv
                const size_t num_threads = std::min(pool.thread_count(), N_rhs);
                std::vector<std::future<void>> futures; 
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &A]() {
                        Vector<T> buf(M);
                        T* LINALG_RESTRICT bp = detail::assume_aligned<64>(buf.data());
                        for (size_t j = t; j < N_rhs; j += num_threads) {
                            if constexpr (L == Layout::RowMajor) {
                                // Gather column j from row-major B into buf
                                for (size_t i = 0; i < M; ++i) bp[i] = B_ptr[i*ldb + j];
                                trsm_col_solve(uplo, trans, diag, A, bp, M);
                                // Scatter back
                                for (size_t i = 0; i < M; ++i) B_ptr[i*ldb + j] = bp[i];
                            } else {
                                // Column-major: column j is already contiguous
                                trsm_col_solve(uplo, trans, diag, A, B_ptr + j*ldb, M);
                            };
                        };
                    }));
                };
                for (auto& f : futures) f.get();
                */
                const size_t n_blocks = (N_rhs + TRSM_RHS_BLOCK - 1) / TRSM_RHS_BLOCK;
                const size_t num_threads = std::min(pool.thread_count(), n_blocks);
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &A]() {
                        // Thread-local column buffer: M rows × up-to-TRSM_RHS_BLOCK cols, stored column-major so trsm_col_solve sees contiguous data.
                        using AlignedBuf = std::vector<T, AlignedAllocator<T>>;
                        AlignedBuf buf(M * TRSM_RHS_BLOCK);
                        for (size_t blk = t; blk < n_blocks; blk += num_threads) {
                            const size_t j0 = blk * TRSM_RHS_BLOCK;
                            const size_t j1 = std::min(j0 + TRSM_RHS_BLOCK, N_rhs);
                            const size_t nb = j1 - j0;  // actual block width
                            T* LINALG_RESTRICT bp = detail::assume_aligned<64>(buf.data());
                            if constexpr (L == Layout::RowMajor) {
                                // Gather B[0:M, j0:j1] into col-major buffer bp[col*M + row]
                                for (size_t i = 0; i < M; ++i) {
                                    const T* LINALG_RESTRICT src = B_ptr + i * ldb + j0;
                                    LINALG_VECTORIZE
                                    for (size_t jj = 0; jj < nb; ++jj) bp[jj * M + i] = src[jj];
                                };
                                // Solve each column of the block (now contiguous)
                                for (size_t jj = 0; jj < nb; ++jj)
                                    trsm_col_solve(uplo, trans, diag, A, bp + jj * M, M);
                                // Scatter back
                                for (size_t i = 0; i < M; ++i) {
                                    T* LINALG_RESTRICT dst = B_ptr + i * ldb + j0;
                                    LINALG_VECTORIZE
                                    for (size_t jj = 0; jj < nb; ++jj) dst[jj] = bp[jj * M + i];
                                };
                            } else {
                                // ColMajor: columns j0...j1-1 are already contiguous segments; use a look-ahead panel update so A is traversed once per block.

                                const bool do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
                                const bool upper = (uplo == 'U' || uplo == 'u');
                                // Determine whether the look-ahead sweep goes forward (k=0...M) or backward (k=M-1...0).  Forward sweep: lower+notrans or upper+trans.
                                const bool forward = (upper == do_trans);
                                // Minimum size to justify the GEMM update overhead
                                const size_t LOOKAHEAD_THRESH = 32;
                                if (M >= LOOKAHEAD_THRESH && nb > 1) {
                                    // Copy the RHS block into a contiguous scratch so gemm_direct always sees a tight leading-dimension layout.
                                    for (size_t jj = 0; jj < nb; ++jj) {
                                        const T* src = B_ptr + (j0 + jj) * ldb;
                                        T* LINALG_RESTRICT dst = bp + jj * M;
                                        LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) dst[i] = src[i];
                                    };
                                    constexpr size_t kb = L2_BLOCK / 2; // reuse blocked size constant
                                    if (forward) {
                                        for (size_t k = 0; k < M; k += kb) {
                                            const size_t ke = std::min(k + kb, M);
                                            const size_t ks = ke - k;  // actual panel height
                                            // Solve the panel rows: bp[k:ke, 0:nb]
                                            for (size_t jj = 0; jj < nb; ++jj)
                                                trsm_col_solve(uplo, trans, diag, A, bp + jj * M + k, ks);
                                            // Look-ahead update
                                            if (ke < M) {
                                                // Build a tight sub-matrix for A[ke:M, k:ke]
                                                const size_t rows_rem = M - ke;
                                                // Extract A panel into a temporary col-major buffer
                                                AlignedBuf a_panel(rows_rem * ks);
                                                T* LINALG_RESTRICT ap = detail::assume_aligned<64>(a_panel.data());
                                                for (size_t kk = 0; kk < ks; ++kk) {
                                                    LINALG_VECTORIZE
                                                    for (size_t ii = 0; ii < rows_rem; ++ii)
                                                        ap[kk * rows_rem + ii] = static_cast<T>(A(ke + ii, k + kk));
                                                };
                                                gemm_direct<T, Layout::ColMajor>(T(-1),
                                                    ap, rows_rem, // A: rows_rem×ks, col-major lda=rows_rem
                                                    bp + k, M, // B: ks*nb, col-major lda=M
                                                    bp + ke, M, // C: rows_rem×nb, col-major lda=M
                                                    rows_rem, nb, ks);
                                            };
                                        };
                                    } else {
                                        // Backward sweep (upper+notrans or lower+trans)
                                        size_t k = M;
                                        while (k > 0) {
                                            const size_t ke = k;
                                            k = (ke > kb) ? ke - kb : 0;
                                            const size_t ks = ke - k;
                                            // Solve bp[k:ke, 0:nb]
                                            for (size_t jj = 0; jj < nb; ++jj)
                                                trsm_col_solve(uplo, trans, diag, A, bp + jj * M + k, ks);
                                            // Update bp[0:k, 0:nb] -= A[0:k, k:ke] * bp[k:ke, 0:nb]
                                            if (k > 0) {
                                                AlignedBuf a_panel(k * ks);
                                                T* LINALG_RESTRICT ap = detail::assume_aligned<64>(a_panel.data());
                                                for (size_t kk = 0; kk < ks; ++kk) {
                                                    LINALG_VECTORIZE
                                                    for (size_t ii = 0; ii < k; ++ii)
                                                        ap[kk * k + ii] = static_cast<T>(A(ii, k + kk));
                                                };
                                                gemm_direct<T, Layout::ColMajor>(T(-1),
                                                    ap, k, // A: k×ks col-major lda=k
                                                    bp + k, M, // B: ks×nb col-major lda=M
                                                    bp, M, // C: k×nb col-major lda=M
                                                    k, nb, ks);
                                            };
                                        };
                                    };
                                    // Write back to B
                                    for (size_t jj = 0; jj < nb; ++jj) {
                                        T* dst = B_ptr + (j0 + jj) * ldb;
                                        const T* LINALG_RESTRICT src = bp + jj * M;
                                        LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) dst[i] = src[i];
                                    };
                                } else {
                                    // Small M or single column: simple column-by-column solve
                                    for (size_t jj = 0; jj < nb; ++jj)
                                        trsm_col_solve(uplo, trans, diag, A, B_ptr + (j0 + jj) * ldb, M);
                                };
                            };
                        };
                    }));
                };
                for (auto& f : futures) f.get();
            } else {
                // Dual: x*op(A)=b is solved as op_dual(A)*x^T=b^T per row.
                const bool conj_sides = (trans == 'C' || trans == 'c');
                //char trans_r = (trans == 'N' || trans == 'n') ? 'T' : (trans == 'T' || trans == 't') ? 'N' : 'N'; // 'C': solve A, conj b/x
                char trans_r = (trans == 'N' || trans == 'n') ? 'T' : 'N';
                const size_t num_threads = std::min(pool.thread_count(), M);
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                const size_t chunk = M / num_threads;
                const size_t rem = M % num_threads;
                size_t offset = 0;
                for (size_t t = 0; t < num_threads; ++t) {
                    const size_t i0 = offset;
                    const size_t i1 = i0 + chunk + (t < rem ? 1 : 0);
                    offset = i1;
                    futures.push_back(pool.enqueue([=, &A]() {
                        // Reusable row buffer — allocated once per thread, not once per row.
                        using AlignedBuf = std::vector<T, AlignedAllocator<T>>;
                        AlignedBuf buf(N_rhs);
                        T* LINALG_RESTRICT bp = detail::assume_aligned<64>(buf.data());
                        for (size_t i = i0; i < i1; ++i) {
                            if constexpr (L == Layout::RowMajor) {
                                T* LINALG_RESTRICT rp = B_ptr + i * ldb;
                                if (conj_sides) { LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) rp[j] = conj(rp[j]); };
                                trsm_col_solve(uplo, trans_r, diag, A, rp, N_rhs);
                                if (conj_sides) { LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) rp[j] = conj(rp[j]); };
                            } else {
                                // Gather row i (stride ldb) into contiguous buffer
                                LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) bp[j] = B_ptr[j*ldb + i];
                                if (conj_sides) { LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) bp[j] = conj(bp[j]); };
                                trsm_col_solve(uplo, trans_r, diag, A, bp, N_rhs);
                                if (conj_sides) { LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) bp[j] = conj(bp[j]); };
                                LINALG_VECTORIZE for (size_t j = 0; j < N_rhs; ++j) B_ptr[j*ldb + i] = bp[j];
                            };
                        };
                    }));
                };
                for (auto& f : futures) f.get();
            };
        };
    };

    // trsm: op(A) * X = alpha * B  (side='L')  or  X * op(A) = alpha * B  (side='R')
    // A is square and triangular
    template<typename T, Layout L, typename EM>
    void trsm(char side, char uplo, char trans, char diag, T alpha, const MatExpr<EM>& A_expr, Matrix<T, L>& B) {
        const bool left = (side == 'L' || side == 'l');
        const size_t N_tri = left ? B.rows() : B.cols();
        BOUNDS_CHECK(A_expr.self().rows() == N_tri && A_expr.self().cols() == N_tri);
        detail::trsm_impl<T, L>(side, uplo, trans, diag, alpha, A_expr.self(), B.data(), B.stride(), B.rows(), B.cols());
    };
 
    template<typename T, Layout L, typename EM>
    void trsm(char side, char uplo, char trans, char diag, T alpha,  const MatExpr<EM>& A_expr, MatrixView<T, L, false, false, true>& B) {
        const bool left = (side == 'L' || side == 'l');
        const size_t N_tri = left ? B.rows() : B.cols();
        BOUNDS_CHECK(A_expr.self().rows() == N_tri && A_expr.self().cols() == N_tri);
        detail::trsm_impl<T, L>(side, uplo, trans, diag, alpha, A_expr.self(), B.data(), B.stride(), B.rows(), B.cols());
    };

    namespace detail {
        // Triangle beta-scaling with beta=0 unconiditional fill
        template<typename T, Layout L>
        LINALG_INLINE void scale_triangle(char uplo, T beta, T* LINALG_RESTRICT cp, size_t ldc, size_t N) {
            if (beta == T(1)) return;
            const bool upper = (uplo == 'U' || uplo == 'u');
            for (size_t i = 0; i < N; ++i) {
                const size_t j0 = upper ? i : 0;
                const size_t j1 = upper ? N : i + 1;
                if constexpr (L == Layout::RowMajor) {
                    T* LINALG_RESTRICT row = cp + i * ldc;
                    if (beta == T(0)) std::fill(row + j0, row + j1, T(0));
                    else LINALG_VECTORIZE for (size_t j = j0; j < j1; ++j) row[j] *= beta;
                } else {
                    for (size_t j = j0; j < j1; ++j) {
                        T& elem = cp[j*ldc+i];
                        elem = (beta == T(0)) ? T(0) : elem * beta;
                    };
                };
            };
        };

        namespace kernels {
            template<typename T, Layout L>
            LINALG_INLINE void syrk_core(char uplo, T alpha, const T* ap, size_t lda, size_t N, size_t K, bool notrans, bool conjugate, T* cp, size_t ldc) {
                const bool upper = (uplo == 'U' || uplo == 'u');
                // Element of A at logical outer-product index (vec, k).
                // notrans -> row  vec of A: A(vec, k)
                // trans -> col  vec of A: A(k, vec)
                auto Aelem = [ap, lda, notrans](size_t vec, size_t k) -> T {
                    if constexpr (L == Layout::RowMajor)
                        return notrans ? ap[vec*lda+k] : ap[k*lda+vec];
                    else
                        return notrans ? ap[k*lda+vec] : ap[vec*lda+k];
                };
                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE,
                    [=, &upper](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i) {
                            const size_t j0 = upper ? i : 0, j1 = upper ? N : i+1;
                            for (size_t j = j0; j < j1; ++j) {
                                T s = T(0);
                                for (size_t k = 0; k < K; ++k)
                                    s += Aelem(i, k) * (conjugate ? conj(Aelem(j, k)) : Aelem(j, k));
                                if constexpr (L == Layout::RowMajor) cp[i*ldc+j] += alpha * s;
                                else cp[j*ldc+i] += alpha * s;
                            };
                        };
                    });
            };
        };

        template<typename T, Layout L, typename EA>
        void syrk_impl(char uplo, char trans, T alpha, const MatExpr<EA>& A_expr, T beta, T* cp, size_t ldc, size_t N, bool conjugate) {
            const bool notrans = (trans == 'N' || trans == 'n');
            const size_t K = notrans ? A_expr.self().cols() : A_expr.self().rows();
            auto a_info = raw_mat_info<T>(A_expr);
            Matrix<T, L> A_tmp;
            if (!a_info) A_tmp = materialise<T, L>(A_expr);
            const T* ap  = a_info ? a_info->data : A_tmp.data();
            size_t lda = a_info ? a_info->lda : A_tmp.stride();
            scale_triangle<T, L>(uplo, beta, cp, ldc, N);
            if (alpha == T(0) || K == 0) return;
            kernels::syrk_core<T, L>(uplo, alpha, ap, lda, N, K, notrans, conjugate, cp, ldc);
        };
    };

    // SYmmetric Rank-K update: C = alpha * op(A) * op(A)^T + beta * C
    template<typename T, Layout L, typename EA>
    void syrk(char uplo, char trans, T alpha, const MatExpr<EA>& A_expr, T beta, Matrix<T, L>& C) {
        const bool notrans = (trans == 'N' || trans == 'n');
        const size_t N = notrans ? A_expr.self().rows() : A_expr.self().cols();
        BOUNDS_CHECK(C.rows() == N && C.cols() == N);
        detail::syrk_impl<T, L>(uplo, trans, alpha, A_expr, beta, C.data(), C.stride(), N, false);
    };
 
    template<typename T, Layout L, typename EA>
    void syrk(char uplo, char trans, T alpha, const MatExpr<EA>& A_expr, T beta, MatrixView<T, L, false, false, true>& C) {
        const bool notrans = (trans == 'N' || trans == 'n');
        const size_t N = notrans ? A_expr.self().rows() : A_expr.self().cols();
        BOUNDS_CHECK(C.rows() == N && C.cols() == N);
        detail::syrk_impl<T, L>(uplo, trans, alpha, A_expr, beta, C.data(), C.stride(), N, false);
    };
    
    // HErmitian Rank-K update: C = alpha * op(A) * op(A)^H + beta * C
    template<typename T, Layout L, typename EA>
    void herk(char uplo, char trans, detail::real_type_t<T> alpha, const MatExpr<EA>& A_expr, detail::real_type_t<T> beta,  Matrix<T, L>& C) {
        const bool notrans = (trans == 'N' || trans == 'n');
        const size_t N = notrans ? A_expr.self().rows() : A_expr.self().cols();
        BOUNDS_CHECK(C.rows() == N && C.cols() == N);
        detail::syrk_impl<T, L>(uplo, trans, static_cast<T>(alpha), A_expr, static_cast<T>(beta), C.data(), C.stride(), N, true);
    };
 
    template<typename T, Layout L, typename EA>
    void herk(char uplo, char trans, detail::real_type_t<T> alpha, const MatExpr<EA>& A_expr, detail::real_type_t<T> beta,  MatrixView<T, L, false, false, true>& C) {
        const bool notrans = (trans == 'N' || trans == 'n');
        const size_t N = notrans ? A_expr.self().rows() : A_expr.self().cols();
        BOUNDS_CHECK(C.rows() == N && C.cols() == N);
        detail::syrk_impl<T, L>(uplo, trans, static_cast<T>(alpha), A_expr, static_cast<T>(beta), C.data(), C.stride(), N, true);
    };
};