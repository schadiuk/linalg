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

        // Structure yielding a layout-aware raw-pointer descriptor for a matrix block
        template<typename T>
        struct MatInfo { const T* data; size_t lda; Layout layout; };
 
        // Fallback: any generic expression must be materialised
        template<typename T, typename E>
        std::optional<MatInfo<T>> raw_mat_info(const MatExpr<E>&) {
            return std::nullopt;
        };
 
        // Bare Matrix<T,L>
        template<typename T, Layout L>
        std::optional<MatInfo<T>> raw_mat_info(const MatExpr<Matrix<T,L>>& e) {
            return MatInfo<T>{ e.self().data(), e.self().stride(), L };
        };
 
        // MatRef<T,L> - thin const-reference wrapper around a Matrix
        template<typename T, Layout L>
        std::optional<MatInfo<T>> raw_mat_info(const MatExpr<MatRef<T,L>>& e) {
            return MatInfo<T>{ e.self().mat.data(), e.self().mat.stride(), L };
        };
 
        // Non-transposed, non-conjugated MatrixView (Mut = true or false).
        // Any such view has a valid strided-BLAS pointer regardless of tightness:
        // stride() is always the correct leading dimension for the BLAS kernels
        template<typename T, Layout L, bool Mut>
        std::optional<MatInfo<T>> raw_mat_info(const MatExpr<MatViewExpr<T, L, false, false, Mut>>& e) {
            const auto& v = e.self().view;
            return MatInfo<T>{ v.data(), v.stride(), L };
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
                        y[i] = (beta == T(0)) ? alpha * acc : alpha * acc + beta * y[i]; // NaN handling: beta = 0 unconditionally
                    };
                });
            };

            // ColMajor kernel
            template<typename T>
            void gemv_kernel_col(T alpha, const T* A, size_t lda, const T* x, size_t incx, T beta, T* y, size_t M, size_t N) {
                if (beta == T(0)) std::fill(y, y + M, T(0)); // NaN does not affect zeroing
                else if (beta != T(1)) {
                    parallel_for(M, PARALLEL_THRESHOLD_SIMPLE, [y, beta](size_t s, size_t e) {
                        for (size_t i = s; i < e; ++i) y[i] *= beta;
                    });
                }
 
                if (alpha == T(0) || N == 0) return;
                auto& pool = ThreadPool::instance();
                const size_t num_threads = std::min(pool.thread_count(), (N + PARALLEL_THRESHOLD_COMPUTE - 1) / PARALLEL_THRESHOLD_COMPUTE);
                if (num_threads <= 1) {
                    // Serial path: straightforward column accumulation, stride-1 inner loop
                    for (size_t j = 0; j < N; ++j) {
                        const T xj = alpha * x[j * incx];
                        if (xj == T(0)) continue;
                        const T* a_col = A + j * lda;
                        for (size_t i = 0; i < M; ++i) y[i] += a_col[i] * xj;
                    };
                    return;
                };
                // Parallel path: each thread accumulates a subset of columns into a private vector, then the private results are summed into y
                std::vector<std::vector<T>> locals(num_threads, std::vector<T>(M, T(0)));
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &locals]() {
                        T* loc = locals[t].data();
                        for (size_t j = t; j < N; j += num_threads) {
                            const T xj = alpha * x[j * incx];
                            if (xj == T(0)) continue;
                            const T* a_col = A + j * lda;
                            for (size_t i = 0; i < M; ++i) loc[i] += a_col[i] * xj;
                        };
                    }));
                };
                for (auto& f : futures) f.get();
                // Reduce thread-local sums into y
                for (size_t t = 0; t < num_threads; ++t) {
                    const T* loc = locals[t].data();
                    for (size_t i = 0; i < M; ++i) y[i] += loc[i];
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
            // Resolving Matrix
            auto a_info = raw_mat_info<T>(a_expr);
            if (a_info) {
                // Fast path: direct pointer with correct layout
                if (a_info->layout == Layout::RowMajor)
                    kernels::gemv_kernel_row(alpha, a_info->data, a_info->lda, x_ptr, incx, beta, y_ptr, M, N);
                else
                    kernels::gemv_kernel_col(alpha, a_info->data, a_info->lda, x_ptr, incx, beta, y_ptr, M, N);
            } else {
                // Materialise A into Layout L, then dispatch
                Matrix<T, L> A_tmp(M, N);
                A_tmp = a_expr;
                if constexpr (L == Layout::RowMajor)
                    kernels::gemv_kernel_row(alpha, A_tmp.data(), A_tmp.stride(), x_ptr, incx, beta, y_ptr, M, N);
                else
                    kernels::gemv_kernel_col(alpha, A_tmp.data(), A_tmp.stride(), x_ptr, incx, beta, y_ptr, M, N);
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

    namespace detail {
        template<typename T, Layout L, typename EX, typename EY>
        void ger_impl(T alpha, const VecExpr<EX>& x_expr, const VecExpr<EY>& y_expr,
                    T* A_ptr, size_t lda, size_t M, size_t N) {
            // Materialise x and y to avoid repeated expression traversal
            const Vector<T> xv = materialise<T>(x_expr);
            const Vector<T> yv = materialise<T>(y_expr);
            const T* xp = xv.data();
            const T* yp = yv.data();
            if constexpr (L == Layout::RowMajor) {
                // Parallel over rows: each row is independent, stride-1 writes
                parallel_for(M, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i) {
                            const T xi = alpha * xp[i];
                            T* a_row = A_ptr + i * lda;
                            size_t j = 0;
                            for (; j + 4 <= N; j += 4) {
                                a_row[j] += xi * yp[j];
                                a_row[j+1] += xi * yp[j+1];
                                a_row[j+2] += xi * yp[j+2];
                                a_row[j+3] += xi * yp[j+3];
                            };
                            for (; j < N; ++j) a_row[j] += xi * yp[j];
                        };
                    });
            } else {
                // ColMajor: parallel over j (columns), stride-1 writes into a_col[i]
                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t js, size_t je) {
                        for (size_t j = js; j < je; ++j) {
                            const T yj = alpha * yp[j];
                            T* a_col = A_ptr + j * lda;
                            size_t i = 0;
                            for (; i + 4 <= M; i += 4) {
                                a_col[i] += xp[i] * yj;
                                a_col[i+1] += xp[i+1] * yj;
                                a_col[i+2] += xp[i+2] * yj;
                                a_col[i+3] += xp[i+3] * yj;
                            };
                            for (; i < M; ++i) a_col[i] += xp[i] * yj;
                        };
                    });
            };
        };

        template<typename T, Layout L, typename EX, typename EY>
        void gerc_impl(T alpha, const VecExpr<EX>& x_expr, const VecExpr<EY>& y_expr,
                    T* A_ptr, size_t lda, size_t M, size_t N) {
            const Vector<T> xv = materialise<T>(x_expr);
            Vector<T> yv = materialise<T>(y_expr);
            // Conjugate y in place
            for (size_t j = 0; j < N; ++j) yv[j] = conj(yv[j]);
            const T* xp = xv.data();
            const T* yp = yv.data();
            if constexpr (L == Layout::RowMajor) {
                parallel_for(M, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i) {
                            const T xi = alpha * xp[i];
                            T* a_row = A_ptr + i * lda;
                            size_t j = 0;
                            for (; j + 4 <= N; j += 4) {
                                a_row[j] += xi * yp[j];
                                a_row[j+1] += xi * yp[j+1];
                                a_row[j+2] += xi * yp[j+2];
                                a_row[j+3] += xi * yp[j+3];
                            };
                            for (; j < N; ++j) a_row[j] += xi * yp[j];
                        };
                    });
            } else {
                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t js, size_t je) {
                        for (size_t j = js; j < je; ++j) {
                            const T yj = alpha * yp[j];
                            T* a_col = A_ptr + j * lda;
                            size_t i = 0;
                            for (; i + 4 <= M; i += 4) {
                                a_col[i] += xp[i] * yj;
                                a_col[i+1] += xp[i+1] * yj;
                                a_col[i+2] += xp[i+2] * yj;
                                a_col[i+3] += xp[i+3] * yj;
                            };
                            for (; i < M; ++i) a_col[i] += xp[i] * yj;
                        };
                    });
            };
        };
    };

    // General Rank-1 update: A = A + alpha * x * y^T (vector treated as a column)
    template<typename T, Layout L, typename EX, typename EY>
    void ger(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y, Matrix<T, L>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::ger_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };
    
    template<typename T, Layout L, typename EX, typename EY>
    void ger(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y,
            MatrixView<T, L, false, false, true>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::ger_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };

    // General Rank-1 Conjugated update: A = A + alpha * x * conj(y)^T (in-place conjugation)
    template<typename T, Layout L, typename EX, typename EY>
    void gerc(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y, Matrix<T, L>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::gerc_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };
    
    template<typename T, Layout L, typename EX, typename EY>
    void gerc(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y,
            MatrixView<T, L, false, false, true>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::gerc_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };

    // TRiangular SolVe: A * x = b, with result stored in x. Parameters follow BLAS convention
    // uplo: 'U'/'L' (upper/lower triangle of A)
    // diag: 'U'/'N' (unit/non-unit diagonal)
    // x: Vector<T>& (in: rhs b, out: solution)
    template<typename T, typename EM>
    void trsv(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, Vector<T>& x) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x.size() == N);
        if (N == 0) return;
        const bool upper = (uplo  == 'U' || uplo  == 'u');
        const bool unit = (diag  == 'U' || diag  == 'u');
        const bool do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
        const bool do_conj = (trans == 'C' || trans == 'c');
        if (!do_trans) {
            if (upper) {
                for (size_t ii = 0; ii < N; ++ii) {
                    const size_t i = N - 1 - ii;
                    T sum = x[i];
                    for (size_t j = i + 1; j < N; ++j)
                        sum -= static_cast<T>(A(i, j)) * x[j];
                    x[i] = unit ? sum : sum / static_cast<T>(A(i, i));
                };
            } else {
                for (size_t i = 0; i < N; ++i) {
                    T sum = x[i];
                    for (size_t j = 0; j < i; ++j)
                        sum -= static_cast<T>(A(i, j)) * x[j];
                    x[i] = unit ? sum : sum / static_cast<T>(A(i, i));
                }
            }
        } else {
            if (upper) {
                // Forward pass over columns of A^T (= rows of A scanned upward)
                for (size_t j = 0; j < N; ++j) {
                    if (!unit)
                        x[j] /= do_conj ? conj(static_cast<T>(A(j, j))) : static_cast<T>(A(j, j));
                    const T xj = x[j];
                    for (size_t i = 0; i < j; ++i) {
                        T aij = do_conj ? conj(static_cast<T>(A(i, j))) : static_cast<T>(A(i, j));
                        x[i] -= aij * xj;
                    };
                };
            } else {
                for (size_t jj = 0; jj < N; ++jj) {
                    const size_t j = N - 1 - jj;
                    if (!unit)
                        x[j] /= do_conj ? conj(static_cast<T>(A(j, j))) : static_cast<T>(A(j, j));
                    const T xj = x[j];
                    for (size_t i = j + 1; i < N; ++i) {
                        T aij = do_conj ? conj(static_cast<T>(A(i, j))) : static_cast<T>(A(i, j));
                        x[i] -= aij * xj;
                    };
                };
            };
        };
    };

    namespace detail {
        template<typename T, typename EM>
        void trmv_impl(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, T* xp, size_t N) {
            const auto& A  = A_expr.self();
            const bool upper = (uplo == 'U' || uplo == 'u');
            const bool unit = (diag == 'U' || diag == 'u');
            const bool do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
            const bool do_conj = (trans == 'C' || trans == 'c');
            // Accumulate into a zeroed temporary — avoids read-after-write aliasing
            Vector<T> tmp(N, T(0));
            T* tp = tmp.data();
            if (!do_trans) {
                // x <- A * x_orig
                for (size_t i = 0; i < N; ++i) {
                    T sum = T(0);
                    if (upper) {
                        if (!unit) sum = static_cast<T>(A(i, i)) * xp[i];
                        else sum = xp[i];
                        for (size_t j = i + 1; j < N; ++j)
                            sum += static_cast<T>(A(i, j)) * xp[j];
                    } else {
                        for (size_t j = 0; j < i; ++j) sum += static_cast<T>(A(i, j)) * xp[j];
                        if (!unit) sum += static_cast<T>(A(i, i)) * xp[i];
                        else sum += xp[i];
                    };
                    tp[i] = sum;
                };
            } else {
                // x <- A^T * x_orig  (or A^H * x_orig)
                // Scatter x_orig[j] * A[k,j] (or conj(A[k,j])) into tp[k]
                for (size_t j = 0; j < N; ++j) {
                    const T xj = xp[j];
                    // Diagonal contribution
                    T ajj = unit ? T(1) : (do_conj ? conj(static_cast<T>(A(j, j))) : static_cast<T>(A(j, j)));
                    tp[j] += ajj * xj;
                    if (upper) {
                        for (size_t k = 0; k < j; ++k) {
                            T akj = do_conj ? conj(static_cast<T>(A(k, j))) : static_cast<T>(A(k, j));
                            tp[k] += akj * xj;
                        };
                    } else {
                        for (size_t k = j + 1; k < N; ++k) {
                            T akj = do_conj ? conj(static_cast<T>(A(k, j))) : static_cast<T>(A(k, j));
                            tp[k] += akj * xj;
                        };
                    };
                };
            };
            // Copy result back
            std::copy(tp, tp + N, xp);
        };
    };

    // TRiangular Matrix-Vector product: x = op(A) * x
    template<typename T, typename EM>
    void trmv(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, Vector<T>& x) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x.size() == N);
        if (N == 0) return;
        detail::trmv_impl(uplo, trans, diag, A_expr, x.data(), N);
    };
 
    template<typename T, typename EM>
    void trmv(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, VectorView<T, true>& x) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x.size() == N && x.stride() == 1);
        if (N == 0) return;
        detail::trmv_impl(uplo, trans, diag, A_expr, const_cast<T*>(x.data()), N);
    };

    //SYmmetric Matrix-Vector product: y = alpha * A * x + beta * y
    template<typename T, typename EM, typename EV>
    void symv(char uplo, T alpha, const MatExpr<EM>& A_expr, const VecExpr<EV>& x_expr, T beta, Vector<T>& y) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x_expr.self().size() == N && y.size() == N);
        if (N == 0) return;
        // Materialise x to avoid repeated expression evaluation per element
        const Vector<T> xv = detail::materialise<T>(x_expr);
        const T* xp = xv.data();
        T* yp = y.data();
        if (beta == T(0)) std::fill(yp, yp + N, T(0));
        else if (beta != T(1)) {
            parallel_for(N, PARALLEL_THRESHOLD_SIMPLE, [yp, beta](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) yp[i] *= beta;
            });
        }
        if (alpha == T(0)) return;
        const bool upper = (uplo == 'U' || uplo == 'u');
        parallel_for(N, PARALLEL_THRESHOLD_COMPUTE,
            [&A, xp, yp, alpha, N, upper](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i) {
                    T sum = static_cast<T>(A(i, i)) * xp[i];  // diagonal
                    if (upper) {
                        // Row i from upper triangle  (j > i)
                        for (size_t j = i + 1; j < N; ++j)
                            sum += static_cast<T>(A(i, j)) * xp[j];
                        // Column i from upper triangle used via symmetry  (j < i)
                        for (size_t j = 0; j < i; ++j)
                            sum += static_cast<T>(A(j, i)) * xp[j];
                    } else {
                        // Row i from lower triangle  (j < i)
                        for (size_t j = 0; j < i; ++j)
                            sum += static_cast<T>(A(i, j)) * xp[j];
                        // Column i from lower triangle via symmetry  (j > i)
                        for (size_t j = i + 1; j < N; ++j)
                            sum += static_cast<T>(A(j, i)) * xp[j];
                    };
                    yp[i] += alpha * sum;
                };
            });
    };

    // HErmitian Matrix-Vector product:  y = alpha * A * x + beta * y, where A is Hermitian Matrix
    template<typename T, typename EM, typename EV>
    void hemv(char uplo, T alpha, const MatExpr<EM>& A_expr, const VecExpr<EV>& x_expr, T beta, Vector<T>& y) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x_expr.self().size() == N && y.size() == N);
        if (N == 0) return;
        const Vector<T> xv = detail::materialise<T>(x_expr);
        const T* xp = xv.data();
        T* yp = y.data();
        if (beta == T(0)) std::fill(yp, yp + N, T(0));
        else if (beta != T(1)) {
            parallel_for(N, PARALLEL_THRESHOLD_SIMPLE, [yp, beta](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) yp[i] *= beta;
            });
        }
        if (alpha == T(0)) return;
        const bool upper = (uplo == 'U' || uplo == 'u');
        parallel_for(N, PARALLEL_THRESHOLD_COMPUTE,
            [&A, xp, yp, alpha, N, upper](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i) {
                    // Diagonal must be real; the cast drops any spurious imaginary part
                    T sum = static_cast<T>(std::real(static_cast<T>(A(i, i)))) * xp[i];
                    if (upper) {
                        for (size_t j = i + 1; j < N; ++j)
                            sum += static_cast<T>(A(i, j)) * xp[j];
                        for (size_t j = 0; j < i; ++j)
                            sum += conj(static_cast<T>(A(j, i))) * xp[j]; // conj symmetry
                    } else {
                        for (size_t j = 0; j < i; ++j)
                            sum += static_cast<T>(A(i, j)) * xp[j];
                        for (size_t j = i + 1; j < N; ++j)
                            sum += conj(static_cast<T>(A(j, i))) * xp[j]; // conj symmetry
                    };
                    yp[i] += alpha * sum;
                };
            });
    };
};