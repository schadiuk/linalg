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
        LINALG_INLINE std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<Vector<T>>& e) {
            return VecInfo<T>(e.self().data(), 1);
        };

        template<typename T>
        LINALG_INLINE std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<VectorView<T, false>>& e) {
            return VecInfo<T>(e.self().data(), e.self().stride());
        };

        template<typename T>
        LINALG_INLINE std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<VectorView<T, true>>& e) {
            return VecInfo<T>(e.self().data(), e.self().stride());
        };

        template<typename T, typename E>
        LINALG_INLINE std::optional<VecInfo<T>> raw_vec_ptr(const VecExpr<E>&) {
            return std::nullopt;
        };

        // Structure yielding a layout-aware raw-pointer descriptor for a matrix block
        template<typename T>
        struct MatInfo { const T* data; size_t lda; Layout layout; };
 
        // Fallback: any generic expression must be materialised
        template<typename T, typename E>
        LINALG_INLINE std::optional<MatInfo<T>> raw_mat_info(const MatExpr<E>&) {
            return std::nullopt;
        };
 
        // Bare Matrix<T,L>
        template<typename T, Layout L>
        LINALG_INLINE std::optional<MatInfo<T>> raw_mat_info(const MatExpr<Matrix<T,L>>& e) {
            return MatInfo<T>{ e.self().data(), e.self().stride(), L };
        };
 
        // MatRef<T,L> - thin const-reference wrapper around a Matrix
        template<typename T, Layout L>
        LINALG_INLINE std::optional<MatInfo<T>> raw_mat_info(const MatExpr<MatRef<T,L>>& e) {
            return MatInfo<T>{ e.self().mat.data(), e.self().mat.stride(), L };
        };
 
        // Non-transposed, non-conjugated MatrixView (Mut = true or false).
        // Any such view has a valid strided-BLAS pointer regardless of tightness:
        // stride() is always the correct leading dimension for the BLAS kernels
        template<typename T, Layout L, bool Mut>
        LINALG_INLINE std::optional<MatInfo<T>> raw_mat_info(const MatExpr<MatViewExpr<T, L, false, false, Mut>>& e) {
            const auto& v = e.self().view;
            return MatInfo<T>{ v.data(), v.stride(), L };
        };

        // Internal materialisation helper
        template<typename T, typename EV>
        LINALG_INLINE Vector<T> materialise(const VecExpr<EV>& v) {
            const auto& vv = v.self();
            const size_t n = vv.size();
            Vector<T> out(n);
            T* LINALG_RESTRICT op = detail::assume_aligned<64>(out.data());
            parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&vv, op](size_t s, size_t e) {
                LINALG_VECTORIZE
                for (size_t i = s; i < e; ++i) op[i] = static_cast<T>(vv(i));
            });
            return out;
        };

        // Pointer resolution: use raw pointer if available, otherwise materialise. Returns {ptr, stride}.  When stride==1 the pointer is contiguous.
        template<typename T, typename EV>
        LINALG_INLINE std::pair<const T*, size_t> resolve_vec(const VecExpr<EV>& expr, Vector<T>& tmp) {
            auto info = raw_vec_ptr<T>(expr);
            if (info) return { info->data, info->stride };
            tmp = materialise<T>(expr);
            return { tmp.data(), size_t(1) };
        };

        // Namespace containing optimised kernels
        namespace kernels {
            // Fused pointer-level gemv kernel for RowMajor layout
            template<typename T>
            LINALG_INLINE void gemv_kernel_row(T alpha, const T* LINALG_RESTRICT A, size_t lda, const T* x, size_t incx, T beta, T* LINALG_RESTRICT y, size_t M, size_t N) {
                const T* Ap = detail::assume_aligned<64>(A);
                parallel_for(M, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i) {
                        const T* LINALG_RESTRICT a_row = A + i * lda;
                        T acc = T(0);
                        if (incx == 1) { // Unit-stride: vectorisable dot product
                        LINALG_VECTORIZE
                        for (size_t j = 0; j < N; ++j) acc += a_row[j] * x[j];
                    } else {
                        for (size_t j = 0; j < N; ++j) acc += a_row[j] * x[j * incx];
                    };
                    y[i] = (beta == T(0)) ? alpha * acc : alpha * acc + beta * y[i];
                };
                });
            };

            // ColMajor kernel
            template<typename T>
            LINALG_INLINE void gemv_kernel_col(T alpha, const T* LINALG_RESTRICT A, size_t lda, const T* x, size_t incx, T beta, T* LINALG_RESTRICT y, size_t M, size_t N) {
                if (beta == T(0)) std::fill(y, y + M, T(0)); // NaN does not affect zeroing
                else if (beta != T(1)) {
                    LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) y[i] *= beta;
                };
 
                if (alpha == T(0) || N == 0) return;
                auto& pool = ThreadPool::instance();
                const size_t num_threads = std::min(pool.thread_count(), (N + PARALLEL_THRESHOLD_COMPUTE - 1) / PARALLEL_THRESHOLD_COMPUTE);
                if (num_threads <= 1) {
                    // Serial path: straightforward column accumulation, stride-1 inner loop
                    for (size_t j = 0; j < N; ++j) {
                        const T xj = alpha * x[j * incx];
                        if (xj == T(0)) continue;
                        const T* LINALG_RESTRICT a_col = A + j * lda;
                        LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) y[i] += a_col[i] * xj;
                    };
                    return;
                };
                // Parallel path: each thread accumulates a subset of columns into a private vector, then the private results are summed into y
                using AlignedVec = std::vector<T, AlignedAllocator<T>>;
                std::vector<AlignedVec> locals(num_threads, AlignedVec(M, T(0)));
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &locals]() {
                        T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                        for (size_t j = t; j < N; j += num_threads) {
                            const T xj = alpha * x[j * incx];
                            if (xj == T(0)) continue;
                            const T* LINALG_RESTRICT a_col = A + j * lda;
                            LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) loc[i] += a_col[i] * xj;
                        };
                    }));
                };
                for (auto& f : futures) f.get();
                // Reduce thread-local sums into y
                for (size_t t = 0; t < num_threads; ++t) {
                    const T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                    LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) y[i] += loc[i];
                };
            };

            // Shared SAXPY-rank-1 kernel used by both ger and gerc.
            template<typename T, Layout L, bool Conj = false>
            LINALG_INLINE void ger_kernel(T alpha, const T* LINALG_RESTRICT xp, const T* LINALG_RESTRICT yp, T* A_ptr, size_t lda, size_t M, size_t N) {
                if (M == 0 || N == 0) return;
                const T* xa = detail::assume_aligned<64>(xp);
                const T* ya = detail::assume_aligned<64>(yp);
                if constexpr (L == Layout::RowMajor) {
                    // Parallel over rows
                    parallel_for(M, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i) {
                            const T xi = alpha * xa[i];
                            T* LINALG_RESTRICT a_row = A_ptr + i * lda;
                            if constexpr (Conj) {
                                LINALG_VECTORIZE
                                for (size_t j = 0; j < N; ++j) a_row[j] += xi * conj(ya[j]);
                            } else {
                                LINALG_VECTORIZE
                                for (size_t j = 0; j < N; ++j) a_row[j] += xi * ya[j];
                            };
                        };
                    });
                } else {
                    // Parallel over columns
                    parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t js, size_t je) {
                        for (size_t j = js; j < je; ++j) {
                            const T yj = alpha * (Conj ? conj(ya[j]) : ya[j]);
                            if (yj == T(0)) continue;
                            T* LINALG_RESTRICT a_col = A_ptr + j * lda;
                            LINALG_VECTORIZE for (size_t i = 0; i < M; ++i) a_col[i] += xa[i] * yj;
                        };
                    });
                };
            };
        };

        // Dispatch: extracts raw pointers whenever possible, else materialises
        template<typename T, Layout L, typename EM, typename EV>
        LINALG_INLINE void gemv_impl(T alpha, const MatExpr<EM>& a_expr, const VecExpr<EV>& x_expr, T beta, T* y_ptr, size_t M, size_t N) {
            Vector<T> x_tmp;
            auto [x_ptr, incx] = resolve_vec<T>(x_expr, x_tmp);
            auto a_info = raw_mat_info<T>(a_expr);
            if (a_info) {
                if (a_info->layout == Layout::RowMajor)
                    kernels::gemv_kernel_row(alpha, a_info->data, a_info->lda, x_ptr, incx, beta, y_ptr, M, N);
                else
                    kernels::gemv_kernel_col(alpha, a_info->data, a_info->lda, x_ptr, incx, beta, y_ptr, M, N);
            } else {
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
    LINALG_INLINE void gemv(T alpha, const MatExpr<EM>& A, const VecExpr<EV>& x, T beta, Vector<T>& y) {
        const size_t M = A.self().rows();
        const size_t N = A.self().cols();
        BOUNDS_CHECK(M == y.size() && N == x.self().size());
        if (M == 0 || N == 0) return;
        detail::gemv_impl<T, L>(alpha, A, x, beta, y.data(), M, N);
    };
    
    // Public interface output: VectorView<T,true>
    template<typename T, Layout L = Layout::RowMajor, typename EM, typename EV>
    LINALG_INLINE void gemv(T alpha, const MatExpr<EM>& A, const VecExpr<EV>& x, T beta,  VectorView<T, true>& y) {
        const size_t M = A.self().rows();
        const size_t N = A.self().cols();
        BOUNDS_CHECK(M == y.size() && N == x.self().size() && y.stride() == 1);
        if (M == 0 || N == 0) return;
        // VectorView with unit stride: data() gives a contiguous T*
        detail::gemv_impl<T, L>(alpha, A, x, beta, const_cast<T*>(y.data()), M, N);
    };

    namespace detail {
        template<typename T, Layout L, typename EX, typename EY>
        LINALG_INLINE void ger_impl(T alpha, const VecExpr<EX>& x_expr, const VecExpr<EY>& y_expr, T* A_ptr, size_t lda, size_t M, size_t N) {
            Vector<T> xtmp, ytmp;
            auto [xp, incx] = resolve_vec<T>(x_expr, xtmp);
            auto [yp, incy] = resolve_vec<T>(y_expr, ytmp);
            // If strides are non-unit, materialise to unit-stride buffers so the inner SAXPY loop can use LINALG_VECTORIZE without gather penalties.
            if (incx != 1) { xtmp = materialise<T>(x_expr); xp = xtmp.data(); }
            if (incy != 1) { ytmp = materialise<T>(y_expr); yp = ytmp.data(); }
            kernels::ger_kernel<T, L, false>(alpha, xp, yp, A_ptr, lda, M, N);
        };

        template<typename T, Layout L, typename EX, typename EY>
        LINALG_INLINE void gerc_impl(T alpha, const VecExpr<EX>& x_expr, const VecExpr<EY>& y_expr, T* A_ptr, size_t lda, size_t M, size_t N) {
            Vector<T> xtmp, ytmp;
            auto [xp, incx] = resolve_vec<T>(x_expr, xtmp);
            auto [yp, incy] = resolve_vec<T>(y_expr, ytmp);
            if (incx != 1) { xtmp = materialise<T>(x_expr); xp = xtmp.data(); };
            if (incy != 1) { ytmp = materialise<T>(y_expr); yp = ytmp.data(); };
            kernels::ger_kernel<T, L, true>(alpha, xp, yp, A_ptr, lda, M, N);
        };
    };

    // General Rank-1 update: A = A + alpha * x * y^T (vector treated as a column)
    template<typename T, Layout L, typename EX, typename EY>
    LINALG_INLINE void ger(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y, Matrix<T, L>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::ger_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };
    
    template<typename T, Layout L, typename EX, typename EY>
    LINALG_INLINE void ger(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y,
            MatrixView<T, L, false, false, true>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::ger_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };

    // General Rank-1 Conjugated update: A = A + alpha * x * conj(y)^T (in-place conjugation)
    template<typename T, Layout L, typename EX, typename EY>
    LINALG_INLINE void gerc(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y, Matrix<T, L>& A) {
        const size_t M = A.rows(), N = A.cols();
        BOUNDS_CHECK(x.self().size() == M && y.self().size() == N);
        if (M == 0 || N == 0) return;
        detail::gerc_impl<T, L>(alpha, x, y, A.data(), A.stride(), M, N);
    };
    
    template<typename T, Layout L, typename EX, typename EY>
    LINALG_INLINE void gerc(T alpha, const VecExpr<EX>& x, const VecExpr<EY>& y,
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
                    LINALG_VECTORIZE
                    for (size_t j = i + 1; j < N; ++j)
                        sum -= static_cast<T>(A(i, j)) * x[j];
                    x[i] = unit ? sum : sum / static_cast<T>(A(i, i));
                };
            } else {
                for (size_t i = 0; i < N; ++i) {
                    T sum = x[i];
                    LINALG_VECTORIZE
                    for (size_t j = 0; j < i; ++j)
                        sum -= static_cast<T>(A(i, j)) * x[j];
                    x[i] = unit ? sum : sum / static_cast<T>(A(i, i));
                };
            };
        } else {
            if (upper) {
                // Forward pass over columns of A^T (= rows of A scanned upward)
                for (size_t i = 0; i < N; ++i) {
                    T sum = x[i];
                    for (size_t j = 0; j < i; ++j) {
                        T aji = do_conj ? conj(static_cast<T>(A(j, i))) : static_cast<T>(A(j, i));
                        sum -= aji * x[j];
                    };
                    x[i] = unit ? sum : sum / (do_conj ? conj(static_cast<T>(A(i, i))) : static_cast<T>(A(i, i)));
                };
            } else {
                for (size_t ii = 0; ii < N; ++ii) {
                    const size_t i = N - 1 - ii;
                    T sum = x[i];
                    for (size_t k = i + 1; k < N; ++k) {
                        T aki = do_conj ? conj(static_cast<T>(A(k, i))) : static_cast<T>(A(k, i));
                        sum -= aki * x[k];
                    };
                    x[i] = unit ? sum : sum / (do_conj ? conj(static_cast<T>(A(i, i))) : static_cast<T>(A(i, i)));
                };
            };
        };
    };

    namespace detail {
        namespace kernels::trmv {
            // notrans, upper: tp[i] = SUM{j>=i} A[i,j] * x[j]
            template<typename T, Layout L, bool Unit>
            LINALG_INLINE void trmv_notrans_upper(const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T* LINALG_RESTRICT tp, size_t N) {
                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i) {
                        T sum;
                        if constexpr (L == Layout::RowMajor) {
                            const T* LINALG_RESTRICT row = Ap + i * lda;
                            sum = Unit ? xp[i] : row[i] * xp[i];
                            LINALG_VECTORIZE
                            for (size_t j = i + 1; j < N; ++j) sum += row[j] * xp[j];
                        } else {
                            // ColMajor: A[i,j] = Ap[j*lda + i]
                            sum = Unit ? xp[i] : Ap[i * lda + i] * xp[i];
                            // j > i: each column j is non-contiguous in i but x[j] is scalar
                            for (size_t j = i + 1; j < N; ++j) sum += Ap[j * lda + i] * xp[j];
                        };
                        tp[i] = sum;
                    };
                });
            };
 
            // notrans, lower: tp[i] = SUM{j<=i} A[i,j] * x[j]
            template<typename T, Layout L, bool Unit>
            LINALG_INLINE void trmv_notrans_lower(const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T* LINALG_RESTRICT tp, size_t N) {
                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i) {
                        T sum = T(0);
                        if constexpr (L == Layout::RowMajor) {
                            const T* LINALG_RESTRICT row = Ap + i * lda;
                            LINALG_VECTORIZE
                            for (size_t j = 0; j < i; ++j) sum += row[j] * xp[j];
                            sum += Unit ? xp[i] : row[i] * xp[i];
                        } else {
                            for (size_t j = 0; j < i; ++j) sum += Ap[j * lda + i] * xp[j];
                            sum += Unit ? xp[i] : Ap[i * lda + i] * xp[i];
                        };
                        tp[i] = sum;
                    };
                });
            };
 
            // trans (or conj-trans), upper: tp[i] += conj?(A[i,j]) * x[j], j <= i
            // Scatter formulation: for each j, update tp[k] for k <= j (upper column j).
            // Uses thread-local accumulators (same pattern as gemv_kernel_col) to avoid data races on tp without atomics.
            template<typename T, Layout L, bool Unit, bool DoConj>
            LINALG_INLINE void trmv_trans_upper(const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T* LINALG_RESTRICT tp, size_t N) {
                auto& pool = ThreadPool::instance();
                const size_t num_threads = std::min(pool.thread_count(),
                    (N + PARALLEL_THRESHOLD_COMPUTE - 1) / PARALLEL_THRESHOLD_COMPUTE);
                // Aval: load A[r,c] and optionally conjugate.
                auto Aval = [=](size_t r, size_t c) -> T {
                    T v;
                    if constexpr (L == Layout::RowMajor) v = Ap[r * lda + c];
                    else v = Ap[c * lda + r];
                    if constexpr (DoConj) return conj(v);
                    else return v;
                };
 
                if (num_threads <= 1) {
                    // Serial scatter: for each column j of A (= row j of A^T), update tp[0...j] with A[k,j]*x[j] for k <= j.
                    for (size_t j = 0; j < N; ++j) {
                        const T xj = xp[j];
                        tp[j] += (Unit ? T(1) : Aval(j, j)) * xj;
                        for (size_t k = j + 1; k < N; ++k) tp[k] += Aval(j, k) * xj;
                    };
                    return;
                };
                using AlignedVec = std::vector<T, AlignedAllocator<T>>;
                std::vector<AlignedVec> locals(num_threads, AlignedVec(N, T(0)));
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &locals, &Aval]() {
                        T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                        for (size_t j = t; j < N; j += num_threads) {
                            const T xj = xp[j];
                            loc[j] += (Unit ? T(1) : Aval(j, j)) * xj;
                            for (size_t k = j + 1; k < N; ++k) loc[k] += Aval(j, k) * xj;
                        };
                    }));
                };
                for (auto& f : futures) f.get();
                for (size_t t = 0; t < num_threads; ++t) {
                    const T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                    LINALG_VECTORIZE for (size_t i = 0; i < N; ++i) tp[i] += loc[i];
                };
            };
 
            // trans (or conj-trans), lower: scatter over rows of A^T that are columns
            // of lower-triangular A, i.e. for each j update tp[j..N-1].
            template<typename T, Layout L, bool Unit, bool DoConj>
            LINALG_INLINE void trmv_trans_lower(const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T* LINALG_RESTRICT tp, size_t N) {
                auto& pool = ThreadPool::instance();
                const size_t num_threads = std::min(pool.thread_count(),
                    (N + PARALLEL_THRESHOLD_COMPUTE - 1) / PARALLEL_THRESHOLD_COMPUTE);
 
                auto Aval = [=](size_t r, size_t c) -> T {
                    T v;
                    if constexpr (L == Layout::RowMajor) v = Ap[r * lda + c];
                    else v = Ap[c * lda + r];
                    if constexpr (DoConj) return conj(v);
                    else return v;
                };
 
                if (num_threads <= 1) {
                    for (size_t j = 0; j < N; ++j) {
                        const T xj = xp[j];
                        tp[j] += (Unit ? T(1) : Aval(j, j)) * xj;
                        for (size_t k = 0; k < j; ++k) tp[k] += Aval(j, k) * xj;
                    };
                    return;
                };
                using AlignedVec = std::vector<T, AlignedAllocator<T>>;
                std::vector<AlignedVec> locals(num_threads, AlignedVec(N, T(0)));
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                for (size_t t = 0; t < num_threads; ++t) {
                    futures.push_back(pool.enqueue([=, &locals, &Aval]() {
                        T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                        for (size_t j = t; j < N; j += num_threads) {
                            const T xj = xp[j];
                            loc[j] += (Unit ? T(1) : Aval(j, j)) * xj;
                            for (size_t k = 0; k < j; ++k) loc[k] += Aval(j, k) * xj;
                        };
                    }));
                };
                for (auto& f : futures) f.get();
                for (size_t t = 0; t < num_threads; ++t) {
                    const T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
                    LINALG_VECTORIZE for (size_t i = 0; i < N; ++i) tp[i] += loc[i];
                };
            };
        };

        template<typename T, typename EM>
        LINALG_INLINE void trmv_impl(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, T* xp, size_t N) {
            const auto& A  = A_expr.self();
            const bool upper = (uplo == 'U' || uplo == 'u');
            const bool unit = (diag == 'U' || diag == 'u');
            const bool do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
            const bool do_conj = (trans == 'C' || trans == 'c');

            auto a_info = raw_mat_info<T>(A_expr);
            Matrix<T, Layout::RowMajor> A_tmp;
            const T* Ap;
            size_t lda;
            Layout layout;
            if (a_info) {
                Ap = a_info->data; lda = a_info->lda; layout = a_info->layout;
            } else {
                A_tmp = Matrix<T, Layout::RowMajor>(A_expr);
                Ap = A_tmp.data(); lda = A_tmp.stride(); layout = Layout::RowMajor;
            };
 
            // Zero-initialised output buffer: breaks aliasing.
            Vector<T> tmp(N, T(0));
            T* LINALG_RESTRICT tp = detail::assume_aligned<64>(tmp.data());
            const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xp);

            // Dispatch to the eight specialised kernel. The if-constexpr inside each kernel eliminates the layout branch at instantiation time.
            auto dispatch_notrans = [&](auto Unit_tag) {
                constexpr bool U = Unit_tag.value;
                if (layout == Layout::RowMajor) {
                    if (upper) kernels::trmv::trmv_notrans_upper<T, Layout::RowMajor, U>(Ap, lda, xa, tp, N);
                    else kernels::trmv::trmv_notrans_lower<T, Layout::RowMajor, U>(Ap, lda, xa, tp, N);
                } else {
                    if (upper) kernels::trmv::trmv_notrans_upper<T, Layout::ColMajor, U>(Ap, lda, xa, tp, N);
                    else kernels::trmv::trmv_notrans_lower<T, Layout::ColMajor, U>(Ap, lda, xa, tp, N);
                };
            };
 
            auto dispatch_trans = [&](auto Unit_tag, auto Conj_tag) {
                constexpr bool U = Unit_tag.value;
                constexpr bool C = Conj_tag.value;
                if (layout == Layout::RowMajor) {
                    if (upper) kernels::trmv::trmv_trans_upper<T, Layout::RowMajor, U, C>(Ap, lda, xa, tp, N);
                    else kernels::trmv::trmv_trans_lower<T, Layout::RowMajor, U, C>(Ap, lda, xa, tp, N);
                } else {
                    if (upper) kernels::trmv::trmv_trans_upper<T, Layout::ColMajor, U, C>(Ap, lda, xa, tp, N);
                    else kernels::trmv::trmv_trans_lower<T, Layout::ColMajor, U, C>(Ap, lda, xa, tp, N);
                };
            };
 
            // Encode runtime bools as std::integral_constant so template args are resolved at compile time inside each kernel.
            if (!do_trans) {
                if (unit) dispatch_notrans(std::true_type{});
                else      dispatch_notrans(std::false_type{});
            } else {
                if (unit && do_conj) dispatch_trans(std::true_type{},  std::true_type{});
                if (unit && !do_conj) dispatch_trans(std::true_type{},  std::false_type{});
                if (!unit && do_conj) dispatch_trans(std::false_type{}, std::true_type{});
                if (!unit && !do_conj) dispatch_trans(std::false_type{}, std::false_type{});
            };
 
            std::copy(tp, tp + N, xp);
        };

        namespace kernels {
            template<typename T, Layout L, bool DoConj>
            LINALG_INLINE void symv_hemv_kernel(T alpha, const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T beta, T* LINALG_RESTRICT yp, size_t N, bool upper) {
                if (beta == T(0)) std::fill(yp, yp + N, T(0));
                else if (beta != T(1)) {
                    LINALG_VECTORIZE for (size_t i = 0; i < N; ++i) yp[i] *= beta;
                };
                if (alpha == T(0)) return;
                // Inline element accessor: loads A[r,c] respecting layout.
                auto Aload = [=](size_t r, size_t c) -> T {
                    if constexpr (L == Layout::RowMajor) return Ap[r * lda + c];
                    else return Ap[c * lda + r];
                };
                // Two-vector sweep: accumulates both the direct (row) and symmetric (column) contributions in a single pass over the stored triangle, touching each stored element exactly once.
                if (N < PARALLEL_THRESHOLD_COMPUTE) {
                    if (upper) {
                        for (size_t i = 0; i < N; ++i) {
                            // Diagonal (real part for Hermitian)
                            const T aii = DoConj ? static_cast<T>(std::real(Aload(i, i))) : Aload(i, i);
                            T sum_i = aii * xp[i];
                            for (size_t j = i + 1; j < N; ++j) {
                                const T aij = Aload(i, j);
                                sum_i += aij * xp[j];
                                // Symmetric contribution: A[j,i] = conj(A[i,j]) for Hermitian
                                if constexpr (DoConj) yp[j] += alpha * conj(aij) * xp[i];
                                else yp[j] += alpha * aij * xp[i];
                            };
                            yp[i] += alpha * sum_i;
                        };
                    } else {
                        // Lower storage
                        for (size_t i = 0; i < N; ++i) {
                            const T aii = DoConj ? static_cast<T>(std::real(Aload(i, i))) : Aload(i, i);
                            T sum_i = aii * xp[i];
                            for (size_t j = 0; j < i; ++j) {
                                const T aij = Aload(i, j);
                                sum_i += aij * xp[j];
                                if constexpr (DoConj) yp[j] += alpha * conj(aij) * xp[i];
                                else yp[j] += alpha * aij * xp[i];
                            };
                            yp[i] += alpha * sum_i;
                        };
                    };
                    return;
                };
                // Parallel path
                auto& pool = ThreadPool::instance();
                const size_t num_threads = std::min(pool.thread_count(), (N + PARALLEL_THRESHOLD_COMPUTE - 1) / PARALLEL_THRESHOLD_COMPUTE);
                // row_acc[i] = direct dot-product contribution for row i
                // scatter[t][j] = symmetric contributions accumulated by thread t
                using AlignedVec = std::vector<T, AlignedAllocator<T>>;
                std::vector<AlignedVec> row_acc(num_threads, AlignedVec(N, T(0)));
                std::vector<AlignedVec> scatter(num_threads, AlignedVec(N, T(0)));
                std::vector<std::future<void>> futures;
                futures.reserve(num_threads);
                const size_t chunk = N / num_threads;
                const size_t remainder = N % num_threads;
                size_t offset = 0;
                for (size_t t = 0; t < num_threads; ++t) {
                    const size_t cnt = chunk + (t < remainder ? 1 : 0);
                    const size_t t_rs = offset;
                    const size_t t_re = offset + cnt;
                    offset += cnt;
                    futures.push_back(pool.enqueue([=, &row_acc, &scatter, &Aload]() {
                        T* LINALG_RESTRICT racc = detail::assume_aligned<64>(row_acc[t].data());
                        T* LINALG_RESTRICT scat = detail::assume_aligned<64>(scatter[t].data());
                        if (upper) {
                            for (size_t i = t_rs; i < t_re; ++i) {
                                const T aii = DoConj ? static_cast<T>(std::real(Aload(i, i))) : Aload(i, i);
                                T sum_i = aii * xp[i];
                                for (size_t j = i + 1; j < N; ++j) {
                                    const T aij = Aload(i, j);
                                    sum_i += aij * xp[j];
                                    if constexpr (DoConj) scat[j] += conj(aij) * xp[i];
                                    else scat[j] += aij * xp[i];
                                };
                                racc[i] = sum_i;
                            };
                        } else {
                            for (size_t i = t_rs; i < t_re; ++i) {
                                const T aii = DoConj ? static_cast<T>(std::real(Aload(i, i))) : Aload(i, i);
                                T sum_i = aii * xp[i];
                                for (size_t j = 0; j < i; ++j) {
                                    const T aij = Aload(i, j);
                                    sum_i += aij * xp[j];
                                    if constexpr (DoConj) scat[j] += conj(aij) * xp[i];
                                    else scat[j] += aij * xp[i];
                                };
                                racc[i] = sum_i;
                            };
                        };
                    }));
                };
                for (auto& f : futures) f.get();

                parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i) {
                        T acc = row_acc[0][i];
                        LINALG_VECTORIZE
                        for (size_t t = 1; t < num_threads; ++t) acc += row_acc[t][i];
                        T sct = scatter[0][i];
                        LINALG_VECTORIZE
                        for (size_t t = 1; t < num_threads; ++t) sct += scatter[t][i];
                        yp[i] += alpha * (acc + sct);
                    };
                });
            };
        };
    };

    // TRiangular Matrix-Vector product: x = op(A) * x
    template<typename T, typename EM>
    LINALG_INLINE void trmv(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, Vector<T>& x) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x.size() == N);
        if (N == 0) return;
        detail::trmv_impl(uplo, trans, diag, A_expr, x.data(), N);
    };
 
    template<typename T, typename EM>
    LINALG_INLINE void trmv(char uplo, char trans, char diag, const MatExpr<EM>& A_expr, VectorView<T, true>& x) {
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

        auto a_info = detail::raw_mat_info<T>(A_expr);
        Matrix<T, Layout::RowMajor> A_tmp;
        const T* Ap; size_t lda; Layout layout;
        if (a_info) {
            Ap = a_info->data; lda = a_info->lda; layout = a_info->layout;
        } else {
            A_tmp = Matrix<T, Layout::RowMajor>(A_expr);
            Ap = A_tmp.data(); lda = A_tmp.stride(); layout = Layout::RowMajor;
        };
        Vector<T> xtmp;
        auto [xp_raw, incx] = detail::resolve_vec<T>(x_expr, xtmp);
        // Force unit-stride x for the kernel
        if (incx != 1) { xtmp = detail::materialise<T>(x_expr); xp_raw = xtmp.data(); }
        const T* LINALG_RESTRICT xp = detail::assume_aligned<64>(xp_raw);
        T* LINALG_RESTRICT yp = detail::assume_aligned<64>(y.data());

        const bool upper = (uplo == 'U' || uplo == 'u');
        if (layout == Layout::RowMajor)
            detail::kernels::symv_hemv_kernel<T, Layout::RowMajor, false>(alpha, Ap, lda, xp, beta, yp, N, upper);
        else
            detail::kernels::symv_hemv_kernel<T, Layout::ColMajor, false>(alpha, Ap, lda, xp, beta, yp, N, upper);
    };

    // HErmitian Matrix-Vector product:  y = alpha * A * x + beta * y, where A is Hermitian Matrix
    template<typename T, typename EM, typename EV>
    void hemv(char uplo, T alpha, const MatExpr<EM>& A_expr, const VecExpr<EV>& x_expr, T beta, Vector<T>& y) {
        const auto& A = A_expr.self();
        const size_t N = A.rows();
        BOUNDS_CHECK(A.cols() == N && x_expr.self().size() == N && y.size() == N);
        if (N == 0) return;

        auto a_info = detail::raw_mat_info<T>(A_expr);
        Matrix<T, Layout::RowMajor> A_tmp;
        const T* Ap; size_t lda; Layout layout;
        if (a_info) {
            Ap = a_info->data; lda = a_info->lda; layout = a_info->layout;
        } else {
            A_tmp = Matrix<T, Layout::RowMajor>(A_expr);
            Ap = A_tmp.data(); lda = A_tmp.stride(); layout = Layout::RowMajor;
        };
 
        Vector<T> xtmp;
        auto [xp_raw, incx] = detail::resolve_vec<T>(x_expr, xtmp);
        if (incx != 1) { xtmp = detail::materialise<T>(x_expr); xp_raw = xtmp.data(); }
        const T* LINALG_RESTRICT xp = detail::assume_aligned<64>(xp_raw);
        T* LINALG_RESTRICT yp = detail::assume_aligned<64>(y.data());
 
        const bool upper = (uplo == 'U' || uplo == 'u');
        if (layout == Layout::RowMajor)
            detail::kernels::symv_hemv_kernel<T, Layout::RowMajor, true>(alpha, Ap, lda, xp, beta, yp, N, upper);
        else
            detail::kernels::symv_hemv_kernel<T, Layout::ColMajor, true>(alpha, Ap, lda, xp, beta, yp, N, upper);
    };
};