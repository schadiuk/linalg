#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>
#include <linalg/core/hints.hpp>
#include <linalg/storage/vector.hpp>

namespace linalg {
    // Forward declarations
    template<typename U> struct MatExpr;
    template<typename E1, typename E2> struct GemmExpr;
    template<typename E1, typename E2> struct MatAddExpr;
    template<typename E1, typename E2> struct MatSubExpr;

    /// @brief Main matrix storage class.
    /// @tparam T Scalar element type. Supports: `float`, `double`, and their `std::complex` counterparts.
    /// @tparam L Layout.
    template<typename T = DefaultScalar, Layout L = Layout::RowMajor> requires Scalar<T>
    class Matrix : public MatExpr<Matrix<T, L>> {
    private:
        /// @brief Builds an `n`-slot flat buffer whose `k`-th slot is placement-constructed from `init_fn(k)`, in a single parallel write pass over freshly allocated memory.
        template<typename F>
        static std::vector<T, UninitAlignedAllocator<T>> fill_construct_flat(size_t n, F&& init_fn) {
            std::vector<T, UninitAlignedAllocator<T>> buf(n);
            T* LINALG_RESTRICT p = buf.data();
            parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [p, &init_fn](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) ::new (static_cast<void*>(p + i)) T(init_fn(i));
            });
            return buf;
        };

        template<typename F>
        static std::vector<T, UninitAlignedAllocator<T>> fill_construct(size_t rows, size_t cols, F&& elem_fn) {
            std::vector<T, UninitAlignedAllocator<T>> buf(rows * cols);
            T* LINALG_RESTRICT p = buf.data();
            const size_t stride = (L == Layout::RowMajor) ? cols : rows;
            if constexpr (L == Layout::RowMajor) {
                const size_t threshold = std::max<size_t>(1, PARALLEL_THRESHOLD_SIMPLE / (cols + 1));
                parallel_for(rows, threshold, [p, stride, cols, &elem_fn](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < cols; ++j) ::new (static_cast<void*>(p + i * stride + j)) T(elem_fn(i, j));
                });
            } else {
                const size_t threshold = std::max<size_t>(1, PARALLEL_THRESHOLD_SIMPLE / (rows + 1));
                parallel_for(cols, threshold, [p, stride, rows, &elem_fn](size_t cs, size_t ce) {
                    for (size_t j = cs; j < ce; ++j)
                        for (size_t i = 0; i < rows; ++i) ::new (static_cast<void*>(p + j * stride + i)) T(elem_fn(i, j));
                });
            };
            return buf;
        };
    public:
        /// @brief Empty matrix constructor. 
        /// @param m Row count.
        /// @param n Column count.
        Matrix(size_t m = 0, size_t n = 0) : rows_(m), cols_(n), stride_(L == Layout::RowMajor ? n : m), data_(fill_construct_flat(m * n, [](size_t) { return T(); })) {};

        /// @brief Uniform constructor that fills matrix with a given value.
        /// @param m Row count.
        /// @param n Column count. 
        /// @param val The fill-in value.
        Matrix(size_t m, size_t n, const T& val) : rows_(m), cols_(n), stride_(L == Layout::RowMajor ? n : m), data_(fill_construct_flat(m * n, [&val](size_t) { return val; })) {};

        /// @brief Constructor from a given `MatExpr`.
        /// @tparam E CRTP-required parameter clause of `MatExpr`.
        /// @param expr The expression.
        /// @param val Placeholder value.
        template<typename E>
        Matrix(const MatExpr<E>& expr, const T& val = T(0)) : rows_(expr.rows()), cols_(expr.cols()), stride_(L == Layout::RowMajor ? expr.cols() : expr.rows()),
            data_(fill_construct(expr.rows(), expr.cols(), [&e = expr.self()](size_t i, size_t j) { return static_cast<T>(e(i, j)); })) {};;

        /// @brief Constructor from a given `MatrixView`.
        /// @tparam Trans Transposition flag required by the view.
        /// @tparam Conj Conjugation flag required by the view.
        /// @tparam Mutable Mutability indicator.
        /// @param view The view.
        template<bool Trans, bool Conj, bool Mutable>
        Matrix(const MatrixView<T, L, Trans, Conj, Mutable>& view) : rows_(view.rows()), cols_(view.cols()), stride_(L == Layout::RowMajor ? view.cols() : view.rows()),
            data_(fill_construct(view.rows(), view.cols(), [&view](size_t i, size_t j) { return static_cast<T>(view(i, j)); })) {};

        /// @brief Constructor from a 2D `std::array` object.
        /// @tparam rows Row count deduced from the array.
        /// @tparam cols Column count deduced likewise.
        /// @param arr The array.
        template<size_t rows, size_t cols>
        Matrix(const std::array<std::array<T, cols>, rows>& arr) : rows_(rows), cols_(cols), stride_(L == Layout::RowMajor ? cols : rows), data_(rows* cols) {
             constexpr size_t total = rows * cols;
            if constexpr (total == 0) return;
            else {
                T* LINALG_RESTRICT p = data_.data();
                const size_t stride = stride_;
                if constexpr (total < PARALLEL_THRESHOLD_SIMPLE) {
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            ::new (static_cast<void*>(p + (L == Layout::RowMajor ? i * stride + j : j * stride + i))) T(arr[i][j]);
                        };
                    };
                }
                else if constexpr (L == Layout::RowMajor) {
                    parallel_for(rows, 1, [p, stride, &arr](size_t start_row, size_t end_row) {
                            for (size_t i = start_row; i < end_row; ++i) {
                                const auto& src_row = arr[i];
                                for (size_t j = 0; j < cols; ++j) {
                                    ::new (static_cast<void*>(p + i * stride + j)) T(src_row[j]);
                                };
                            };
                        });
                }
                else {
                    parallel_for(cols, 1, [p, stride, &arr](size_t start_col, size_t end_col) {
                            for (size_t j = start_col; j < end_col; ++j) {
                                for (size_t i = 0; i < rows; ++i) {
                                    ::new (static_cast<void*>(p + j * stride + i)) T(arr[i][j]);
                                };
                            };
                        });
                };
            };
        };

        /// @brief Constructor from a 2D C-styled array.
        /// @tparam rows Row count deduced from the array.
        /// @tparam cols Column count obtained likewise.
        /// @param arr The array. 
        template<size_t rows, size_t cols>
        Matrix(const T(&arr)[rows][cols]) : rows_(rows), cols_(cols), stride_(L == Layout::RowMajor ? cols : rows), data_(rows* cols) {
             constexpr size_t total = rows * cols;
            if constexpr (total == 0) return;
            else {
                T* LINALG_RESTRICT p = data_.data();
                const size_t stride = stride_;
                if constexpr (total < PARALLEL_THRESHOLD_SIMPLE) {
                    for (size_t i = 0; i < rows; ++i) {
                        for (size_t j = 0; j < cols; ++j) {
                            ::new (static_cast<void*>(p + (L == Layout::RowMajor ? i * stride + j : j * stride + i))) T(arr[i][j]);
                        };
                    };
                }
                else if constexpr (L == Layout::RowMajor) {
                    parallel_for(rows, 1, [p, stride, &arr](size_t start_row, size_t end_row) {
                            for (size_t i = start_row; i < end_row; ++i) {
                                for (size_t j = 0; j < cols; ++j) {
                                    ::new (static_cast<void*>(p + i * stride + j)) T(arr[i][j]);
                                };
                            };
                        });
                }
                else {
                    parallel_for(cols, 1, [p, stride, &arr](size_t start_col, size_t end_col) {
                            for (size_t j = start_col; j < end_col; ++j) {
                                for (size_t i = 0; i < rows; ++i) {
                                    ::new (static_cast<void*>(p + j * stride + i)) T(arr[i][j]);
                                };
                            };
                        });
                };
            };
        };

        /// @brief Assignment operator from a C-styled array.
        /// @tparam rows Row count deduced from the array.
        /// @tparam cols Column count deduced likewise.
        /// @param arr The array. 
        /// @return `*this` Matrix<T, L> pointer.
        template<size_t rows, size_t cols>
        Matrix<T, L>& operator=(const T(&arr)[rows][cols]) {
            BOUNDS_CHECK(this->rows_ == rows && this->cols_ == cols);

            const size_t total = rows * cols;
            if (total == 0) return *this;

            if constexpr (total < PARALLEL_THRESHOLD_SIMPLE) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        (*this)(i, j) = arr[i][j];
                    };
                };
            }
            else if constexpr (L == Layout::RowMajor) {
                parallel_for(rows, 1, [this, &arr](size_t start_row, size_t end_row) {
                        for (size_t i = start_row; i < end_row; ++i) {
                            for (size_t j = 0; j < cols; ++j) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
            }
            else {
                parallel_for(cols, 1, [this, &arr](size_t start_col, size_t end_col) {
                        for (size_t j = start_col; j < end_col; ++j) {
                            for (size_t i = 0; i < rows; ++i) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
            };
            return *this;
        };

        /// @brief Assignment operator from an expression.
        /// @tparam E CRTP-required parameter clause of `MatExpr`.
        /// @param expr The expression.
        /// @return `*this` Matrix<T, L> pointer.
        template<typename E>
        Matrix<T, L>& operator=(const MatExpr<E>& expr) {
            const auto& e = expr.self();
            BOUNDS_CHECK(this->rows_ == e.rows() && this->cols_ == e.cols());
            const void* data_ptr = data_.data();
            const size_t data_bytes = data_.size() * sizeof(T);

            constexpr bool elementwise = detail::expr_is_elementwise_v<E>;
            bool depends = !elementwise && e.depends_on(data_ptr, data_bytes);
            const size_t total = this->rows_ * this->cols_;

            if (total < PARALLEL_THRESHOLD_SIMPLE || depends) {
                if (depends) {
                    data_ = fill_construct(this->rows_, this->cols_, [&e](size_t i, size_t j) { return static_cast<T>(e(i, j)); });
                }
                else {
                    for (size_t i = 0; i < this->rows_; ++i) {
                        for (size_t j = 0; j < this->cols_; ++j) {
                            (*this)(i, j) = e(i, j);
                        };
                    };
                };
            }
            else {
                if constexpr (L == Layout::RowMajor) {
                    parallel_for(this->rows_, 1, [this, &e](size_t start_row, size_t end_row) {
                            for (size_t i = start_row; i < end_row; ++i) {
                                for (size_t j = 0; j < this->cols_; ++j) {
                                    (*this)(i, j) = e(i, j);
                                };
                            };
                        });
                }
                else {
                    parallel_for(this->cols_, 1, [this, &e](size_t start_col, size_t end_col) {
                            for (size_t j = start_col; j < end_col; ++j) {
                                for (size_t i = 0; i < this->rows_; ++i) {
                                    (*this)(i, j) = e(i, j);
                                };
                            };
                        });
                };
            };
            return *this;
        };

		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
		bool depends_on(const void* p, size_t bytes) const {			
			const void* start = static_cast<const void*>(data_.data());
			const void* end = static_cast<const void*>(data_.data() + data_.size());
			const void* other_end = static_cast<const char*>(p) + bytes;
			return (p < end) && (other_end > start);
		};

        /// @brief Converion to row-major array.
        /// @return 2D `std::array`.
        template<size_t rows, size_t cols>
        std::array<std::array<T, cols>, rows> to_array() const {
            BOUNDS_CHECK(this->rows_ == rows && this->cols_ == cols);
            std::array<std::array<T, cols>, rows> result;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = (*this)(i, j);
                };
            };
            return result;
        };

        /// @brief Layout-aware reshaping helper.
        /// @param new_rows Desired row count.
        /// @param new_cols Desired column count.
        void reshape(size_t new_rows, size_t new_cols) {
            BOUNDS_CHECK(new_rows * new_cols == rows_ * cols_);
            rows_ = new_rows;
            cols_ = new_cols;
            stride_ = (L == Layout::RowMajor) ? cols_ : rows_;
        };

        size_t rows() const { return rows_; };
        size_t cols() const { return cols_; };
        size_t stride() const { return stride_; };
        T* data() { return data_.data(); };
        const T* data() const { return data_.data(); };

        /// @brief Safe element access, as `operator()` does not check if index is valid.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)`.
        /// @throws `linalg::detail::BoundsError` exception.
        T& at(size_t i, size_t j) {
            BOUNDS_CHECK(i < rows_ && j < cols_);
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };
 
        /// @brief Safe element access, as `operator()` does not check if index is valid.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)`.
        /// @throws `linalg::detail::BoundsError` exception.
        const T& at(size_t i, size_t j) const {
            BOUNDS_CHECK(i < rows_ && j < cols_);
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        /// @brief Unchecked element indexation.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)` if such is legal, undefined otherwise. 
        LINALG_INLINE
        T& operator()(size_t i, size_t j) {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        /// @brief Unchecked element indexation.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)` if such is legal, undefined otherwise.
        LINALG_INLINE
        const T& operator()(size_t i, size_t j) const {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        /// @brief Static factory method for creating `n * n` identity matrix.
        /// @param n Dimension.
        /// @return The identity matrix of size `n`.
        static Matrix identity(size_t n) {
            Matrix mat(n, n, T(0));
            for (size_t i = 0; i < n; ++i) { mat(i, i) = T(1); };
            return mat;
        };

        /// @brief Static factory method for creating `m * n` matrix initialised with all ones.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix ones(size_t m, size_t n) { return Matrix(m, n, T(1)); };

        /// @brief Static factory method for creating `m * n` matrix initialised with all zeros.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix zeros(size_t m, size_t n) { return Matrix(m, n, T(0)); };

        /// @brief Static factory method for creating `m * n` matrix initialised with random entries.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix random(size_t m, size_t n) {
            Matrix mat;
            mat.data_ = fill_construct_flat(m * n, [](size_t) { return randomScalar<T>(); });
            mat.rows_ = m; mat.cols_ = n; mat.stride_ = (L == Layout::RowMajor) ? n : m;
            return mat;
        };

        /// @brief Sub-row extraction.
        /// @param i Row index.
        /// @param j0 Starting column index.
        /// @param count Number of elements to extract.
        /// @return Sub-row: row `i`, columns `[j0, j0 + count)` as an independent `Vector`.
        Vector<T> row(size_t i, size_t j0, size_t count) const {
            BOUNDS_CHECK(i < rows_ && j0 <= cols_ && count <= cols_ - j0);
            Vector<T> r(count);
            T* LINALG_RESTRICT dst = r.data();
            if constexpr (L == Layout::RowMajor) {
                const T* LINALG_RESTRICT src = data_.data() + i * stride_ + j0;
                std::copy(src, src + count, dst);
            } else {
                const size_t threshold = std::max<size_t>(1, PARALLEL_THRESHOLD_SIMPLE / (rows_ + 1));
                parallel_for(count, threshold, [this, i, j0, dst](size_t s, size_t e) {
                    for (size_t k = s; k < e; ++k) dst[k] = (*this)(i, j0 + k);
                });
            };
            return r;
        };

        /// @brief Full row extraction.
        /// @param i Row index.
        /// @return The extracted row.
        Vector<T> row(size_t i) const { return row(i, 0, cols_); };

        /// @brief Sub-column extraction.
        /// @param j Rolumn index.
        /// @param i0 Row offset.
        /// @param count Number of elements to extract.
        /// @return Sub-column: column `j`, rows `[i0, i0 + count)`, as a new, independent `Vector`.
        Vector<T> col(size_t j, size_t i0, size_t count) const {
            BOUNDS_CHECK(j < cols_ && i0 <= rows_ && count <= rows_ - i0);
            Vector<T> c(count);
            T* LINALG_RESTRICT dst = c.data();
            if constexpr (L == Layout::ColMajor) {
                const T* LINALG_RESTRICT src = data_.data() + j * stride_ + i0;
                std::copy(src, src + count, dst);
            } else {
                const size_t threshold = std::max<size_t>(1, PARALLEL_THRESHOLD_SIMPLE / (cols_ + 1));
                parallel_for(count, threshold, [this, j, i0, dst](size_t s, size_t e) {
                    for (size_t k = s; k < e; ++k) dst[k] = (*this)(i0 + k, j);
                });
            };
            return c;
        };

        /// @brief Full column extraction.
        /// @param j Column index.
        /// @return The extracted column.
        Vector<T> col(size_t j) const { return col(j, 0, rows_); };

        template<typename E1, typename E2>
        Matrix(const GemmExpr<E1, E2>& expr);
        
        template<typename E1, typename E2>
        Matrix<T, L>& operator=(const GemmExpr<E1, E2>& expr);

        // GEMM-accumulate dispatch hooks:
        template<typename E, typename Ea, typename Eb>
        Matrix(const MatAddExpr<E, GemmExpr<Ea, Eb>>& expr);
        template<typename E, typename Ea, typename Eb>
        Matrix<T, L>& operator=(const MatAddExpr<E, GemmExpr<Ea, Eb>>& expr);

        template<typename Ea, typename Eb, typename E>
        Matrix(const MatAddExpr<GemmExpr<Ea, Eb>, E>& expr);
        template<typename Ea, typename Eb, typename E>
        Matrix<T, L>& operator=(const MatAddExpr<GemmExpr<Ea, Eb>, E>& expr);

        template<typename E, typename Ea, typename Eb>
        Matrix(const MatSubExpr<E, GemmExpr<Ea, Eb>>& expr);
        template<typename E, typename Ea, typename Eb>
        Matrix<T, L>& operator=(const MatSubExpr<E, GemmExpr<Ea, Eb>>& expr);

        template<typename Ea, typename Eb, typename E>
        Matrix(const MatSubExpr<GemmExpr<Ea, Eb>, E>& expr);
        template<typename Ea, typename Eb, typename E>
        Matrix<T, L>& operator=(const MatSubExpr<GemmExpr<Ea, Eb>, E>& expr);

    private:
        template<typename ESeed, typename Ea, typename Eb>
        Matrix<T, L>& assign_gemm_accumulate(const ESeed& seed, T gemm_alpha, const Ea& a, const Eb& b, T gemm_beta = T(1)) {
            const size_t bytes = data_.size() * sizeof(T);
            const void* dst = static_cast<const void*>(data_.data());
            const bool aliased = seed.depends_on(dst, bytes) || a.depends_on(dst, bytes) || b.depends_on(dst, bytes);
            if (aliased) {
                Matrix<T, L> tmp(seed);
                gemm(gemm_alpha, a, b, gemm_beta, tmp);
                *this = std::move(tmp);
            } else {
                *this = seed;
                gemm(gemm_alpha, a, b, gemm_beta, *this);
            };
            return *this;
        };

        // Data storage and dimensions
        std::vector<T, UninitAlignedAllocator<T>> data_;
        size_t rows_, cols_, stride_;

        template<typename U, Layout LL, bool Trans, bool Conj, bool Mutable> friend class MatrixView;
    };
};