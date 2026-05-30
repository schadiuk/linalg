#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>
#include <linalg/core/hints.hpp>

namespace linalg {
    // Forward declaration of expression template class
    template<typename U> struct MatExpr;

    /// @brief Main matrix storage class.
    /// @tparam T Scalar element type. Supports: float, double, and their std::complex counterparts.
    /// @tparam L Layout.
    template<typename T = DefaultScalar, Layout L = Layout::RowMajor> requires Scalar<T>
    class Matrix : public MatExpr<Matrix<T, L>> {
    public:
        /// @brief Empty matrix constructor. 
        /// @param m Row count.
        /// @param n Column count.
        /// @note Delegates to constructor of `std::vector` of m * n size.
        Matrix(size_t m = 0, size_t n = 0) : rows_(m), cols_(n), stride_(L == Layout::RowMajor ? n : m), data_(m * n) {};

        /// @brief Uniform constructor that fills matrix with a given value.
        /// @param m Row count.
        /// @param n Column count. 
        /// @param val The fill-in value.
        Matrix(size_t m, size_t n, const T& val) : rows_(m), cols_(n), stride_(L == Layout::RowMajor ? n : m), data_() {
            size_t total = m * n;
            data_.resize(total);
            parallel_for(total, PARALLEL_THRESHOLD_SIMPLE,
                [this, val](size_t start, size_t end) {
                    for (size_t i = start; i < end; ++i) {
                        data_[i] = val;
                    };
                });
        };

        /// @brief Constructor from a given `MatExpr`.
        /// @tparam E CRTP-required parameter clause of `MatExpr`.
        /// @param expr The expression.
        /// @param val Placeholder value.
        template<typename E>
        Matrix(const MatExpr<E>& expr, const T& val = T(0)) : rows_(expr.rows()), cols_(expr.cols()), stride_(L == Layout::RowMajor ? expr.cols() : expr.rows()),
            data_(expr.cols() * expr.rows(), val) {
            *this = expr;
        };

        /// @brief Constructor from a given `MatrixView`.
        /// @tparam Trans Transposition flag required by the view.
        /// @tparam Conj Conjugation flag required by the view.
        /// @tparam Mutable Mutability indicator.
        /// @param view The view.
        template<bool Trans, bool Conj, bool Mutable>
        Matrix(const MatrixView<T, L, Trans, Conj, Mutable>& view) : rows_(view.rows()), cols_(view.cols()), stride_(L == Layout::RowMajor ? view.cols() : view.rows()),
            data_(view.rows() * view.cols()) {
            *this = expr(view);
        };

        /// @brief Constructor from a 2D `std::array` object.
        /// @tparam rows Row count deduced from the array.
        /// @tparam cols Column count deduced likewise.
        /// @param arr The array.
        template<size_t rows, size_t cols>
        Matrix(const std::array<std::array<T, cols>, rows>& arr) : rows_(rows), cols_(cols), stride_(L == Layout::RowMajor ? cols : rows), data_(rows* cols) {
            const size_t total = rows * cols;
            if (total == 0) return;
            if constexpr (total < PARALLEL_THRESHOLD_SIMPLE) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        (*this)(i, j) = arr[i][j];
                    };
                };
            }
            else if constexpr (L == Layout::RowMajor) {
                parallel_for(rows, 1,
                    [this, &arr](size_t start_row, size_t end_row) {
                        for (size_t i = start_row; i < end_row; ++i) {
                            const auto& src_row = arr[i];
                            for (size_t j = 0; j < cols; ++j) {
                                (*this)(i, j) = src_row[j];
                            };
                        };
                    });
            }
            else {
                parallel_for(cols, 1,
                    [this, &arr](size_t start_col, size_t end_col) {
                        for (size_t j = start_col; j < end_col; ++j) {
                            for (size_t i = 0; i < rows; ++i) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
            };
        };

        /// @brief Constructor from a 2D C-styled array.
        /// @tparam rows Row count deduced from the array.
        /// @tparam cols Column count obtained likewise.
        /// @param arr The array. 
        template<size_t rows, size_t cols>
        Matrix(const T(&arr)[rows][cols]) : rows_(rows), cols_(cols),
            stride_(L == Layout::RowMajor ? cols : rows), data_(rows* cols) {
            const size_t total = rows * cols;
            if (total == 0) return;

            if constexpr (total < PARALLEL_THRESHOLD_SIMPLE) {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        (*this)(i, j) = arr[i][j];
                    };
                };
            }
            else if constexpr (L == Layout::RowMajor) {
                parallel_for(rows, 1,
                    [this, &arr](size_t start_row, size_t end_row) {
                        for (size_t i = start_row; i < end_row; ++i) {
                            for (size_t j = 0; j < cols; ++j) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
            }
            else {
                parallel_for(cols, 1,
                    [this, &arr](size_t start_col, size_t end_col) {
                        for (size_t j = start_col; j < end_col; ++j) {
                            for (size_t i = 0; i < rows; ++i) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
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
                parallel_for(rows, 1,
                    [this, &arr](size_t start_row, size_t end_row) {
                        for (size_t i = start_row; i < end_row; ++i) {
                            for (size_t j = 0; j < cols; ++j) {
                                (*this)(i, j) = arr[i][j];
                            };
                        };
                    });
            }
            else {
                parallel_for(cols, 1,
                    [this, &arr](size_t start_col, size_t end_col) {
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
            bool depends = e.depends_on(data_ptr, data_bytes);
            const size_t total = this->rows_ * this->cols_;

            if (total < PARALLEL_THRESHOLD_SIMPLE || depends) {
                if (depends) {
                    std::vector<T, AlignedAllocator<T>> temp(total);
                    size_t idx = 0;
                    for (size_t i = 0; i < this->rows_; ++i) {
                        for (size_t j = 0; j < this->cols_; ++j) {
                            temp[idx++] = e(i, j);
                        };
                    };
                    data_ = std::move(temp);
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
                    parallel_for(this->rows_, 1,
                        [this, &e](size_t start_row, size_t end_row) {
                            for (size_t i = start_row; i < end_row; ++i) {
                                for (size_t j = 0; j < this->cols_; ++j) {
                                    (*this)(i, j) = e(i, j);
                                };
                            };
                        });
                }
                else {
                    parallel_for(this->cols_, 1,
                        [this, &e](size_t start_col, size_t end_col) {
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
        /// @tparam rows 
        /// @tparam cols 
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

        /// @brief Safe element access, as operator() does not check if index is valid.
        /// @pre Matrix<T, L> A.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)`.
        /// @throws linalg::detail::BoundsError.
        T& at(size_t i, size_t j) {
            BOUNDS_CHECK(i < rows_ && j < cols_);
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };
 
        const T& at(size_t i, size_t j) const {
            BOUNDS_CHECK(i < rows_ && j < cols_);
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        // Unchecked element indexation
        LINALG_INLINE
        /// @brief Unchecked element indexation.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)` if such is legal, undefined otherwise. 
        T& operator()(size_t i, size_t j) {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        LINALG_INLINE
        const T& operator()(size_t i, size_t j) const {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        /// @brief Static factory method for creating n * n identity matrix.
        /// @param n Dimension.
        /// @return The identity matrix of size `n`.
        static Matrix identity(size_t n) {
            Matrix mat(n, n, T(0));
            for (size_t i = 0; i < n; ++i) { mat(i, i) = T(1); };
            return mat;
        };

        /// @brief Static factory method for creating m * n matrix initialised with all ones.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix ones(size_t m, size_t n) { return Matrix(m, n, T(1)); };

        /// @brief Static factory method for creating m * n matrix initialised with all zeros.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix zeros(size_t m, size_t n) { return Matrix(m, n, T(0)); };

        /// @brief Static factory method for creating m * n matrix initialised with random entries.
        /// @param m Row count.
        /// @param n Column count.
        /// @return The matrix.
        static Matrix random(size_t m, size_t n) {
            Matrix mat(m, n);
            size_t total = m * n;
            parallel_for(total, PARALLEL_THRESHOLD_SIMPLE,
                [&mat](size_t start, size_t end) {
                    for (size_t i = start; i < end; ++i) {
                        mat.data_[i] = randomScalar<T>();
                    };
                });
            return mat;
        };

    private:
        // Data storage and dimensions
        std::vector<T, AlignedAllocator<T>> data_;
        size_t rows_, cols_, stride_;

        template<typename U, Layout LL, bool Trans, bool Conj, bool Mutable> friend class MatrixView;
    };
};