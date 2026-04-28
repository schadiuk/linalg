#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>

namespace linalg {
    // Forward declaration of expression template class
    template<typename U> struct MatExpr;

    // Main storage class for matrices that supports both memory layouts
    template<typename T = DefaultScalar, Layout L = Layout::RowMajor> requires Scalar<T>
    class Matrix : public MatExpr<Matrix<T, L>> {
    public:
        // Empty Matrix constructor
        Matrix(size_t m = 0, size_t n = 0) : rows_(m), cols_(n), stride_(L == Layout::RowMajor ? n : m), data_(m * n) {};

        // Parallelised constructor that fills matrix with a value given by the 3rd argument
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

        // Constructor from a given expression
        template<typename E>
        Matrix(const MatExpr<E>& expr, const T& val = T(0)) : rows_(expr.rows()), cols_(expr.cols()), stride_(L == Layout::RowMajor ? expr.cols() : expr.rows()),
            data_(expr.cols()* expr.rows(), val) {
            *this = expr;
        };

        // Constructor from a given view
        template<bool Trans, bool Conj, bool Mutable>
        Matrix(const MatrixView<T, L, Trans, Conj, Mutable>& view) : rows_(view.rows()), cols_(view.cols()), stride_(L == Layout::RowMajor ? view.cols() : view.rows()),
            data_(view.rows()* view.cols()) {
            //*this = expr(view);
            *this = view;
        };

        // Constructor from std::array
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

        // Constructor from C-styled 2d array
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

        // Assignent from C-styled array
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

        // Assignment from expression
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
                    std::vector<T> temp(total);
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

        // Aliasing detection
		bool depends_on(const void* p, size_t bytes) const {			
			const void* start = static_cast<const void*>(data_.data());
			const void* end = static_cast<const void*>(data_.data() + data_.size());
			const void* other_end = static_cast<const char*>(p) + bytes;
			return (p < end) && (other_end > start);
		};

        // Conversion helper
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

        // Reshape Matrix to the given dimensions, keeping data and Layout
        void reshape(size_t new_rows, size_t new_cols) {
            BOUNDS_CHECK(new_rows * new_cols == rows_ * cols_);
            rows_ = new_rows;
            cols_ = new_cols;
            stride_ = (L == Layout::RowMajor) ? cols_ : rows_;
        };

        // Accessors
        size_t rows() const { return rows_; };
        size_t cols() const { return cols_; };
        size_t stride() const { return stride_; };
        T* data() { return data_.data(); };
        const T* data() const { return data_.data(); };

        // Safe element indexation        
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
        T& operator()(size_t i, size_t j) {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        const T& operator()(size_t i, size_t j) const {
            size_t idx = (L == Layout::RowMajor) ? (i * stride_ + j) : (j * stride_ + i);
            return data_[idx];
        };

        // Static factory methods
        static Matrix identity(size_t n) {
            Matrix mat(n, n, T(0));
            for (size_t i = 0; i < n; ++i) { mat(i, i) = T(1); };
            return mat;
        };

        static Matrix ones(size_t m, size_t n) { return Matrix(m, n, T(1)); };

        static Matrix zeros(size_t m, size_t n) { return Matrix(m, n, T(0)); };

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
        std::vector<T> data_;
        size_t rows_, cols_, stride_;

        template<typename U, Layout LL, bool Trans, bool Conj, bool Mutable> friend class MatrixView;
    };
};