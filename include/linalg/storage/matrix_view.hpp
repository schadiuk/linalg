#pragma once

#include <linalg/storage/matrix.hpp>

namespace linalg {
	template<typename T, Layout L> requires Scalar<T>
	class Matrix;

	/// @brief View class supporting zero-overhead transposition and complex conjugation.
	/// @tparam T Scalar element type. Supports: float, double, and their std::complex counterparts.
	/// @tparam L Layout.
	/// @tparam Trans Transposition flag.
	/// @tparam Conj Conjugation flag.
	/// @tparam Mutable Mutability flag.
	template<typename T, Layout L, bool Trans = false, bool Conj = false, bool Mutable = false>
	class MatrixView : public MatExpr<MatrixView<T, L, Trans, Conj, Mutable>> {
		using Ptr = std::conditional_t<Mutable, T*, const T*>;
	public:
        /// @brief Read-only constructor from a const Matrix.
		/// @param mat Constant Matrix.
    	explicit MatrixView(const Matrix<T, L>& mat) requires (!Mutable) : data_(mat.data()), rows_(mat.rows()), cols_(mat.cols()),
        		stride_(L == Layout::RowMajor ? mat.cols() : mat.rows()) {};
 
    	/// @brief Constructor from non-const (mutable) Matrix.
		/// @param mat The Matrix object.
    	explicit MatrixView(Matrix<T, L>& mat) requires Mutable : data_(mat.data()), rows_(mat.rows()), cols_(mat.cols()),
          		stride_(L == Layout::RowMajor ? mat.cols() : mat.rows()) {};

    	/// @brief Raw-pointer constructor.
    	/// @param data Pointer to element (0,0) of this view's logical extent.
    	/// @param rows Physical row count (before applying Trans).
    	/// @param cols Physical column count (before applying Trans).
    	/// @param stride Leading dimension of the parent allocation.
    	MatrixView(Ptr data, size_t rows, size_t cols, size_t stride) : data_(data), rows_(rows), cols_(cols), stride_(stride) {};
		
		/// @brief Conversion operator with implicit narrowing: mutable -> const.
		operator MatrixView<T, L, Trans, Conj, false>() const requires Mutable {
			return MatrixView<T, L, Trans, Conj, false>(static_cast<const T*>(data_), rows_, cols_, stride_);
		};

		size_t rows() const { return Trans ? cols_ : rows_; };
		size_t cols() const { return Trans ? rows_ : cols_; };
		size_t stride() const { return stride_; };
		Ptr data() const { return data_; };

        /// @brief Indexation helper.
        size_t index(size_t i, size_t j) const {
			const size_t ii = Trans ? j : i;
			const size_t jj = Trans ? i : j;
			if constexpr (L == Layout::RowMajor) return ii * stride_ + jj;
			return jj * stride_ + ii;
		};

		/// @brief Safe element access, as operator() does not check if index is valid.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)`.
        /// @throws linalg::detail::BoundsError.
		T at(size_t i, size_t j) const {
			BOUNDS_CHECK(i < rows() && j < cols());
			return Conj ? conj(data_[index(i, j)]) : data_[index(i, j)];
		};

		/// @brief Unchecked element indexation.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Element with given indices `A(i, j)`. 
		const T& operator()(size_t i, size_t j) const requires (!Conj) {
			return data_[index(i, j)];
		};
	
		/// @brief Unchecked element read-only element access.
        /// @param i Row index.
        /// @param j Column index.
        /// @return Conjugate of the element with given indices `conj(A(i, j))`. 
		T operator()(size_t i, size_t j) const requires Conj {
			return conj(data_[index(i, j)]);
		};
	
		/// @brief Write access.
        /// @param i Row index.
        /// @param j Column index. 
		/// @return Element with given indices `A(i, j)`.
		/// @note  Works only for mutable, non-conjugate views.
		T& operator()(size_t i, size_t j) requires (Mutable && !Conj) {
			return data_[index(i, j)];
		};

        /// @brief  Assignment from a given matrix expression.
        /// @tparam E CRTP-required parameter clause of `MatExpr`.
        /// @param expr The expression. 
        /// @return `*this` pointer.
        template<typename E>
        MatrixView& operator=(const MatExpr<E>& expr) requires (Mutable && !Conj) {
            const auto& e = expr.self();
            BOUNDS_CHECK(rows() == e.rows() && cols() == e.cols());
            const bool dep = e.depends_on(static_cast<const void*>(data_), memory_span());
            if (dep) {
                // Materialise into a temporary to break the aliasing cycle
                std::vector<T> temp(rows() * cols());
                size_t k = 0;
                for (size_t i = 0; i < rows(); ++i)
                    for (size_t j = 0; j < cols(); ++j)
                        temp[k++] = static_cast<T>(e(i, j));
                k = 0;
                for (size_t i = 0; i < rows(); ++i)
                    for (size_t j = 0; j < cols(); ++j)
                        (*this)(i, j) = temp[k++];
            } else {
                for (size_t i = 0; i < rows(); ++i)
                    for (size_t j = 0; j < cols(); ++j)
                        (*this)(i, j) = static_cast<T>(e(i, j));
            };
            return *this;
        };

		/// @brief  Homogeneous copy between views of any Trans/Conj combination. 
		template<bool T2, bool C2, bool M2>
        MatrixView& operator=(const MatrixView<T, L, T2, C2, M2>& other) requires (Mutable && !Conj) {
            return *this = static_cast<const MatExpr<MatrixView<T, L, T2, C2, M2>>&>(other);
        };

		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator. 
		bool depends_on(const void* p, size_t bytes) const {
			if (rows_ == 0 || cols_ == 0) return false;
			const void* start = static_cast<const void*>(data_);
			const void* end;
			if constexpr (L == Layout::RowMajor)
				end = static_cast<const void*>(data_ + (rows_ - 1) * stride_ + cols_);
			else
				end = static_cast<const void*>(data_ + (cols_ - 1) * stride_ + rows_);
			const void* other_end = static_cast<const char*>(p) + bytes;
			return (p < end) && (other_end > start);
		};


	private:
		/// @brief  Memory structure helper
		size_t memory_span() const {
			if (rows_ == 0 || cols_ == 0) return 0;
			if constexpr (L == Layout::RowMajor) return ((rows_ - 1) * stride_ + cols_) * sizeof(T);
			else return ((cols_ - 1) * stride_ + rows_) * sizeof(T);
    	};

        // Data storage and dimensions
		Ptr data_;
		size_t rows_, cols_, stride_;
	};

	/// @brief Helper function to create mutable views.
	/// @param mat Non-const matrix.
	/// @return The view.
	template<typename T, Layout L>
	MatrixView<T, L, false, false, true> view(Matrix<T, L>& mat) {
		return MatrixView<T, L, false, false, true>(mat);
	};
	
	/// @brief Helper function to create read-only views.
	/// @param mat Const matrix.
	/// @return The view.
	template<typename T, Layout L>
	MatrixView<T, L, false, false, false> view(const Matrix<T, L>& mat) {
		return MatrixView<T, L, false, false, false>(mat);
	};
	
    /// @brief Mutable transpose of a matrix.
    /// @param mat Non-const matrix.
    /// @return The view.
    template<typename T, Layout L>
    auto transpose(Matrix<T, L>& mat) { return MatrixView<T, L, true, false, true>(mat); };

    /// @brief Read-only transpose of a matrix.
    /// @param mat Const matrix.
    /// @return The view.
    template<typename T, Layout L>
    auto transpose(const Matrix<T, L>& mat) { return MatrixView<T, L, true, false, false>(mat); };

	/// @brief Hermitian (conjugate transpose) view of a matrix. 
	/// @param mat The matrix.
	/// @return The view.
	/// @note Semantically read-only: one cannot deduce the intent when writing to the view.
	template<typename T, Layout L>
    auto hermitian(const Matrix<T, L>& mat) { return MatrixView<T, L, true, true, false>(mat); };
};