#pragma once

#include <linalg/storage/matrix.hpp>

namespace linalg {
	template<typename T, Layout L> requires Scalar<T>
	class Matrix;

    // View class supporting zero-overhead transposition and complex conjugation
	template<typename T, Layout L, bool Trans = false, bool Conj = false, bool Mutable = false>
	class MatrixView : public MatExpr<MatrixView<T, L, Trans, Conj, Mutable>> {
		using Ptr = std::conditional_t<Mutable, T*, const T*>;
	public:
        // Constructor from const Matrix (always read-only)
    	explicit MatrixView(const Matrix<T, L>& mat) requires (!Mutable) : data_(mat.data()), rows_(mat.rows()), cols_(mat.cols()),
        		stride_(L == Layout::RowMajor ? mat.cols() : mat.rows()) {};
 
    	// Constructor from mutable Matrix
    	explicit MatrixView(Matrix<T, L>& mat) requires Mutable : data_(mat.data()), rows_(mat.rows()), cols_(mat.cols()),
          		stride_(L == Layout::RowMajor ? mat.cols() : mat.rows()) {};

		// Raw-pointer constructor with:
		// data: pointer to element (0,0) of this view's logical extent
		// rows: physical row count (before applying Trans)
		// cols: physical column count (before applying Trans)
		// stride: leading dimension of the parent allocation
    	MatrixView(Ptr data, size_t rows, size_t cols, size_t stride) : data_(data), rows_(rows), cols_(cols), stride_(stride) {};
		
		// Implicit narrowing: mutable -> const
		operator MatrixView<T, L, Trans, Conj, false>() const requires Mutable {
			return MatrixView<T, L, Trans, Conj, false>(static_cast<const T*>(data_), rows_, cols_, stride_);
		};

        // Accessors
		size_t rows() const { return Trans ? cols_ : rows_; };
		size_t cols() const { return Trans ? rows_ : cols_; };
		size_t stride() const { return stride_; };
		Ptr data() const { return data_; };

        size_t index(size_t i, size_t j) const {
			const size_t ii = Trans ? j : i;
			const size_t jj = Trans ? i : j;
			if constexpr (L == Layout::RowMajor) return ii * stride_ + jj;
			return jj * stride_ + ii;
		};


	T at(size_t i, size_t j) const {
		BOUNDS_CHECK(i < rows() && j < cols());
		return Conj ? conj(data_[index(i, j)]) : data_[index(i, j)];
	};

		const T& operator()(size_t i, size_t j) const requires (!Conj) {
			return data_[index(i, j)];
		};
	
		// Read access, conjugate: returns T by value
		T operator()(size_t i, size_t j) const requires Conj {
			return conj(data_[index(i, j)]);
		};
	
		// Write access: only for mutable, non-conjugate views.
		T& operator()(size_t i, size_t j) requires (Mutable && !Conj) {
			return data_[index(i, j)];
		};

        // Aliasing detection
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
		 // Memory structure helper
		size_t memory_span() const {
			if (rows_ == 0 || cols_ == 0) return 0;
			if constexpr (L == Layout::RowMajor) return ((rows_ - 1) * stride_ + cols_) * sizeof(T);
			else return ((cols_ - 1) * stride_ + rows_) * sizeof(T);
    	};

        // Data storage and dimensions
		Ptr data_;
		size_t rows_, cols_, stride_;
	};

    // Helper functions to create views
	template<typename T, Layout L>
	MatrixView<T, L, false, false, true>  view(Matrix<T, L>& mat) {
		return MatrixView<T, L, false, false, true>(mat);
	};
	
	template<typename T, Layout L>
	MatrixView<T, L, false, false, false> view(const Matrix<T, L>& mat) {
		return MatrixView<T, L, false, false, false>(mat);
	};
	
    template<typename T, Layout L>
    auto transpose(Matrix<T, L>& mat) { return MatrixView<T, L, true, false, true>(mat); };

    template<typename T, Layout L>
    auto transpose(const Matrix<T, L>& mat) { return MatrixView<T, L, true, false, false>(mat); };

	// Semantically read-only: one cannot deduce the intent when writing to the view
	template<typename T, Layout L>
    auto hermitian(const Matrix<T, L>& mat) { return MatrixView<T, L, true, true, false>(mat); };
};