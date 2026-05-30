#pragma once

#include <linalg/storage/vector.hpp>

namespace linalg {
    /// @brief View class.
    /// @tparam T Scalar element type. Supports: float, double, and their std::complex counterparts.
    /// @tparam Mutable Mutability indicator.
    template<typename T, bool Mutable = false>
    class VectorView : public VecExpr<VectorView<T, Mutable>> {
        using Ptr = std::conditional_t<Mutable, T*, const T*>;
    public:
        /// @brief Read-only constructor from a const Vector.
		/// @param vet The vector.
        explicit VectorView(const Vector<T>& vec) requires (!Mutable) : data_(vec.data()), size_(vec.size()), stride_(1) {};
    
        /// @brief Constructor from a non-const Vector.
		/// @param vet The vector.
        explicit VectorView(Vector<T>& vec) requires Mutable : data_(vec.data()), size_(vec.size()), stride_(1) {};
    
        /// @brief Raw-pointer constructor.
        /// @param data Pointer to element (0,0) of this view's logical extent.
        /// @param size Physical size.
        /// @param stride Stride of the parent allocation.
        VectorView(Ptr data, size_t size, size_t stride = 1) : data_(data), size_(size), stride_(stride) {};
    
        /// @brief Conversion operator with implicit narrowing: mutable -> const.
        operator VectorView<T, false>() const requires Mutable {
            return VectorView<T, false>(static_cast<const T*>(data_), size_, stride_);
        };

        size_t size() const { return size_; };
        size_t stride() const { return stride_;};
        Ptr data() const { return data_; };

        /// @brief Checked element indexation, read-only.
		/// @param i Index.
		/// @return Element given by `Vec(i)` if such exists, undefined otherwise.
		/// @note Both operator() and operator[] exist, and are equivalent.
        /// @note Methods, analoguous to `at()`, were removed.
        const T& operator()(size_t i) const {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        const T& operator[](size_t i) const {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        /// @brief Checked element indexation, allows writing to mutable views.
		/// @param i Index.
		/// @return Element given by `Vec(i)` if such exists, undefined otherwise.
		/// @note Both operator() and operator[] exist, and are equivalent.
        /// @note Methods, analoguous to `at()`, were removed.
        T& operator()(size_t i) requires Mutable {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        T& operator[](size_t i) requires Mutable {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };

        /// @brief Assignment operator from an expression for mutable views.
        /// @tparam E CRTP-required parameter clause of `VecExpr`.
        /// @param expr The expression.
        /// @return `*this` pointer.
        template<typename E>
        VectorView& operator=(const VecExpr<E>& expr) requires Mutable {
            const auto& e = expr.self();
            BOUNDS_CHECK(size_ == e.size());
            const bool depends = e.depends_on(static_cast<const void*>(data_), memory_span());
            if (depends) {
                std::vector<T> temp(size_);
                for (size_t i = 0; i < size_; ++i) temp[i] = e(i);
                for (size_t i = 0; i < size_; ++i) (*this)(i) = temp[i];
            } else {
                for (size_t i = 0; i < size_; ++i) (*this)(i) = e(i);
            };
            return *this;
        };

		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
	    bool depends_on(const void* p, size_t bytes) const {
            if (size_ == 0) return false;
            const void* start = static_cast<const void*>(data_);
            const void* end = static_cast<const void*>(data_ + (size_ - 1) * stride_ + 1);
            const void* other_end = static_cast<const char*>(p) + bytes;
            return (p < end) && (other_end > start);
        };

    private:
        /// @brief Memory structure helper.
	    size_t memory_span() const { return (size_ == 0) ? 0 : ((size_ - 1) * stride_ + 1) * sizeof(T); };

        // Data pointer and dimensions
        Ptr data_;
        size_t size_, stride_;
    
        template<typename U, bool M> friend struct VecViewRef;
    };
 
    /// @brief Mutable view of a vector.
    /// @param vec Non-constant vector.
    /// @return The view.
    template<typename T>
    VectorView<T, true>  view(Vector<T>& vec) { return VectorView<T, true >(vec); };

    /// @brief Immutable view of a vector.
    /// @param vec Constant vector.
    /// @return The view.
    template<typename T>
    VectorView<T, false> view(const Vector<T>& vec) { return VectorView<T, false>(vec); };
};