#pragma once

#include <linalg/storage/vector.hpp>

namespace linalg {
    template<typename T, bool Mutable = false>
    class VectorView : public VecExpr<VectorView<T, Mutable>> {
        using Ptr = std::conditional_t<Mutable, T*, const T*>;
    public:
        // From const Vector: always produces a read-only view
        explicit VectorView(const Vector<T>& vec) requires (!Mutable) : data_(vec.data()), size_(vec.size()), stride_(1) {};
    
        // From mutable Vector
        explicit VectorView(Vector<T>& vec) requires Mutable : data_(vec.data()), size_(vec.size()), stride_(1) {};
    
        // Raw-pointer constructor
        VectorView(Ptr data, size_t size, size_t stride = 1) : data_(data), size_(size), stride_(stride) {};
    
        // Implicit narrowing: mutable -> const
        operator VectorView<T, false>() const requires Mutable {
            return VectorView<T, false>(static_cast<const T*>(data_), size_, stride_);
        };

        // Accesors
        size_t size() const { return size_; };
        size_t stride() const { return stride_;};
        Ptr data() const { return data_; };

        // Read access: always available, avoids copying data
        const T& operator()(size_t i) const {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        const T& operator[](size_t i) const {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        // Write access: only synthesised for mutable views
        T& operator()(size_t i) requires Mutable {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };
    
        T& operator[](size_t i) requires Mutable {
            BOUNDS_CHECK(i < size_);
            return data_[i * stride_];
        };

        // Assignment from expression: only mutable views
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

        // Aliasing detection
	    bool depends_on(const void* p, size_t bytes) const {
            if (size_ == 0) return false;
            const void* start = static_cast<const void*>(data_);
            const void* end = static_cast<const void*>(data_ + (size_ - 1) * stride_ + 1);
            const void* other_end = static_cast<const char*>(p) + bytes;
            return (p < end) && (other_end > start);
        };

    private:
        // Memory structure: a pointer to the first element, the number of elements, and the stride (in elements)
	    size_t memory_span() const { return (size_ == 0) ? 0 : ((size_ - 1) * stride_ + 1) * sizeof(T); };

        // Data pointer and dimensions
        Ptr data_;
        size_t size_, stride_;
    
        template<typename U> friend struct VecViewRef;
    };
 
    template<typename T>
    VectorView<T, true>  view(Vector<T>& vec) { return VectorView<T, true >(vec); };
    
    template<typename T>
    VectorView<T, false> view(const Vector<T>& vec) { return VectorView<T, false>(vec); };
};