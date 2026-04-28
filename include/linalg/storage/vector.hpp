#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>

namespace linalg {
	// Forward declaration of expression template class
	template<typename U> struct VecExpr;

	// Main storage class for vectors
	template<typename T = DefaultScalar> requires Scalar<T>
	class Vector : public VecExpr<Vector<T>> {
	public:
		// Empty Vector constructor
		Vector(size_t n = 0) : data_(n), size_(n) {};

		// Constructor from initialiser list
		Vector(std::initializer_list<T> init) : data_(init), size_(init.size()) {};

		Vector(size_t n, const T& val) : data_(), size_(n) {
			data_.resize(n);
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [this, val](size_t start, size_t end) {
				for (size_t i = start; i < end; ++i) {
					data_[i] = val;
				}
				});
		};

		// Constructor from a given expression
		template<typename E>
		Vector(const VecExpr<E>& expr) : data_(expr.size()), size_(expr.size()) { *this = expr; };

		// Constructor from a given view
		template<bool Mutable>
		Vector(const VectorView<T, Mutable>& view) : data_(view.size()), size_(view.size()) {
			for (size_t i = 0; i < size_; ++i) {
				data_[i] = view(i);
			};
		};

	    // Constructor from std::array	
		template<size_t n>
		Vector(const std::array<T, n>& arr) : data_(n), size_(n) {
			if constexpr (n < PARALLEL_THRESHOLD_SIMPLE / 4) {
				std::copy(arr.begin(), arr.end(), data_.begin());
			}
			else {
				parallel_for(n, PARALLEL_THRESHOLD_SIMPLE,
					[this, &arr](size_t start, size_t end) {
						for (size_t i = start; i < end; ++i) {
							data_[i] = arr[i];
						};
					});
			};
		};

        // Assignment from expression
		template<typename E>
		Vector<T>& operator=(const VecExpr<E>& expr) {
			const auto& e = expr.self();
			BOUNDS_CHECK(size_ == e.size());
			const void* data_ptr = data_.data();
			const size_t data_bytes = size_ * sizeof(T);
			bool depends = e.depends_on(data_ptr, data_bytes);
			const size_t total = size_;

			if (total < PARALLEL_THRESHOLD_SIMPLE || depends) {
				if (depends) {
					std::vector<T> temp(total);
					for (size_t i = 0; i < total; ++i) {
						temp[i] = e(i);
					};
					data_ = std::move(temp);
				}
				else {
					for (size_t i = 0; i < total; ++i) {
						data_[i] = e(i);
					};
				};
			}
			else {
				parallel_for(total, PARALLEL_THRESHOLD_SIMPLE,
					[this, &e](size_t start, size_t end) {
						for (size_t i = start; i < end; ++i) {
							data_[i] = e(i);
						};
					});
			};
			return *this;
		};
        
		// Aliasing detection
		bool depends_on(const void* p, size_t bytes) const {			
			const void* start = static_cast<const void*>(data_.data());
			const void* end = static_cast<const void*>(data_.data() + size_);
			const void* other_end = static_cast<const char*>(p) + bytes;
			return (p < end) && (other_end > start);
		};

		// Accessors
		size_t size() const { return size_; };
		T* data() { return data_.data(); };
		const T* data() const { return data_.data(); };

		// Safe element indexation
		T& at(size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		const T& at(size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

        // Unchecked element indexation. NOTE: both operator [] and () are provided
		T& operator[](size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		const T& operator[](size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		T& operator()(size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		const T& operator()(size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		// std::vector-like iterators
		auto begin() { return data_.begin(); };
		auto end() { return data_.end(); };
		auto begin() const { return data_.begin(); };
		auto end() const { return data_.end(); };

		// Static factory methods
		static Vector ones(size_t n) { return Vector(n, T(1.)); };
		static Vector zeros(size_t n) { return Vector(n, T(0.)); };

		static Vector random(size_t n) {
			Vector vec(n);
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&vec](size_t start, size_t end) {
				for (size_t i = start; i < end; ++i) {
					vec.data_[i] = randomScalar<T>();
				};
				});
			return vec;
		};

	private:
	    // Data storage and dimension
		std::vector<T> data_;
		size_t size_;

        template<typename U, bool M> friend class VectorView;
	};
};