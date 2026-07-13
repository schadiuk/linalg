#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>
#include <linalg/core/hints.hpp>

namespace linalg {
	// Forward declarations
	template<typename U> struct VecExpr;
	template<typename EM, typename EV> struct GemvExpr;
	template<typename EV, typename EM> struct VgemExpr;

	/// @brief Main vector storage class.
	/// @tparam T scalar element type. Supports: float, double, and their std::complex counterparts.
	template<typename T = DefaultScalar> requires Scalar<T>
	class Vector : public VecExpr<Vector<T>> {
	private:
		/// @brief Builds an `n`-slot buffer whose `i`-th slot is placement-constructed from `init_fn(i)`, in a single parallel write pass over freshly (uninitialised) allocated memory.
		template<typename F>
		static std::vector<T, UninitAlignedAllocator<T>> fill_construct(size_t n, F&& init_fn) {
			std::vector<T, UninitAlignedAllocator<T>> buf(n);
			T* LINALG_RESTRICT p = buf.data();
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [p, &init_fn](size_t start, size_t end) {
				for (size_t i = start; i < end; ++i) ::new (static_cast<void*>(p + i)) T(init_fn(i));
			});
			return buf;
		};

	public:
		/// @brief Empty vector constructor.
		/// @param n Size (length) parameter.
		Vector(size_t n = 0) : data_(fill_construct(n, [](size_t) { return T(); })), size_(n) {};

		/// @brief Constructor from initializer list.
		/// @param init The list.
		Vector(std::initializer_list<T> init) : data_(init), size_(init.size()) {};

		/// @brief Uniform fill-in constructor.
		/// @param n Size.
		/// @param val Fill value.
		Vector(size_t n, const T& val) : data_(fill_construct(n, [&val](size_t) { return val; })), size_(n) {};

		/// @brief Constructor from a given `VecExpr`.
		/// @tparam E CRTP-required parameter clause of expression.
		/// @param expr The expression.
		/// @note A freshly allocated buffer never aliases `expr`so this always constructs directly rather than going through the aliasing-aware `operator=` machinery.
		template<typename E>
		Vector(const VecExpr<E>& expr) : data_(fill_construct(expr.size(), [&e = expr.self()](size_t i) { return static_cast<T>(e(i)); })), size_(expr.size()) {};

		/// @brief Constructor from a given `VectorView` object.
		/// @tparam Mutable Mutability indicator.
		/// @param view The view.
		template<bool Mutable>
		Vector(const VectorView<T, Mutable>& view) : data_(fill_construct(view.size(), [&view](size_t i) { return view(i); })), size_(view.size()) {};

		/// @brief Constructor from a given `std::array` object.
		/// @tparam n Size parameter deduced from the array.
		/// @param arr The array.
		template<size_t n>
		Vector(const std::array<T, n>& arr) : data_(fill_construct(n, [&arr](size_t i) { return arr[i]; })), size_(n) {};

		/// @brief Assignment operator from an expression.
		/// @tparam E CRTP-required parameter clause.
		/// @param expr the expression.
		/// @return `*this` Vector<T> pointer.
		template<typename E>
		Vector<T>& operator=(const VecExpr<E>& expr) {
			const auto& e = expr.self();
			BOUNDS_CHECK(size_ == e.size());
			const void* data_ptr = data_.data();
			const size_t data_bytes = size_ * sizeof(T);

			constexpr bool elementwise = detail::expr_is_elementwise_v<E>;
			bool depends = !elementwise && e.depends_on(data_ptr, data_bytes);
			const size_t total = size_;
 
			if (total < PARALLEL_THRESHOLD_SIMPLE || depends) {
				if (depends) {
					// Build the replacement into a brand-new buffer (reads against the still-intact old `data_` happen entirely before the swap), then move it in.
					data_ = fill_construct(total, [&e](size_t i) { return static_cast<T>(e(i)); });
				}
				else {
					for (size_t i = 0; i < total; ++i) data_[i] = e(i);
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
        
		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
		bool depends_on(const void* p, size_t bytes) const {			
			const void* start = static_cast<const void*>(data_.data());
			const void* end = static_cast<const void*>(data_.data() + size_);
			const void* other_end = static_cast<const char*>(p) + bytes;
			return (p < end) && (other_end > start);
		};

		/// @brief O(1) member-wise swap: exchanges backing buffers/sizes rather than elements.
		/// @param other Vector to swap contents with.
		/// @note Both operands own independent, non-aliased heap buffers, so this is a pointer exchange regardless of length.
		void swap(Vector<T>& other) noexcept {
			data_.swap(other.data_);
			std::swap(size_, other.size_);
		};

		size_t size() const { return size_; };
		T* data() { return data_.data(); };
		const T* data() const { return data_.data(); };

		/// @brief Safe element access, as operator() does not check if index is valid.
        /// @pre Vector<T> Vec.
        /// @param i Index.
        /// @return Element given by `Vec(i)`.
        /// @throws linalg::detail::BoundsError.
		T& at(size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		const T& at(size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

        // Unchecked element indexation. NOTE: both operator [] and () are provided
		LINALG_INLINE
		/// @brief Unchecked element indexation.
		/// @param i Index.
		/// @return Element given by `Vec[i]` if such exists, undefined otherwise.
		/// @note Both operator() and operator[] exist, and are equivalent.
		T& operator[](size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		LINALG_INLINE
		const T& operator[](size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		LINALG_INLINE
		/// @brief Unchecked element indexation.
		/// @param i Index.
		/// @return Element given by `Vec(i)` if such exists, undefined otherwise.
		/// @note Both operator() and operator[] exist, and are equivalent.
		T& operator()(size_t i) {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		LINALG_INLINE
		const T& operator()(size_t i) const {
			BOUNDS_CHECK(i < size_);
			return data_[i];
		};

		// std::vector-like iterators
		auto begin() { return data_.begin(); };
		auto end() { return data_.end(); };
		auto begin() const { return data_.begin(); };
		auto end() const { return data_.end(); };

		/// @brief Static factory method that produces `n`-long vector initialised with all ones.
		/// @param n Size.
		/// @return The vector.
		static Vector ones(size_t n) { return Vector(n, T(1.)); };

		/// @brief Static factory method that produces `n`-long vector initialised with all zeros.
		/// @param n Size.
		/// @return The vector.
		static Vector zeros(size_t n) { return Vector(n, T(0.)); };

		/// @brief Static factory method that produces `n`-long vector initialised with random entries.
		/// @param n Size.
		/// @return The vector.
		static Vector random(size_t n) {
			Vector vec;
			vec.data_ = fill_construct(n, [](size_t) { return randomScalar<T>(); });
			vec.size_ = n;
			return vec;
		};

		template<typename EM, typename EV>
		Vector(const GemvExpr<EM, EV>& expr);

		template<typename EM, typename EV>
		Vector<T>& operator=(const GemvExpr<EM, EV>& expr);
 
		template<typename EV, typename EM>
		Vector(const VgemExpr<EV, EM>& expr);

		template<typename EV, typename EM>
		Vector<T>& operator=(const VgemExpr<EV, EM>& expr);

	private:
	    // Data storage and dimension
		std::vector<T, UninitAlignedAllocator<T>> data_;
		size_t size_;

        template<typename U, bool M> friend class VectorView;
	};
};