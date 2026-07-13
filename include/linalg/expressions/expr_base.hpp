#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/hints.hpp>

namespace linalg {
	template<typename T, Layout L> requires Scalar<T> class Matrix;
	template<typename T> requires Scalar<T> class Vector;
	template<typename T, Layout L, bool Trans, bool Conj, bool Mutable> class MatrixView;

	namespace detail {
		template<typename E>
		struct is_owning_leaf : std::false_type {};

		template<typename T, Layout L>
		struct is_owning_leaf<Matrix<T, L>> : std::true_type {};

		template<typename T>
		struct is_owning_leaf<Vector<T>> : std::true_type {};

		template<typename E>
		inline constexpr bool is_owning_leaf_v = is_owning_leaf<E>::value;
 
		template<typename E>
		using operand_t = std::conditional_t<is_owning_leaf_v<E>, const E&, E>;
	};

	/// @brief  Key matrix expression base class, providing common interface and used in lazy CRTP-induced evaluation.
	/// @tparam T CRTP-required template.
	template<typename T>
	struct MatExpr {
		/// @brief `*this` pointer accessor.
		T& self() { return static_cast<T&>(*this); };
		const T& self() const { return static_cast<const T&>(*this); };

		/// @brief Row count accessor.
		/// @return Row count.
		LINALG_INLINE
		size_t rows() const { return self().rows(); };

		/// @brief column count accessor.
		/// @return Column count.
		LINALG_INLINE
		size_t cols() const { return self().cols(); };

		/// @brief Element indexation.
		/// @param i Row index.
		/// @param j Column index.
		/// @return Element with given indices `A(i, j)`.
		auto operator()(size_t i, size_t j) const { return self()(i, j); };

		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes
		/// @return Boolean indicator.
		bool depends_on(const void* p, size_t bytes) const { return self().depends_on(p, bytes); };
	};

    /// @brief Key vector expression base class, providing common interface and used in lazy CRTP-induced evaluation.
    /// @tparam T CRTP-required template.
    template<typename T>
    struct VecExpr {
		/// @brief `*this` pointer accessor. 
		T& self() { return static_cast<T&>(*this); };
	    const T& self() const { return static_cast<const T&>(*this); };
		
	    /// @brief Size accessor.
	    /// @return Size (length) of the vector.
		LINALG_INLINE
	    size_t size() const { return self().size(); };

	    /// @brief Element access.
	    /// @param i Index.
	    /// @return Element `Vec[i]`.
	    auto operator()(size_t i) const { return self()(i); };

		/// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes
		/// @return Boolean indicator.
	    bool depends_on(const void* p, size_t bytes) const { return self().depends_on(p, bytes); };
    };
};