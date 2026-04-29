#pragma once

namespace linalg {
	template<typename T, Layout L> requires Scalar<T> class Matrix;
	template<typename T, Layout L, bool Trans, bool Conj, bool Mutable> class MatrixView;

	// Key Matrix expression base class, providing common interface and used in lazy CRTP-induced evaluation
	template<typename T>
	struct MatExpr {
		T& self() { return static_cast<T&>(*this); };
		const T& self() const { return static_cast<const T&>(*this); };
		
		size_t rows() const { return self().rows(); };
		size_t cols() const { return self().cols(); };

		auto operator()(size_t i, size_t j) const { return self()(i, j); };

		bool depends_on(const void* p, size_t bytes) const { return self().depends_on(p, bytes); };
	};

	// Key Vector expression base class
    template<typename T>
    struct VecExpr {
		T& self() { return static_cast<T&>(*this); };
	    const T& self() const { return static_cast<const T&>(*this); };
		
	    size_t size() const { return self().size(); };

	    auto operator()(size_t i) const { return self()(i); };

	    bool depends_on(const void* p, size_t bytes) const { return self().depends_on(p, bytes); };
    };
};