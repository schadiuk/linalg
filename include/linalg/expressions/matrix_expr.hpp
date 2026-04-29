#pragma once

#include <linalg/expressions/expr_base.hpp>

namespace linalg {
    template<typename T, Layout L>
    struct MatRef : MatExpr<MatRef<T, L>> {
	    const Matrix<T, L>& mat;

	    explicit MatRef(const Matrix<T, L>& m) : mat(m) {};

	    size_t rows() const { return mat.rows(); };
	    size_t cols() const { return mat.cols(); };
        size_t stride() const { return mat.stride(); };

	    T operator()(size_t i, size_t j) const { return mat(i, j); };

	    bool depends_on(const void* p, size_t bytes) const { return mat.depends_on(p, bytes); };
    };

    template<typename T, Layout L, bool Trans, bool Conj, bool Mutable = false> // Mutable = true preserves write-back capability when the wrapper is used as an assignment target
    struct MatViewExpr : MatExpr<MatViewExpr<T, L, Trans, Conj, Mutable>> {
        // Storage by value
        MatrixView<T, L, Trans, Conj, Mutable> view;

        explicit MatViewExpr(const MatrixView<T, L, Trans, Conj>& v) : view(v) {};

        // Implicit conversion to non-owning view, with mutable -> const allowed
        operator MatViewExpr<T, L, Trans, Conj, false>() const requires Mutable {
            return MatViewExpr<T, L, Trans, Conj, false>(view);
        };
        // Accessors
        size_t rows() const { return view.rows(); };
        size_t cols() const { return view.cols(); };
        size_t stride() const { return view.stride(); };

        // Read path: always available
        T operator()(size_t i, size_t j) const { return view(i, j); };
 
        // Write path: only synthesised for mutable wrappers
        T& operator()(size_t i, size_t j) requires (Mutable && !Conj) { return view(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return view.depends_on(p, bytes); };
    };

    // Addition expression
    template<typename E1, typename E2>
    struct MatAddExpr : MatExpr<MatAddExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        MatAddExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(e1.rows() == e2.rows() && e1.cols() == e2.cols()); };

        size_t rows() const { return e1.rows(); };
        size_t cols() const { return e1.cols(); };

        auto operator()(size_t i, size_t j) const { return e1(i, j) + e2(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    // Subtraction expression
    template<typename E1, typename E2>
    struct MatSubExpr : MatExpr<MatSubExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        MatSubExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(e1.rows() == e2.rows() && e1.cols() == e2.cols()); };

        size_t rows() const { return e1.rows(); };
        size_t cols() const { return e1.cols(); };

        auto operator()(size_t i, size_t j) const { return e1(i, j) - e2(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    // Hadamard product
    template<typename E1, typename E2>
    struct MatMulExpr : MatExpr<MatMulExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        MatMulExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(e1.rows() == e2.rows() && e1.cols() == e2.cols()); };

        size_t rows() const { return e1.rows(); };
        size_t cols() const { return e1.cols(); };

        auto operator()(size_t i, size_t j) const { return e1(i, j) * e2(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    // Scalar multiplication
    template<typename T, typename E>
    struct ScMatMulExpr : MatExpr<ScMatMulExpr<T, E>> {
        T scalar;
        const E& expr;

        ScMatMulExpr(T s, const E& e) : scalar(s), expr(e) {};

        size_t rows() const { return expr.rows(); };
        size_t cols() const { return expr.cols(); };

        T operator()(size_t i, size_t j) const { return expr(i, j) * scalar; };

        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    // GEneral Matrix Multiplication
    template<typename E1, typename E2>
    struct GemmExpr : MatExpr<GemmExpr<E1, E2>> {
        const E1& a;
        const E2& b;

        GemmExpr(const E1& a0, const E2& b0) : a(a0), b(b0) { BOUNDS_CHECK(a.cols() == b.rows()); };

        size_t rows() const { return a.rows(); };
        size_t cols() const { return b.cols(); };

        auto operator()(size_t i, size_t j) const {
            using Type = decltype(a(i, 0)* b(0, j));
            Type sum = Type(0);
            for (size_t k = 0; k < a.cols(); ++k) { sum += a(i, k) * b(k, j); };
            return sum;
        };

        bool depends_on(const void* p, size_t bytes) const { return a.depends_on(p, bytes) || b.depends_on(p, bytes); };
    };

    template<typename E1, typename E2>
    auto operator+(const MatExpr<E1>& a, const MatExpr<E2>& b) {
        return MatAddExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto operator-(const MatExpr<E1>& a, const MatExpr<E2>& b) {
        return MatSubExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto operator*(const MatExpr<E1>& a, const MatExpr<E2>& b) {
        return GemmExpr<E1, E2>(a.self(), b.self());
    };

    template<typename T, typename E> requires Scalar<T>
    auto operator*(const MatExpr<E>& e, T s) {
        return ScMatMulExpr<T, E>(s, e.self());
    };

    template<typename T, typename E> requires Scalar<T>
    auto operator*(T s, const MatExpr<E>& e) {
        return ScMatMulExpr<T, E>(s, e.self());
    };

    // Elementwise division
    template<typename E1, typename E2>
    struct MatDivExpr : MatExpr<MatDivExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        MatDivExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(e1.rows() == e2.rows() && e1.cols() == e2.cols()); };

        size_t rows() const { return e1.rows(); };
        size_t cols() const { return e1.cols(); };

        auto operator()(size_t i, size_t j) const { return e1(i, j) / e2(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    template<typename E1, typename E2>
    auto elementwise_multiply(const MatExpr<E1>& a, const MatExpr<E2>& b) {
        return MatMulExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto elementwise_divide(const MatExpr<E1>& a, const MatExpr<E2>& b) {
        return MatDivExpr<E1, E2>(a.self(), b.self());
    };

    // Scalar / matrix (element-wise)
    template<typename S, typename E> requires Scalar<S>
    struct ScMatDivLeft : MatExpr<ScMatDivLeft<S, E>> {
        S scalar;
        const E& expr;

        ScMatDivLeft(S s, const E& e) : scalar(s), expr(e) {};

        size_t rows() const { return expr.rows(); };
        size_t cols() const { return expr.cols(); };

        auto operator()(size_t i, size_t j) const { return scalar / expr(i, j); };

        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator/(S s, const MatExpr<E>& e) {
        return ScMatDivLeft<S, E>(s, e.self());
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator/(const MatExpr<E>& e, S s) {
        return ScMatMulExpr<decltype(S(1) / s), E>(S(1) / s, e.self());
    };

    // Unary expression for elementwise functions
    template<typename F, typename E>
    struct UnaryMatExpr : MatExpr<UnaryMatExpr<F, E>> {
        F func;
        const E& expr;

        UnaryMatExpr(F f, const E& e) : func(f), expr(e) {};

        size_t rows() const { return expr.rows(); };
        size_t cols() const { return expr.cols(); };

        auto operator()(size_t i, size_t j) const { return func(expr(i, j)); };

        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    // Upper-triangular extraction
    template<typename E>
    struct TriuExpr : MatExpr<TriuExpr<E>> {
        const E& expr; int k;

        TriuExpr(const E& e, int k_) : expr(e), k(k_) {};

        size_t rows() const { return expr.rows(); };
        size_t cols() const { return expr.cols(); };

        auto operator()(size_t i, size_t j) const {
            auto val = expr(i, j);
            using T = std::remove_cv_t<std::remove_reference_t<decltype(val)>>;
            return (static_cast<int>(j) - static_cast<int>(i) >= k) ? val : T(0);
        };
        bool depends_on(const void* p, size_t n) const { return expr.depends_on(p, n); };
    };
    
    // Lower-triangular extraction
    template<typename E>
    struct TrilExpr : MatExpr<TrilExpr<E>> {
        const E& expr; int k;

        TrilExpr(const E& e, int k_) : expr(e), k(k_) {};

        size_t rows() const { return expr.rows(); };
        size_t cols() const { return expr.cols(); };
        
        auto operator()(size_t i, size_t j) const {
            auto val = expr(i, j);
            using T = std::remove_cv_t<std::remove_reference_t<decltype(val)>>;
            return (static_cast<int>(j) - static_cast<int>(i) <= k) ? val : T(0);
        };
        bool depends_on(const void* p, size_t n) const { return expr.depends_on(p, n); };
    };

    // expr() wrapper
    template<typename T, Layout L>
    MatRef<T, L> expr(const Matrix<T, L>& mat) { return MatRef<T, L>(mat); };
 
    // const view -> read-only expression wrapper
    template<typename T, Layout L, bool Trans, bool Conj>
    MatViewExpr<T, L, Trans, Conj, false>
    expr(const MatrixView<T, L, Trans, Conj, false>& view) { return MatViewExpr<T, L, Trans, Conj, false>(view); };
 
    // Mutable view -> mutable expression wrapper (preserves write-back)
    template<typename T, Layout L, bool Trans, bool Conj>
    MatViewExpr<T, L, Trans, Conj, true>
    expr(MatrixView<T, L, Trans, Conj, true>& view) { return MatViewExpr<T, L, Trans, Conj, true>(view); };
 
    // const-ref overload for mutable views when only reading is needed
    template<typename T, Layout L, bool Trans, bool Conj>
    MatViewExpr<T, L, Trans, Conj, false>
    expr(const MatrixView<T, L, Trans, Conj, true>& view) {
        return MatViewExpr<T, L, Trans, Conj, false>(MatrixView<T, L, Trans, Conj, false>(view));
    };
};