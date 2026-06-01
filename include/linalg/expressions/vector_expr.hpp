#pragma once

#include <linalg/expressions/expr_base.hpp>

namespace linalg {
    /// @brief Vector reference class. Alongside with VecExpr provides infrastructure for CRTP-based lazy evaluation. 
    /// @tparam T Scalar element type.
    template<typename T>
    struct VecRef : VecExpr<VecRef<T>> {
        /// @brief Internal object reference.
        const Vector<T>& vec;

        /// @brief Constructor from reference.
        /// @param v The reference to be assigned.
        explicit VecRef(const Vector<T>& v) : vec(v) {};

        size_t size() const { return vec.size(); };

        /// @brief Element indexation.
        /// @param i The index.
        /// @return Element given by `Vec[i]` if such exists, undefined otherwise
        LINALG_INLINE
        T operator()(size_t i) const { return vec[i]; };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return vec.depends_on(p, bytes); };
    };

    /// @brief Structure responsible for vector view support.
    /// @tparam T Scalar type.
    /// @tparam Mutable Mutability flag.
    template<typename T, bool Mutable = false> // Mutable = true preserves write-back capability when the wrapper is used as an assignment target
    struct VecViewRef : VecExpr<VecViewRef<T, Mutable>> {
        /// @brief Storage of the view.
        VectorView<T, Mutable>& view;

        /// @brief Constructor from a given view.
        /// @param v The view.
        explicit VecViewRef(const VectorView<T, Mutable>& v) : view(v) {};

        /// @brief Conversion operator.
        /// @note Implicit narrowing: a mutable wrapper can be used wherever a const one is expected.
        operator VecViewRef<T, false>() const requires Mutable {
            return VecViewRef<T, false>(static_cast<const VectorView<T, false>&>(view));
        };

        size_t size() const { return view.size(); };

        /// @brief Unchecked, read-only element access.
        /// @param i Index.
        /// @return Element given by `Vec[i]`.
        LINALG_INLINE
        T operator()(size_t i) const { return view(i); };
 
        /// @brief Write access.
        /// @param i Index.
        /// @return Element given by `Vec[i]`.
        /// @note Only synthesised for mutable wrappers.
        LINALG_INLINE
        T& operator()(size_t i) requires Mutable { return view(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return view.depends_on(p, bytes); };
    };

    /// @brief Vector addition expresion.
    /// @tparam E1 Left operand type (that of `e1`).
    /// @tparam E2 Right operand type (that of `e2`).
    template<typename E1, typename E2>
    struct VecAddExpr : VecExpr<VecAddExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        VecAddExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(a.size() == b.size()); };

        size_t size() const { return e1.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Sum at the given index: `e1(i) + e2(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return e1(i) + e2(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    /// @brief Vector subtraction expresion.
    /// @tparam E1 Left operand type (that of `e1`).
    /// @tparam E2 Right operand type (that of `e2`).
    template<typename E1, typename E2>
    struct VecSubExpr : VecExpr<VecSubExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        VecSubExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(a.size() == b.size()); };


        size_t size() const { return e1.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Difference at the given index: `e1(i) - e2(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return e1(i) - e2(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    /// @brief Expression for Hadamard product of vectors.
    /// @tparam E1 Left operand type (that of `e1`).
    /// @tparam E2 Right operand type (that of `e2`). 
    template<typename E1, typename E2>
    struct VecMulExpr : VecExpr<VecMulExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        VecMulExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(a.size() == b.size()); };

        size_t size() const { return e1.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Product at the given index: `e1(i) * e2(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return e1(i) * e2(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    /// @brief Expression for elementwise quotient of vectors.
    /// @tparam E1 Left operand type (that of `e1`).
    /// @tparam E2 Right operand type (that of `e2`).
    template<typename E1, typename E2>
    struct VecDivExpr : VecExpr<VecDivExpr<E1, E2>> {
        const E1& e1;
        const E2& e2;

        VecDivExpr(const E1& a, const E2& b) : e1(a), e2(b) { BOUNDS_CHECK(a.size() == b.size()); };

        size_t size() const { return e1.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Quotient at the given index: `e1(i) / e2(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return e1(i) / e2(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return e1.depends_on(p, bytes) || e2.depends_on(p, bytes); };
    };

    /// @brief Expression denoting vector multiplication by a scalar.
    /// @tparam S Scalar type.
    /// @tparam E Vector operand type.
    template<typename S, typename E>
    struct ScVecMulExpr : VecExpr<ScVecMulExpr<S, E>> {
        S scalar;
        const E& expr;

        ScVecMulExpr(S s, const E& e) : scalar(s), expr(e) {};

        size_t size() const { return expr.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Product at the given index: `scalar * expr(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return scalar * expr(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    /// @brief Expression denoting GEneral Matrix-Vector Multiplication.
    /// @tparam EM Matrix operand.
    /// @tparam EV Vector operand.
    /// @note Evaluates `Mat * Vec` product with vector as a column.
    template<typename EM, typename EV>
    struct GemvExpr : VecExpr<GemvExpr<EM, EV>> {
        const EM& mat;
        const EV& vec;

        GemvExpr(const EM& m, const EV& v) : mat(m), vec(v) { BOUNDS_CHECK(mat.cols() == v.size()); };

        size_t size() const { return mat.rows(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Product result at the given index. 
        LINALG_INLINE
        auto operator()(size_t i) const {
            using Type = decltype(mat(i, 0)* vec(0));
            Type sum = Type(0);
            for (size_t k = 0; k < mat.cols(); ++k) { sum += mat(i, k) * vec(k); };
            return sum;
        };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return mat.depends_on(p, bytes) || vec.depends_on(p, bytes); };
    };

    /// @brief Expression denoting elementwise scalar-vector division.
    /// @tparam S Scalar type.
    /// @tparam E Vector operand type.
    template<typename S, typename E> requires Scalar<S>
    struct ScVecDiv : VecExpr<ScVecDiv<S, E>> {
        S scalar;
        const E& expr;

        ScVecDiv(S s, const E& e) : scalar(s), expr(e) {};

        size_t size() const { return expr.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Product at the given index: `scalar / expr(i)`
        LINALG_INLINE
        auto operator()(size_t i) const { return scalar / expr(i); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator/(S s, const VecExpr<E>& e) {
        return ScVecDiv<S, E>(s, e.self());
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator/(const VecExpr<E>& e, S s) {
        return ScVecMulExpr<decltype(S(1) / s), E>(S(1) / s, e.self());
    };

    /// @brief Unary expression for applying elementwise operations/functions on vectors.
    /// @tparam F The `func` function type.
    /// @tparam E The vector expression type.
    template<typename F, typename E>
    struct UnaryVecExpr : VecExpr<UnaryVecExpr<F, E>> {
        F func;
        const E& expr;

        UnaryVecExpr(F f, const E& e) : func(f), expr(e) {};

        size_t size() const { return expr.size(); };

        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Result at the given index: `func(expr(i))`
        LINALG_INLINE
        auto operator()(size_t i) const { return func(expr(i)); };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return expr.depends_on(p, bytes); };
    };

    /// @brief Expression denoting vector-matrix multiplication.
    /// @tparam EM Matrix operand.
    /// @tparam EV Vector operand.
    /// @note Evaluates `Vec * Mat` product with vector as a row.
    template<typename EV, typename EM>
    struct VgemExpr : VecExpr<VgemExpr<EV, EM>> {
        const EV& vec;
        const EM& mat;

        VgemExpr(const EV& v, const EM& m) : vec(v), mat(m) { BOUNDS_CHECK(vec.size() == m.rows()); };

        size_t size() const { return mat.cols(); };
        
        /// @brief Element indexation of the expression.
        /// @param i Index.
        /// @return Product result at the given index. 
        LINALG_INLINE
        auto operator()(size_t j) const {
            using Type = decltype(vec(0)* mat(0, j));
            Type sum = Type(0);
            for (size_t k = 0; k < mat.rows(); ++k) { sum += vec(k) * mat(k, j); };
            return sum;
        };

        /// @brief Aliasing detection utility.
		/// @param p Wildcard pointer.
		/// @param bytes 
		/// @return Boolean indicator.
        bool depends_on(const void* p, size_t bytes) const { return vec.depends_on(p, bytes) || mat.depends_on(p, bytes); };
    };

    template<typename E1, typename E2>
    auto operator+(const VecExpr<E1>& a, const VecExpr<E2>& b) {
        return VecAddExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto operator-(const VecExpr<E1>& a, const VecExpr<E2>& b) {
        return VecSubExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto operator*(const VecExpr<E1>& a, const VecExpr<E2>& b) {
        return VecMulExpr<E1, E2>(a.self(), b.self());
    };

    template<typename E1, typename E2>
    auto operator/(const VecExpr<E1>& a, const VecExpr<E2>& b) {
        return VecDivExpr<E1, E2>(a.self(), b.self());
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator*(S s, const VecExpr<E>& e) {
        return ScVecMulExpr<S, E>(s, e.self());
    };

    template<typename S, typename E> requires Scalar<S>
    auto operator*(const VecExpr<E>& e, S s) {
        return ScVecMulExpr<S, E>(s, e.self());
    };
    // Vector operand as column
    template<typename EM, typename EV>
    auto operator*(const MatExpr<EM>& m, const VecExpr<EV>& v) {
        return GemvExpr<EM, EV>(m.self(), v.self());
    };
    // Vector operand as row
    template<typename EV, typename EM>
    auto operator*(const VecExpr<EV>& v, const MatExpr<EM>& m) {
        return VgemExpr<EV, EM>(v.self(), m.self());
    };

    // expr() wrapper
    template<typename T>
    VecRef<T> expr(const Vector<T>& vec) { return VecRef<T>(vec); };
 
    // const view -> read-only expression wrapper
    template<typename T>
    VecViewRef<T, false> expr(const VectorView<T, false>& view) { return VecViewRef<T, false>(view); };
 
    // Mutable view -> mutable expression wrapper (preserves write-back)
    template<typename T>
    VecViewRef<T, true> expr(VectorView<T, true>& view) { return VecViewRef<T, true>(view); };
 
    // const-ref overload for mutable views when only reading is needed
    template<typename T>
    VecViewRef<T, false> expr(const VectorView<T, true>& view) {
        return VecViewRef<T, false>(static_cast<const VectorView<T, false>&>(view));
    };
};