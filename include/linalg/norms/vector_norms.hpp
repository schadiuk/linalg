#pragma once

#include <linalg/blas/level1.hpp>

namespace linalg {
    /// @brief L0 pseudo-norm.
    /// @param x Vector expression.
    /// @return Number of non-zero entries.
    /// @note This construction fails absolute homogenity condition.
    template<typename E>
    double norm_l0(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        using T = std::remove_cvref_t<decltype(xx(0))>;
        return parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE,
            [&xx](size_t i) -> double { return (xx(i) != T{}) ? 1.0 : 0.0; });
    };

    /// @brief L1 norm.
    /// @param x Vector expression.
    /// @return Absolute value sum of the entries.
    template<typename E>
    double norm_l1(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        return parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE,
            [&xx](size_t i) -> double { return std::abs(xx(i)); });
    };

    /// @brief L2 norm
    /// @param x Vector expression.
    /// @return Square root of the sum of all elements' squares.
    template<typename E>
    double norm_l2(const VecExpr<E>& x) { return nrm2(x); };

    /// @brief Infinity norm.
    /// @param x Vector expression.
    /// @return Maximum absolute entry.
    template<typename E>
    double norm_inf(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        return parallel_reduce_assoc<double>(n, PARALLEL_THRESHOLD_REDUCE, 0.0,
            [&xx](size_t i) -> double { return std::abs(xx(i)); },
            [](double a, double b) { return std::max(a, b); });
    };

    /// @brief Negative infinity "norm".
    /// @param x Vector expression.
    /// @return Smallest absolute entry.
    template<typename E>
    double norm_neg_inf(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        return parallel_reduce_assoc<double>(n, PARALLEL_THRESHOLD_REDUCE, std::numeric_limits<double>::infinity(),
            [&xx](size_t i) -> double { return std::abs(xx(i)); },
            [](double a, double b) { return std::min(a, b); });
    };

    /// @brief Vector norm dispatch.
    /// @param x Vector expression.
    /// @param kind Supported norm kinds: `1`, `2` (or `fro` - default), `inf`, `-inf`.
    /// @return Specified norm.
    template<typename E>
    double norm(const VecExpr<E>& x, std::string kind = "2") {
        if (kind == "0") return norm_l0(x);
        else if (kind == "1") return norm_l1(x);
        else if (kind == "2" || kind == "fro") return norm_l2(x);
        else if (kind == "inf") return norm_inf(x);
        else if (kind == "-inf") return norm_neg_inf(x);
        else throw std::invalid_argument("Unrecognised norm kind: '" + kind + "'.");
    };
};