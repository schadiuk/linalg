#pragma once

#include <linalg/expressions/matrix_expr.hpp>
#include <linalg/expressions/vector_expr.hpp>

namespace linalg {
    // Applies a unary fuction elementwise
    template<typename F, typename E>
    auto apply(F func, const MatExpr<E>& e) {
        return UnaryMatExpr<F, E>(func, e.self());
    };

    template<typename E>
    auto abs(const MatExpr<E>& e) {
        return apply([](auto x) { return std::abs(x); }, e);
    };

    template<typename E>
    auto sqrt(const MatExpr<E>& e) {
        return apply([](auto x) { return std::sqrt(x); }, e);
    };

    template<typename E>
    auto exp(const MatExpr<E>& e) {
        return apply([](auto x) { return std::exp(x); }, e);
    };

    template<typename E>
    auto log(const MatExpr<E>& e) {
        return apply([](auto x) { return std::log(x); }, e);
    };

    template<typename E>
    auto sin(const MatExpr<E>& e) {
        return apply([](auto x) { return std::sin(x); }, e);
    };

    template<typename E>
    auto asin(const MatExpr<E>& e) {
        return apply([](auto x) { return std::asin(x); }, e);
    };

    template<typename E>
    auto cos(const MatExpr<E>& e) {
        return apply([](auto x) { return std::cos(x); }, e);
    };

    template<typename E>
    auto acos(const MatExpr<E>& e) {
        return apply([](auto x) { return std::acos(x); }, e);
    };

    template<typename E>
    auto tan(const MatExpr<E>& e) {
        return apply([](auto x) { return std::tan(x); }, e);
    };

    template<typename E>
    auto atan(const MatExpr<E>& e) {
        return apply([](auto x) { return std::atan(x); }, e);
    };

    template<typename E, typename S>
    auto pow(const MatExpr<E>& e, S p) { return apply([p](auto x){ return std::pow(x, p); }, e); };

    // Sum reduction
    template<typename E>
    auto sum(const MatExpr<E>& x) {
        const auto& xx = x.self();
        using Type = std::remove_cvref_t<decltype(xx(0, 0))>;
        const size_t rows = xx.rows();
        const size_t cols = xx.cols();
        const size_t total = rows * cols;
        if (total == 0) return Type(0);
        if (total < PARALLEL_THRESHOLD_REDUCE) {
            Type s = Type(0);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    s += xx(i, j);
                };
            };
            return s;
        };

        auto& pool = ThreadPool::instance();
        size_t num_threads = std::min(pool.thread_count(), (total + PARALLEL_THRESHOLD_REDUCE - 1) / PARALLEL_THRESHOLD_REDUCE);

        std::vector<Padded<Type>> partials(num_threads);
        std::vector<std::future<void>> futures;

        size_t chunk = total / num_threads;
        size_t remainder = total % num_threads;
        size_t offset = 0;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t count = chunk + (t < remainder ? 1 : 0);
            size_t start = offset;
            futures.push_back(pool.enqueue([&xx, rows, cols, partials_ptr = partials.data(), start, count, t]() {
                Type ps = Type(0);
                for (size_t idx = start; idx < start + count; ++idx) {
                    size_t i = idx / cols;
                    size_t j = idx % cols;
                    ps += xx(i, j);
                };
                partials_ptr[t].value = ps;
                }));
            offset += count;
        };

        for (auto& f : futures) { f.get(); }
        Type result = Type(0);
        for (const auto& p : partials) { result += p.value; }
        return result;
    };

    // Mean reduction
    template<typename E>
    auto mean(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t total = xx.rows() * xx.cols();
        auto s = sum(x);
        return s / static_cast<decltype(s)>(total);
    };

    // Variance reduction (no Bessel's correction applied)
    template<typename E>
    double variance(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t total = xx.rows() * xx.cols();
        BOUNDS_CHECK(total > 0);
        auto m = mean(x);
        double var = 0.0;
        for (size_t i = 0; i < xx.rows(); ++i)
            for (size_t j = 0; j < xx.cols(); ++j) {
                auto diff = xx(i,j) - m;
                var += std::norm(diff);
            };
        return var / static_cast<double>(total);
    };

    // Standard deviation (no Bessel's correction applied)
    template<typename E> double stddev(const MatExpr<E>& x) { return std::sqrt(variance(x)); };

    template<typename E> auto floor(const MatExpr<E>& e) { return apply([](auto x){ return std::floor(x); }, e); };

    template<typename E> auto ceil(const MatExpr<E>& e)  { return apply([](auto x){ return std::ceil(x);  }, e); };

    template<typename E> auto round(const MatExpr<E>& e) { return apply([](auto x){ return std::round(x); }, e); };

    template<typename E> auto real(const MatExpr<E>& e) { return apply([](auto x){ return std::real(x); }, e); };

    template<typename E> auto imag(const MatExpr<E>& e) { return apply([](auto x){ return std::imag(x); }, e); };

    template<typename E> auto conj(const MatExpr<E>& e) { return apply([](auto x){ return linalg::conj(x); }, e); };

    template<typename E>
    auto trace(const MatExpr<E>& x) {
        const auto& xx = x.self();
        BOUNDS_CHECK(xx.rows() == xx.cols());
        // std::remove_cvref_t: operator() may return const T& for raw Matrix; strip the ref
        using Type = std::remove_cvref_t<decltype(xx(0, 0))>;
        Type tr = Type(0);
        for (size_t i = 0; i < xx.rows(); ++i) { tr += xx(i, i); };
        return tr;
    };

    // Vector -> diagonal matrix
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> diag(const Vector<T>& v, int k = 0) {
        const size_t n = v.size();
        const size_t abs_k = static_cast<size_t>(std::abs(k));
        const size_t dim = n + abs_k;
        Matrix<T, L> result = Matrix<T, L>::zeros(dim, dim);
        if (k >= 0) {
            for (size_t i = 0; i < n; ++i) { result(i, i + abs_k) = v[i]; };
        } else {
            for (size_t i = 0; i < n; ++i) { result(i + abs_k, i) = v[i]; };
        };
        return result;
    };

    // VecExpr -> diagonal matrix
    template<typename E, Layout L = Layout::RowMajor>
    auto diag(const VecExpr<E>& v_expr, int k = 0) {
        const auto& v = v_expr.self();
        const size_t n = v.size();
        const size_t abs_k = static_cast<size_t>(std::abs(k));
        const size_t dim = n + abs_k;
        // std::remove_cvref_t: operator() on VecRef returns T by value, but on raw Vector it returns const T&
        using T = std::remove_cvref_t<decltype(v(0))>;
        Matrix<T, L> result = Matrix<T, L>::zeros(dim, dim);
        if (k >= 0) {
            for (size_t i = 0; i < n; ++i) { result(i, i + abs_k) = v(i); };
        } else {
            for (size_t i = 0; i < n; ++i) { result(i + abs_k, i) = v(i); };
        };
        return result;
    };

    // Matrix -> diagonal vector
    template<typename T, Layout L>
    Vector<T> diag(const Matrix<T, L>& mat, int k = 0) {
        const size_t rows = mat.rows();
        const size_t cols = mat.cols();
        // Guard against unsigned underflow before performing size_t arithmetic
        size_t diag_length = 0;
        if (k >= 0) {
            const size_t sk = static_cast<size_t>(k);
            if (sk < cols) diag_length = std::min(rows, cols - sk);
        } else {
            const size_t sk = static_cast<size_t>(-k);
            if (sk < rows) diag_length = std::min(rows - sk, cols);
        };
        if (diag_length == 0) return Vector<T>(0);
        Vector<T> result(diag_length);
        if (k >= 0) {
            const size_t sk = static_cast<size_t>(k);
            for (size_t i = 0; i < diag_length; ++i) { result[i] = mat(i, i + sk); };
        } else {
            const size_t sk = static_cast<size_t>(-k);
            for (size_t i = 0; i < diag_length; ++i) { result[i] = mat(i + sk, i); };
        };
        return result;
    };

    // MatExpr -> diagonal vector
    template<typename E>
    auto diag(const MatExpr<E>& mat_expr, int k = 0) {
        const auto& mat = mat_expr.self();
        const size_t rows = mat.rows();
        const size_t cols = mat.cols();

        size_t diag_length = 0;
        if (k >= 0) {
            const size_t sk = static_cast<size_t>(k);
            if (sk < cols) diag_length = std::min(rows, cols - sk);
        } else {
            const size_t sk = static_cast<size_t>(-k);
            if (sk < rows) diag_length = std::min(rows - sk, cols);
        };
        // std::remove_cvref_t: operator() may return const T& for raw Matrix expressions
        using T = std::remove_cvref_t<decltype(mat(0, 0))>;
        if (diag_length == 0) return Vector<T>(0);
        Vector<T> result(diag_length);
        if (k >= 0) {
            const size_t sk = static_cast<size_t>(k);
            for (size_t i = 0; i < diag_length; ++i) { result[i] = mat(i, i + sk); };
        } else {
            const size_t sk = static_cast<size_t>(-k);
            for (size_t i = 0; i < diag_length; ++i) { result[i] = mat(i + sk, i); };
        };
        return result;
    };

    // Extract upper triangular matrix
    template<typename E> auto triu(const MatExpr<E>& e, int k = 0) { return TriuExpr<E>(e.self(), k); };

    // Extract lower triangular matrix
    template<typename E> auto tril(const MatExpr<E>& e, int k = 0) { return TrilExpr<E>(e.self(), k); };

    // Collapse Matrix into a Vector
    template<typename E>
    auto flatten(const MatExpr<E>& x) {
        const auto& xx = x.self();
        using T = std::remove_cvref_t<decltype(xx(0,0))>;
        const size_t total = xx.rows() * xx.cols();
        Vector<T> result(total);
        size_t idx = 0;
        for (size_t i = 0; i < xx.rows(); ++i)
            for (size_t j = 0; j < xx.cols(); ++j)
                result[idx++] = xx(i, j);
        return result;
    };
};