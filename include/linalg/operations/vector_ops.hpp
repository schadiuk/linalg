# pragma once

#include <linalg/storage/vector.hpp>

namespace linalg {
    // Creates a Vector with num uniformly spaced values between start and stop, inclusive if endpoint is true
    template<typename T = double>
    Vector<T> linspace(T start, T stop, size_t num = 50, bool endpoint = true) {
        if (num == 0) return Vector<T>();
        if (num == 1) {
            Vector<T> res(1);
            res[0] = start;
            return res;
        };
        const double denom = endpoint ? static_cast<double>(num - 1)
                                      : static_cast<double>(num);
        const double step  = (static_cast<double>(stop) - static_cast<double>(start)) / denom;
        Vector<T> res(num);
        parallel_for(num, PARALLEL_THRESHOLD_SIMPLE, [start, step, &res](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i)
                res[i] = static_cast<T>(static_cast<double>(start) + static_cast<double>(i) * step);
        });
        if (endpoint) res[num - 1] = stop;
        return res;
    };
 
    // Creates a Vector with values beginning at start and incremented by step until reaching stop (exclusive)
    template<typename T = int>
    Vector<T> arange(T start, T stop, T step = T(1)) {
        if (step == T(0)) throw std::invalid_argument("arange: step cannot be zero");
        if ((step > T(0) && start >= stop) ||
            (step < T(0) && start <= stop)) return Vector<T>();
 
        size_t num;
        if constexpr (std::is_integral_v<T>) {
            T diff = stop - start;
            T q = diff / step;
            T r = diff % step;
            num = static_cast<size_t>(q) + (r != T(0) ? 1 : 0);
        } else {
            double raw = std::ceil(
                (static_cast<double>(stop) - static_cast<double>(start))
                / static_cast<double>(step));
            num = (raw <= 0.0) ? 0 : static_cast<size_t>(raw);
        };
 
        Vector<T> res(num);
        parallel_for(num, PARALLEL_THRESHOLD_SIMPLE,
            [start, step, &res](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i)
                    res[i] = start + static_cast<T>(static_cast<double>(i) * static_cast<double>(step));
            });
        if constexpr (!std::is_integral_v<T>) {
            if (num > 0) {
                const T last = res[num - 1];
                if ((step > T(0) && last >= stop) ||
                    (step < T(0) && last <= stop)) {
                    Vector<T> trimmed(num - 1);
                    for (size_t i = 0; i < num - 1; ++i) trimmed[i] = res[i];
                    return trimmed;
                };
            };
        };
        return res;
    };

    // Convenience wrapper: arange(stop) -> [0, stop)
    template<typename T = int>
    Vector<T> arange(T stop) { return arange<T>(T(0), stop, T(1)); };

    // Applies a unary fuction elementwise
    template<typename F, typename E>
    auto apply(F func, const VecExpr<E>& e) {
        return UnaryVecExpr<F, E>(func, e.self());
    };

    template<typename E>
    auto abs(const VecExpr<E>& e) {
        return apply([](auto x) { return std::abs(x); }, e);
    };

    template<typename E>
    auto sqrt(const VecExpr<E>& e) {
        return apply([](auto x) { return std::sqrt(x); }, e);
    };

    template<typename E>
    auto exp(const VecExpr<E>& e) {
        return apply([](auto x) { return std::exp(x); }, e);
    };

    template<typename E>
    auto log(const VecExpr<E>& e) {
        return apply([](auto x) { return std::log(x); }, e);
    };

    template<typename E>
    auto sin(const VecExpr<E>& e) {
        return apply([](auto x) { return std::sin(x); }, e);
    };

    template<typename E>
    auto asin(const VecExpr<E>& e) {
        return apply([](auto x) { return std::asin(x); }, e);
    };

    template<typename E>
    auto cos(const VecExpr<E>& e) {
        return apply([](auto x) { return std::cos(x); }, e);
    };

    template<typename E>
    auto acos(const VecExpr<E>& e) {
        return apply([](auto x) { return std::acos(x); }, e);
    };

    template<typename E>
    auto tan(const VecExpr<E>& e) {
        return apply([](auto x) { return std::tan(x); }, e);
    };

    template<typename E>
    auto atan(const VecExpr<E>& e) {
        return apply([](auto x) { return std::atan(x); }, e);
    };

    template<typename E>
    auto sinh(const VecExpr<E>& e) {
        return apply([](auto x) { return std::sinh(x); }, e);
    };

    template<typename E>
    auto cosh(const VecExpr<E>& e) {
        return apply([](auto x) { return std::cosh(x); }, e);
    };

    template<typename E>
    auto tanh(const VecExpr<E>& e) {
        return apply([](auto x) { return std::tanh(x); }, e);
    };

    template<typename E, typename R>
    auto pow(const VecExpr<E>& e, R p) {
        return apply([p](auto x) { return std::pow(x, p); }, e);
    };

    template<typename E> auto floor(const VecExpr<E>& e) { return apply([](auto x){ return std::floor(x); }, e); };

    template<typename E> auto ceil(const VecExpr<E>& e)  { return apply([](auto x){ return std::ceil(x);  }, e); };

    template<typename E> auto round(const VecExpr<E>& e) { return apply([](auto x){ return std::round(x); }, e); };
    
    template<typename E> auto real(const VecExpr<E>& e) { return apply([](auto x){ return std::real(x); }, e); };
    
    template<typename E> auto imag(const VecExpr<E>& e) { return apply([](auto x){ return std::imag(x); }, e); };

    template<typename E> auto conj(const VecExpr<E>& e) { return apply([](auto x){ return linalg::conj(x); }, e); };

    // Sum reduction
    template<typename E>
    auto sum(const VecExpr<E>& x) {
        const auto& xx = x.self();
        // std::remove_cvref_t: operator() on raw Vector returns const T&; stripping the reference gives the plain value type for accumulators
        using Type = std::remove_cv_t<std::remove_reference_t<decltype(xx(0))>>;
        const size_t total = xx.size();
        if (total == 0) return Type(0);
        if (total < PARALLEL_THRESHOLD_REDUCE) {
            Type s = Type(0);
            for (size_t i = 0; i < total; ++i) { s += xx(i); };
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
            futures.push_back(pool.enqueue([&xx, partials_ptr = partials.data(), start, count, t]() {
                Type ps = Type(0);
                for (size_t i = 0; i < count; ++i) {
                    ps += xx(start + i);
                };
                partials_ptr[t].value = ps;
                }));
            offset += count;
        };

        for (auto& f : futures) { f.get(); };
        Type total_sum = Type(0);
        for (const auto& p : partials) { total_sum += p.value; };
        return total_sum;
    };

    // Mean reduction
    template<typename E>
    auto mean(const VecExpr<E>& x) {
        auto s = sum(x);
        return s / static_cast<decltype(s)>(x.self().size());
    };
 
    // Variance reduction (no Bessel's correction applied)
    template<typename E>
    double variance(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        BOUNDS_CHECK(n > 0);
        auto m = mean(x);
        double var = 0.0;
        for (size_t i = 0; i < n; ++i) {
            auto diff = xx(i) - m;
            var += std::norm(diff);
        };
        return var / static_cast<double>(n);
    };

    // Standard deviation (no Bessel's correction applied)
    template<typename E> double stddev(const VecExpr<E>& x) { return std::sqrt(variance(x)); };
};