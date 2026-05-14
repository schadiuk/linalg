#pragma once

#include <linalg/blas/level1.hpp>

namespace linalg {
    // Pseudo-norm: count of non-zero elements
    template<typename E>
    double norm_l0(const VecExpr<E>& x) {
        const auto& xx = x.self();
        size_t count = 0;
        for (size_t i = 0; i < xx.size(); ++i) if (xx(i) != decltype(xx(i)){}) ++count;
        return static_cast<double>(count);
    };

    // L1 norm
    template<typename E>
    double norm_l1(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        if (n < PARALLEL_THRESHOLD_REDUCE) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += std::abs(xx(i));
            return s;
        };    
        auto& pool = ThreadPool::instance();
        const size_t nt = std::min(pool.thread_count(), (n + PARALLEL_THRESHOLD_REDUCE - 1) / PARALLEL_THRESHOLD_REDUCE);
        std::vector<Padded<double>> partials(nt);
        std::vector<std::future<void>> futures;
        size_t chunk = n / nt, rem = n % nt, offset = 0;
        for (size_t t = 0; t < nt; ++t) {
            const size_t cnt = chunk + (t < rem ? 1 : 0), start = offset;
            futures.push_back(pool.enqueue([&xx, &partials, start, cnt, t]() {
                double ps = 0.0;
                for (size_t i = 0; i < cnt; ++i) ps += std::abs(xx(start + i));
                partials[t].value = ps;
            }));
            offset += cnt;
        };
        for (auto& f : futures) f.get();
        double s = 0.0;
        for (const auto& p : partials) s += p.value;
        return s;
    };

    // L2 norm
    template<typename E>
    double norm_l2(const VecExpr<E>& x) { return nrm2(x); };

    // Infinity norm: max(abs(x_i))
    template<typename E>
    double norm_inf(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        double res = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = std::abs(xx(i));
            if (v > res) res = v;
        };
        return res;
    };

    // Negative infinity norm: min(abs(x_i))
    template<typename E>
    double norm_neg_inf(const VecExpr<E>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        double res = std::abs(xx(0));
        for (size_t i = 1; i < n; ++i) {
            const double v = std::abs(xx(i));
            if (v < res) res = v;
        };
        return res;
    };

    // Norm dispatch wrapper
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