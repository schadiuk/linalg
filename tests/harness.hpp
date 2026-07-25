#pragma once

// Self-contained test harness: EXPECT()/EXPECT_NEAR() with file/line reports.

#include <include/linalg.hpp>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
 
namespace test {
    inline int g_failures = 0;
    inline int g_checks = 0;
    inline std::string g_current_test;

    inline void report_fail(const std::string& expr, const char* file, int line) {
        ++g_failures;
        std::cerr << "[FAIL] " << g_current_test << " : " << expr << "  (" << file << ":" << line << ")\n";
    };

    #define EXPECT(cond) do { \
        ::lgtest::g_checks++; \
        if (!(cond)) ::lgtest::report_fail(#cond, __FILE__, __LINE__); \
    } while(0)

    #define EXPECT_NEAR(a, b, tol) do { \
        ::lgtest::g_checks++; \
        auto _a = (a); auto _b = (b); auto _t = (tol); \
        double _d = std::abs(_a - _b); \
        if (!(_d <= _t)) { \
            std::ostringstream _ss; \
            _ss << #a " ~= " #b " (|" << _d << "| > " << _t << ")"; \
            ::lgtest::report_fail(_ss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

    #define RUN_TEST(fn) do { \
        ::lgtest::g_current_test = #fn; \
        fn(); \
    } while(0)

    // Seeded XorShift64 RNG.
    struct XorShift64 {
        uint64_t state;
        explicit XorShift64(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {};
        uint64_t next() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            return state;
        };
        double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); };
        double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); };
    };

    // Typed tolerances.
    template<typename T> struct tol_traits;

    template<> struct tol_traits<float> { static double value() { return 1e-3; }; };
    template<> struct tol_traits<double> { static double value() { return 1e-9; }; };
    template<> struct tol_traits<std::complex<float>> { static double value() { return 1e-3; }; };
    template<> struct tol_traits<std::complex<double>> { static double value() { return 1e-9; }; };
    template<typename T> double tol() { return tol_traits<T>::value(); };
 
    // Random fill helpers (structurally independent of library RNG).
    template<typename T>
    T rand_scalar(XorShift64& rng) {
        if constexpr (linalg::detail::is_complex_v<T>) {
            using R = typename T::value_type;
            return T(static_cast<R>(rng.uniform(-1.0, 1.0)), static_cast<R>(rng.uniform(-1.0, 1.0)));
        } else {
            return static_cast<T>(rng.uniform(-1.0, 1.0));
        };
    };

    template<typename T, linalg::Layout L>
    linalg::Matrix<T, L> random_matrix(size_t m, size_t n, XorShift64& rng) {
        linalg::Matrix<T, L> A(m, n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) A(i, j) = rand_scalar<T>(rng);
        return A;
    };

    template<typename T>
    linalg::Vector<T> random_vector(size_t n, XorShift64& rng) {
        linalg::Vector<T> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = rand_scalar<T>(rng);
        return v;
    };

    // Random HPD matrix: `A = B * B^H + n * I` (guarantees positive-definiteness).
    template<typename T, linalg::Layout L>
    linalg::Matrix<T, L> random_hpd(size_t n, XorShift64& rng) {
        auto B = random_matrix<T, L>(n, n, rng);
        linalg::Matrix<T, L> A(n, n, T(0));
        linalg::herk('L', 'N', typename linalg::detail::real_type_t<T>(1), linalg::expr(B), typename linalg::detail::real_type_t<T>(0), A);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) A(i, j) = linalg::conj(A(j, i));
        for (size_t i = 0; i < n; ++i) A(i, i) += T(static_cast<double>(n));
        return A;
    };

    // Frobenius-norm-based residual check.
    template<typename T, linalg::Layout L>
    bool residual_small(const linalg::Matrix<T, L>& A, const linalg::Matrix<T, L>& B, double tol_) {
        linalg::Matrix<T, L> D(A.rows(), A.cols());
        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < A.cols(); ++j) D(i, j) = A(i, j) - B(i, j);
        double nd = linalg::norm_fro(linalg::expr(D));
        double na = linalg::norm_fro(linalg::expr(A));
        return nd <= tol_ * std::max(1.0, na);
    };

    // Orthonormality check: columns of `Q` satisfy `Q^H * Q == I`.
    template<typename T, linalg::Layout L>
    bool is_orthonormal_cols(const linalg::Matrix<T, L>& Q, double tol_) {
        const size_t k = Q.cols();
        linalg::Matrix<T, L> G(k, k, T(0));
        linalg::gemm(T(1), linalg::hermitian(Q), linalg::expr(Q), T(0), G);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j) {
                T expect = (i == j) ? T(1) : T(0);
                if (std::abs(G(i, j) - expect) > tol_) return false;
            };
        return true;
    };

    // Finiteness checks: no NaN/Inf leaks.
    template<typename T>
    bool all_finite(const T& x) {
        if constexpr (linalg::detail::is_complex_v<T>) return std::isfinite(x.real()) && std::isfinite(x.imag());
        else return std::isfinite(static_cast<double>(x));
    };

    template<typename T>
    bool all_finite(const linalg::Vector<T>& v) {
        for (size_t i = 0; i < v.size(); ++i) if (!all_finite(v[i])) return false;
        return true;
    };

    template<typename T, linalg::Layout L>
    bool all_finite(const linalg::Matrix<T, L>& A) {
        for (size_t i = 0; i < A.rows(); ++i)
            for (size_t j = 0; j < A.cols(); ++j) if (!all_finite(A(i, j))) return false;
        return true;
    };

    // Spectral condition number via SVD.
    template<typename T, linalg::Layout L>
    double cond2(const linalg::Matrix<T, L>& A) {
        auto res = linalg::svd(A);
        const size_t k = res.s.size();
        if (k == 0) return 0.0;
        double smax = res.s[0], smin = res.s[k - 1];
        if (smin <= 0.0) return std::numeric_limits<double>::infinity();
        return smax / smin;
    };
};