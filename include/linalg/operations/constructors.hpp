# pragma once

#include <linalg/operations/matrix_ops.hpp>

namespace linalg {
    namespace detail {
        /// @brief Combinatoric helper.
        /// @return Binomial coefficient C(n,k) as double.
        LINALG_INLINE double binom(size_t n, size_t k) noexcept {
            if (k > n) return 0.0;
            if (k == 0 || k == n) return 1.0;
            if (k > n - k) k = n - k;
            double res = 1.0;
            for (size_t i = 0; i < k; ++i) {
                res *= static_cast<double>(n - i);
                res /= static_cast<double>(i + 1);
            };
            return res;
        };
    };

    /// @brief Constructor: each consecutive row is a cyclic right-shift of the previous.
    /// @param v First row, given by a vector of length `n`.
    /// @return `n * n` square circulant matrix.
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> circulant(const Vector<T>& v) {
        const size_t n = v.size();
        Matrix<T, L> C(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) C(i, j) = v[(j + n - i) % n];
        });
        return C;
    };

    /// @brief Toeplitz matrix: constant along each diagonal.
    /// @param c Column generating vector (length `m`).
    /// @param r Row generating vector (length `n`).
    /// @return `m * n` Toeplitz matrix.
    /// @note First elements of `c` and `r` (the corner) must agree.
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> toeplitz(const Vector<T>& c, const Vector<T>& r) {
        const size_t m = c.size(), n = r.size();
        BOUNDS_CHECK(m > 0 && n > 0);
        if (c[0] != r[0]) throw std::invalid_argument("toeplitz: first element must be shared (c[0] = r[0]).");

        Matrix<T, L> A(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) A(i, j) = (i <= j) ? r[j - i] : c[i - j];
        });
        return A;
    };

    /// @brief Symmetric Toeplitz matrix.
    /// @param c Generating vector of length `n`.
    /// @return `n * n` Toeplitz matrix.
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> toeplitz(const Vector<T>& c) {
        return toeplitz<T, L>(c, c);
    };

    /// @brief Hankel matrix: constant along each anti-diagonal.
    /// @param c Column generating vector (length `m`).
    /// @param r Row generating vector (length `n`).
    /// @return `m * n` Hankel matrix.
    /// @note First element of `r` and last element of `c` must agree.
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> hankel(const Vector<T>& c, const Vector<T>& r) {
        const size_t m = c.size(), n = r.size();
        BOUNDS_CHECK(m > 0 && n > 0);
        if (c[m - 1] != r[0]) throw std::invalid_argument("hankel: first element of r must be equal to the last of c");

        Matrix<T, L> A(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const size_t idx = i + j;
                    A(i, j) = (idx < m) ? c[idx] : r[idx - m + 1];
                };
        });
        return A;
    };

    /// @brief Companion matrix of a monic polynomial (degree `n`).
    /// @param p Vector of coefficients `p[0...n-1]` of `p[0] + p[1]*x + ... + p[n-1]*x^{n-1} + x^n`.
    /// @return `n * n` structured compainion matrix.
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> companion(const Vector<T>& p) {
        const size_t n = p.size();
        BOUNDS_CHECK(n > 0);
        Matrix<T, L> C(n, n, T(0));
        for (size_t i = 1; i < n; ++i) C(i, i - 1) = T(1);
        // Last column: negated coefficients:
        for (size_t i = 0; i < n; ++i) C(i, n - 1) = -p[i];
        return C;
    };

    /// @brief Symmetric Pascal matrix: `P(i,j) = C(i+j, i)`.
    /// @param n Matrix size.
    /// @return `n * n` Pascal matrix.
    /// @note Unit determinant for all `n`.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> pascal(size_t n) {
        Matrix<T, L> P(n, n, T(0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) P(i, j) = static_cast<T>(detail::binom(i + j, i));
        return P;
    };
};