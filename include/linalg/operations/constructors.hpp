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

    /// @brief Upper triangular Kahan matrix: `K(i,j) = -s^i * c^(j-i)` for `j > i`. 
    /// @param n Matrix size.
    /// @param theta Angle parameter: `s = sin(theta), c = cos(theta)`.
    /// @param perturbation Diagonal offset: `K(i,i) = s^i + perturbation`.
    /// @return `n * n` Kahan matrix.
    /// @note Non-zero diagonal perturbation can break exact singularity.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> kahan(size_t n, double theta = std::numbers::pi / 2, double perturbation = 0.0) {
        const double s = std::sin(theta), c = std::cos(theta);
        Matrix<T, L> K(n, n, T(0));
        std::vector<double> sp(n);
        double si = 1.0;
        for (size_t i = 0; i < n; ++i) { sp[i] = si; si *= s; };
        for (size_t i = 0; i < n; ++i) {
            K(i, i) = static_cast<T>(sp[i] + perturbation);
            double cj = c;
            for (size_t j = i + 1; j < n; ++j) {
                K(i, j) = static_cast<T>(-sp[i] * cj);
                cj *= c;
            };
        };
        return K;
    };

    /// @brief Hilbert matrix: `H(i,j) = 1 / (i + j + 1)`.
    /// @param n Matrix size.
    /// @return `n * n` symmetric Hilbert matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> hilbert(size_t n) {
        Matrix<T, L> H(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) H(i, j) = T(1) / static_cast<T>(i + j + 1);
        });
        return H;
    };

    /// @brief Lehmer matrix: `A(i, j) = min(i, j) / max(i, j)` (1-indexed formula).
    /// @param n Matrix size.
    /// @return `n * n` symmetric Lehmer matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> lehmer(size_t n) {
        Matrix<T, L> A(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i ) {
                const double ip1 = static_cast<double>(i + 1);
                for (size_t j = 0; j < n; ++j) {
                    const double jp1 = static_cast<double>(j + 1);
                    A(i, j) = static_cast<T>(std::min(ip1, jp1) / std::max(ip1, jp1));
                };
            };
        });
        return A;
    };

    /// @brief Wilkinson matrix.
    /// @param n Matrix size.
    /// @return `n * n` tridiagonal matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> wilkinson(size_t n) {
        Matrix<T, L> W(n, n, T(0));
        const size_t half = n / 2;
        for (size_t i = 0; i < n; ++i) {
            W(i, i) = static_cast<T>(i < half ? half - i : i - half);
            if (i + 1 < n) { W(i, i + 1) = T(1); W(i + 1, i) = T(1); };
        };
        return W;
    };

    /// @brief Frank matrix: `F(i,j) = n - max(i,j)` (1-indexed formula).
    /// @param n Matrix size.
    /// @return `n * n` Frank matrix.
    /// @note Unit determinant for all `n`.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> frank(size_t n) {
        Matrix<T, L> F(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for(size_t i = rs; i< re; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const size_t m = std::max(i, j);
                    if (n > m) F(i, j) = static_cast<T>(n - m);
                };
        });
        return F;
    };
};