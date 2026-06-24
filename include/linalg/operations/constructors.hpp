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

    /// @brief Frank matrix: `F(i, j) = n - max(i, j)` (1-indexed formula).
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

    /// @brief Redheffer matrix: `R(i, j) = 1` if `j == 1` or `i` divides `j` (1-indexed formula).
    /// @param n Matrix size.
    /// @return `n * n` Redheffer matrix.
    /// @note Determinant is given by the Mertens function of order `n`.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> redheffer(size_t n) {
        Matrix<T, L> R(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                R(i, 0) = T(1);
                for (size_t j = 1; j < n; ++j) if ((j + 1) % (i + 1) == 0) R(i, j) = T(1);
            };
        });
        return R;
    };

    /// @brief Hadamard matrix: mutually orthogonal rows; entries are either `+1` or `-1`.
    /// @param n Matrix size (required to be a power of 2).
    /// @return `n * n` symmetric Hadamard matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> hadamard(size_t n) {
        if (n == 0 || n & (n + 1) != 0) throw std::invalid_argument("hadamard: n must be a positve power of 2.");
        Matrix<T, L> H(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j)
                    H(i, j) = (std::popcount(static_cast<unsigned long long>(i & j)) & 1) ? T(-1) : T(1);
        });
        return H;
    };

    /// @brief Constructor: tridiagonal matrix.
    /// @param dl Sub-diagonal (length `n - 1`).
    /// @param d Main diagonal (length `n`).
    /// @param du Super-diagonal.
    /// @return `n * n` tridiagonal matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> tridiagonal(const Vector<T>& dl, const Vector<T>& d, const Vector<T>& du) {
        const size_t n = d.size();
        BOUNDS_CHECK(dl.size() == n - 1 && du.size() == n - 1);
        Matrix<T, L> A(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            A(i, i) = d[i];
            if (i > 0) A(i, i - 1) = dl[i - 1];
            if (i + 1 < n) A(i, i + 1) = du[i];
        };
        return A;
    };

    /// @brief Constructor: bidiagonal matrix.
    /// @param d Main diagonal (length `n`).
    /// @param e Off-diagonal entries (length `n - 1`).
    /// @param upper Upper/lower sub-diagonal flag.
    /// @return `n * n` bidiagonal matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> bidiagonal(const Vector<T>& d, const Vector<T>& e, bool upper = true) {
        const size_t n = d.size();
        BOUNDS_CHECK(e.size() == n - 1);
        Matrix<T, L> B(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            B(i, i) = d[i];
            if (i + 1 < n) {
                if (upper) B(i, i + 1) = e[i];
                else B(i + 1, i) = e[i];
            };
        };
        return B;
    };

    /// @brief Constructor: "arrowhead" matrix.
    /// @param d Diagonal entries (length `n - 1`).
    /// @param c Last column (length `n - 1`).
    /// @param r Last row (length `n - 1`).
    /// @param alpha Corner entry: `A(n, n) = alpha`.
    /// @return `n * n` matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> arrowhead(const Vector<T>& d, const Vector<T>& c, const Vector<T>& r, T alpha) {
        const size_t nm = d.size();
        BOUNDS_CHECK(c.size() == nm && r.size() == nm);
        const size_t n = nm + 1;
        Matrix<T, L> A(n, n, T(0));
        for (size_t i = 0; i < nm; ++i) {
            A(i, i) = d[i];
            A(i, nm) = c[i];
            A(nm, i) = r[i];
        };
        A(nm, nm) = alpha;
        return A;
    };

    /// @brief Moler matrix: `M(i, j) = min(i, j) - 2`, diagonal is `M(i, i) = i`(1-indexed).
    /// @param n Matrix size.
    /// @return `n * n` symmetric matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> moler(size_t n) {
        Matrix<T, L> M(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            M(i, i) = static_cast<T>(i + 1);
            for (size_t j = 0; j < n; ++j) {
                const double ip1 = static_cast<double>(i + 1);
                const double jp1 = static_cast<double>(j + 1);
                if (i != j) M(i, j) = static_cast<T>(std::min(ip1, jp1) - 2.0);
            };
        };
        return M;
    };

    /// @brief Grcar matrix: `G(i, i + j) = 1` for `j` in `{-1,0,1...k}`.
    /// @param n Matrix size.
    /// @param k Number of nonzero super-diagonals (default `k = 3`).
    /// @return `n * n` matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> grcar(size_t n, size_t k = 3) {
        Matrix<T, L> G(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            if (i > 0) G(i, i - 1) = T(-1); // Sub-diagonal.
            for (size_t j = i; j < n && j <= i + k; ++j) G(i, j) = T(1);
        };
        return G;
    };

    /// @brief Lotkin matrix: Hilbert matrix with first row made unit.
    /// @param n Matrix size.
    /// @return Symmetric `n * n` matrix.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> lotkin(size_t n) {
        auto A = hilbert<T, L>(n);
        for (size_t j = 0; j < n; ++j) A(0, j) = T(1);
        return A;
    };

    /// @brief Clement matrix: `C(i, i) = 0`, `C(i, i-1) = i-1`, `C(i, i+1) = i+1`.
    /// @param n Matrix size.
    /// @param symmetric When set to `true`, replaces the off-diagonals with their geometric means.
    /// @return `n * n` tridiagonal matrix.
    /// @note Singular for odd `n`.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> clement(size_t n, bool symmetric = false) {
        Matrix<T, L> C(n, n, T(0));
        for (size_t i = 0; i + 1 < n; ++i) {
            double lower = static_cast<double>(i + 1);
            double upper = static_cast<double>(n - i - 1);
            double v;
            if (symmetric) v = std::sqrt(lower * upper);
            else v = lower;
            C(i, i + 1) = static_cast<T>(v);
            C(i + 1, i) = static_cast<T>(v);
        };
        return C;
    };

    /// @brief Pei matrix: `P = alpha * I + ones * ones^T`.
    /// @param n Matrix size.
    /// @param alpha Scaling factor.
    /// @return `n * n` matrix.
    /// @note Eigenvalues are `alpha + n` (single) and `alpha`.
    template <typename T = double, Layout L = Layout::RowMajor>
    Matrix<T, L> pei(size_t n, T alpha = T(1)) {
        Matrix<T, L> P(n, n, T(1));
        for (size_t i = 0; i < n; ++i) P(i, i) += alpha;
        return P;
    };

    /// @brief Standard Vandermonde matrix: `V(i, j) = x[i]^j`.
    /// @param x Input vector (length `m`).
    /// @param n Number of columns (default `m`).
    /// @return `m * n` Vandermonde matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> vandermonde(const Vector<T>& x, size_t n = 0) {
        const size_t m = x.size();
        if (n == 0) n = m;
        BOUNDS_CHECK(n > 0 && m > 0);
        Matrix<T, L> V(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                T pk = T(1);
                for (size_t j = 0; j < n; ++j) {
                    V(i, j) = pk;
                    pk *= x[i];
                };
            };
        });
        return V;
    };

    /// @brief Generalised Vandermonde matrix: `V(i, j) = x[i]^{alpha[j]}`. 
    /// @param x Input vector (length `m`).
    /// @param alpha Exponent vector (length `n`).
    /// @return `m * n` matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> vandermonde_gen(const Vector<T>& x, const Vector<T>& alpha) {
        const size_t m = x.size(), n = alpha.size();
        BOUNDS_CHECK(m > 0 && n > 0);
        Matrix<T, L> V(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) V(i, j) = std::pow(x[i], alpha[j]);
        });
        return V;
    };

    /// @brief Rank-1 outer product matrix: `A(i, j) = x[i] * y[j]`.
    /// @param x Input vector (length `m`).
    /// @param y Input vector (length `n`).
    /// @return `m * n` matrix.
    /// @note Not to be confused with BLAS `ger` (which accumulates).
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> outer(const Vector<T>& x, const Vector<T>& y) {
        const size_t m = x.size(), n = y.size();
        BOUNDS_CHECK(m > 0 && n > 0);
        Matrix<T, L> A(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                const T xi = x[i];
                for (size_t j = 0; j < n; ++j) A(i, j) = xi * y[j];
            };
        });
        return A;
    };

    /// @brief Distance matrix: `D(i, j) = |x[i] - x[j]|`.
    /// @param x Input vector (length `n`).
    /// @return `n * n` symmetric distance matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> distance(const Vector<T>& x) {
        const size_t n = x.size();
        BOUNDS_CHECK(n > 0);
        Matrix<T, L> D(n, n, T(0));
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) D(i, j) = static_cast<T>(std::abs(x[i] - x[j]));
        });
        return D;
    };

    /// @brief Distance matrix: `D(i, j) = |x[i] - y[j]|`.
    /// @param x Input vector (length `m`).
    /// @param y Input vector (length `n`).
    /// @return `m * n` matrix.
    template <typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> distance(const Vector<T>& x, const Vector<T>& y) {
        const size_t m = x.size(), n = y.size();
        BOUNDS_CHECK(m > 0 && n > 0);
        Matrix<T, L> D(m, n, T(0));
        parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i)
                for (size_t j = 0; j < n; ++j) D(i, j) = static_cast<T>(std::abs(x[i] - y[j]));
        });
        return D;
    };
};