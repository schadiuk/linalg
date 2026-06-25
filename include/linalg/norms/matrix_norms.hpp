#pragma once

#include <linalg/storage/matrix.hpp>
#include <linalg/blas/level3.hpp>

namespace linalg {
    /// @brief L1 norm.
    /// @param x Matrix expression.
    /// @return Maximum absolute column sum.
    template<typename E>
    double norm_l1(const MatExpr<E>& x) {
        const auto& xx = x.self();
        double res = 0.0;
        for (size_t j = 0; j < xx.cols(); ++j) {
            double cs = 0.0;
            for (size_t i = 0; i < xx.rows(); ++i) cs += std::abs(xx(i, j));
            if (cs > res) res = cs;
        };
        return res;
    };

    /// @brief Frobenius norm.
    /// @param x Matrix expression.
    /// @return Square root of the sum of all elements' squares.
    template<typename E>
    double norm_fro(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t rows = xx.rows(), cols = xx.cols();
        if (rows == 0 || cols == 0) return 0.0;
        // Pass 1: sequential max
        double scale = 0.0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                const double a = std::abs(xx(i, j));
                if (a > scale) scale = a;
            };
        if (scale == 0.0) return 0.0;
        // Pass 2, parallelised over rows
        const double inv_scale = 1.0 / scale;
        const size_t row_threshold = std::max(size_t(1), PARALLEL_THRESHOLD_REDUCE / (cols + 1));
        const double ssq = parallel_reduce<double>(rows, row_threshold,
            [&xx, cols, inv_scale](size_t i) -> double {
                double row_sq = 0.0;
                for (size_t j = 0; j < cols; ++j) {
                    const double a = std::abs(xx(i, j)) * inv_scale;
                    row_sq += a * a;
                };
                return row_sq;
            });
        return scale * std::sqrt(ssq);
    };

    /// @brief Infinity norm.
    /// @param x Matrix expression.
    /// @return Maximum absolute row sum.
    template<typename E>
    double norm_inf(const MatExpr<E>& x) {
        const auto& xx = x.self();
        double result = 0.0;
        for (size_t i = 0; i < xx.rows(); ++i) {
            double rs = 0.0;
            for (size_t j = 0; j < xx.cols(); ++j) rs += std::abs(xx(i, j));
            if (rs > result) result = rs;
        };
        return result;
    };

    /// @brief Negative infinity "norm" (after NumPy convention).
    /// @param x Matrix expression.
    /// @return Minimum absolute row sum.
    /// @note Whereas this fails the positive definedness condition, the "norm" is useful for detecting structural properties (e.g. singularity).
    template<typename E>
    double norm_neg_inf(const MatExpr<E>& x) {
        const auto& xx = x.self();
        if (xx.rows() == 0 || xx.cols() == 0) return 0.0;
        double result = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < xx.rows(); ++i) {
            double rs = 0.0;
            for (size_t j = 0; j < xx.cols(); ++j) rs += std::abs(xx(i, j));
            if (rs < result) result = rs;
        };
        return result;
    };

    /// @brief L2 matrix norm.
    /// @param x Matrix expression.
    /// @param max_iter Iteration cap.
    /// @param tol Convergence tolerance.
    /// @return Largest singular value.
    /// @note Computed via power iteration; materialises expression for optimal data access.
    template<typename E>
    double norm_l2(const MatExpr<E>& x, int max_iter = 200, double tol = 1.49e-8) {
        const auto& xx = x.self();
        const size_t m = xx.rows(), n = xx.cols();
        if (m == 0 || n == 0) return 0.0;
        using T = std::remove_cvref_t<decltype(x.self()(0,0))>;
        Matrix<T, Layout::RowMajor> A = detail::materialise<T, Layout::RowMajor>(x);
        const T* LINALG_RESTRICT ap = detail::assume_aligned<64>(A.data());
        const size_t lda = A.stride();
        // Uniform start avoids zero inner product with dominant singular vector.
        std::vector<T> v(n), w(m), vn(n);
        const double inv_sqn = 1.0 / std::sqrt(static_cast<double>(n));
        for (size_t j = 0; j < n; ++j) v[j] = T(inv_sqn);
        double sigma = 0.0;
        for (int it = 0; it < max_iter; ++it) {
            double ssq_w = 0.0;
            for (size_t i = 0; i < m; ++i) {
                T s = T(0);
                const T* LINALG_RESTRICT ai = ap + i * lda;
                LINALG_VECTORIZE
                for (size_t j = 0; j < n; ++j) s += ai[j] * v[j];
                w[i] = s;
                ssq_w += std::norm(s);
            };
            const double new_sigma = std::sqrt(ssq_w);
            if (new_sigma < 1e-300) return 0.0;
            // vn = A^H (w / new_sigma).
            const T inv_sw = T(1.0 / new_sigma);
            for (size_t j = 0; j < n; ++j) vn[j] = T(0);
            for (size_t i = 0; i < m; ++i) {
                const T wi = w[i] * inv_sw;
                const T* LINALG_RESTRICT ai = ap + i * lda;
                LINALG_VECTORIZE
                for (size_t j = 0; j < n; ++j) vn[j] += linalg::conj(ai[j]) * wi;
            };
            // Normalize vn -> next iterate v.
            double ssq_vn = 0.0;
            LINALG_VECTORIZE
            for (size_t j = 0; j < n; ++j) ssq_vn += std::norm(vn[j]);
            const double norm_vn = std::sqrt(ssq_vn);
            if (norm_vn < 1e-300) return new_sigma;
            const T inv_vn = T(1.0 / norm_vn);
            LINALG_VECTORIZE
            for (size_t j = 0; j < n; ++j) v[j] = vn[j] * inv_vn;

            if (it > 0 && std::abs(new_sigma - sigma) <= tol * new_sigma) return new_sigma;
            sigma = new_sigma;
        };
        return sigma;
    };


    /// @brief Matrix norm dispatch.
    /// @param x Matrix expression.
    /// @param kind Supported norm kinds: `fro` (default), `1`, `2`, `inf`, `-inf`.
    /// @return Specified norm.
    template<typename E>
    double norm(const MatExpr<E>& x, std::string kind = "fro") {
        if (kind == "1") return norm_l1(x);
        else if (kind == "2") return norm_l2(x);
        else if (kind == "fro") return norm_fro(x);
        else if (kind == "inf") return norm_inf(x);
        else if (kind == "-inf") return norm_neg_inf(x);
        else throw std::invalid_argument("Unrecognised norm kind: '" + kind + "'.");
    };
};