#pragma once

#include <linalg/storage/matrix.hpp>
#include <linalg/blas/level3.hpp>
#include <random>

namespace linalg {
    /// @brief L1 norm.
    /// @param x Matrix expression.
    /// @return Maximum absolute column sum.
    template<typename E>
    double norm_l1(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t rows = xx.rows(), cols = xx.cols();
        if (rows == 0 || cols == 0) return 0.0;
        const size_t threshold = std::max(size_t(1), PARALLEL_THRESHOLD_REDUCE / (rows + 1));
        return parallel_reduce_assoc<double>(cols, threshold, 0.0,
            [&xx, rows](size_t j) -> double {
                double cs = 0.0;
                for (size_t i = 0; i < rows; ++i) cs += std::abs(xx(i, j));
                return cs;
            },
            [](double a, double b) { return std::max(a, b); });
    };

    /// @brief Frobenius norm.
    /// @param x Matrix expression.
    /// @return Square root of the sum of all elements' squares.
    template<typename E>
    double norm_fro(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t rows = xx.rows(), cols = xx.cols();
        if (rows == 0 || cols == 0) return 0.0;
        using T = std::remove_cvref_t<decltype(xx(0,0))>;
        auto info = detail::raw_mat_info<T>(x);
        const bool usable = info.has_value() && !info->conj;
        Matrix<T, Layout::RowMajor> tmp;
        const T* ap; size_t lda; Layout layout;
        if (usable) { ap = info->data; lda = info->lda; layout = info->layout; }
        else { tmp = detail::materialise<T, Layout::RowMajor>(x); ap = tmp.data(); lda = tmp.stride(); layout = Layout::RowMajor; };

        auto at = [ap, lda, layout](size_t i, size_t j) -> T {
            return (layout == Layout::RowMajor) ? ap[i * lda + j] : ap[j * lda + i];
        };

        const size_t row_threshold = std::max(size_t(1), PARALLEL_THRESHOLD_REDUCE / (cols + 1));
        // Pass 1: parallel max.
        const double scale = parallel_reduce_assoc<double>(rows, row_threshold, 0.0,
            [at, cols](size_t i) -> double {
                double m = 0.0;
                for (size_t j = 0; j < cols; ++j) {
                    const double a = std::abs(at(i, j));
                    if (a > m) m = a;
                };
                return m;
            },
            [](double a, double b) { return std::max(a, b); });
        if (scale == 0.0) return 0.0;
        // Pass 2: parallel SUM(|A(i,j)|/scale)^2.
        const double inv_scale = 1.0 / scale;
        const double ssq = parallel_reduce<double>(rows, row_threshold,
            [at, cols, inv_scale](size_t i) -> double {
                double row_sq = 0.0;
                for (size_t j = 0; j < cols; ++j) {
                    const double a = std::abs(at(i, j)) * inv_scale;
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
        const size_t rows = xx.rows(), cols = xx.cols();
        if (rows == 0 || cols == 0) return 0.0;
        const size_t threshold = std::max(size_t(1), PARALLEL_THRESHOLD_REDUCE / (cols + 1));
        return parallel_reduce_assoc<double>(rows, threshold, 0.0,
            [&xx, cols](size_t i) -> double {
                double rs = 0.0;
                for (size_t j = 0; j < cols; ++j) rs += std::abs(xx(i, j));
                return rs;
            },
            [](double a, double b) { return std::max(a, b); });
    };

    /// @brief Negative infinity "norm" (after NumPy convention).
    /// @param x Matrix expression.
    /// @return Minimum absolute row sum.
    /// @note Whereas this fails the positive definedness condition, the "norm" is useful for detecting structural properties (e.g. singularity).
    template<typename E>
    double norm_neg_inf(const MatExpr<E>& x) {
        const auto& xx = x.self();
        const size_t rows = xx.rows(), cols = xx.cols();
        if (rows == 0 || cols == 0) return 0.0;
        const size_t threshold = std::max(size_t(1), PARALLEL_THRESHOLD_REDUCE / (cols + 1));
        return parallel_reduce_assoc<double>(rows, threshold, std::numeric_limits<double>::infinity(),
            [&xx, cols](size_t i) -> double {
                double rs = 0.0;
                for (size_t j = 0; j < cols; ++j) rs += std::abs(xx(i, j));
                return rs;
            },
            [](double a, double b) { return std::min(a, b); });
    };

    /// @brief L2 matrix norm.
    /// @param x Matrix expression.
    /// @param max_iter Iteration cap.
    /// @param tol Convergence tolerance.
    /// @return Largest singular value.
    /// @note Computed via power iteration.
    template<typename E>
    double norm_l2(const MatExpr<E>& x, int max_iter = 200, double tol = 1.49e-8) {
        const auto& xx = x.self();
        const size_t m = xx.rows(), n = xx.cols();
        if (m == 0 || n == 0) return 0.0;
        using T = std::remove_cvref_t<decltype(xx(0,0))>;

        Vector<T> v(n), w(m), vn(n), cw(m); // Uniform start avoids zero inner product with a dominant singular vector.
        const double inv_sqn = 1.0 / std::sqrt(static_cast<double>(n));
        for (size_t j = 0; j < n; ++j) v[j] = T(inv_sqn);
        // Deterministic, seeded RNG only exercised on a degenerate start.
        std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
        std::uniform_real_distribution<double> udist(-1.0, 1.0);
        bool retried = false;
        double sigma = 0.0;
        for (int it = 0; it < max_iter; ++it) {
            gemv(T(1), x, v, T(0), w);
            const double new_sigma = nrm2(w);
            if (new_sigma < 1e-300) {
                if (!retried) {
                    // Degenerate start: this particular fixed uniform vector happens to lie in (or near) A's null space.
                    for (size_t j = 0; j < n; ++j) v[j] = static_cast<T>(udist(rng));
                    const double vnorm = nrm2(v);
                    if (vnorm > 0.0) scal(T(1.0 / vnorm), v);
                    retried = true;
                    sigma = 0.0;
                    continue;
                };
                return 0.0;
            };
            const T inv_sw = T(1.0 / new_sigma);
            for (size_t i = 0; i < m; ++i) cw[i] = linalg::conj(w[i] * inv_sw);
            vgem(cw, x, vn);
            for (size_t j = 0; j < n; ++j) vn[j] = linalg::conj(vn[j]);
            const double norm_vn = nrm2(vn);
            if (norm_vn < 1e-300) return new_sigma;
            scal(T(1.0 / norm_vn), vn);
            v.swap(vn);
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