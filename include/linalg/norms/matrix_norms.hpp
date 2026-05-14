#pragma once

#include <linalg/storage/matrix.hpp>

namespace linalg {
    // L1 norm: maximum absolute column sum
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

    // Frobenius norm
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

    // Infinity norm: maximum absolute row sum
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

    // Negative infinity norm: minimum absolute row sum
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

    // Norm dispatch wrapper
    template<typename E>
    double norm(const MatExpr<E>& x, std::string kind = "fro") {
        if (kind == "1") return norm_l1(x);
        else if (kind == "fro") return norm_fro(x);
        else if (kind == "inf") return norm_inf(x);
        else if (kind == "-inf") return norm_neg_inf(x);
        else throw std::invalid_argument("Unrecognised norm kind: '" + kind + "'.");
    };
};