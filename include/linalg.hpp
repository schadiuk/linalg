#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>

#include <linalg/storage/vector.hpp>
#include <linalg/storage/matrix.hpp>
#include <linalg/storage/vector_view.hpp>
#include <linalg/storage/matrix_view.hpp>

#include <linalg/expressions/expr_base.hpp>
#include <linalg/expressions/vector_expr.hpp>
#include <linalg/expressions/matrix_expr.hpp>

#include <linalg/operations/vector_ops.hpp>
#include <linalg/operations/matrix_ops.hpp>

#include <linalg/blas/level1.hpp>
#include <linalg/blas/level2.hpp>
#include <linalg/blas/level3.hpp>

#include <string>
#include <sstream>
#include <iomanip>

namespace linalg {
    // Formatter
    struct IOFormat {
        int precision = 3;
        bool scientific = false;
        bool complex_as_pair = true; // False: a + bi, true: (a,b)
    };

    template<typename T>
    std::string format_scalar(const T& x, const IOFormat& fmt) {
        std::ostringstream ss;
        if (fmt.scientific) ss << std::scientific;
        else ss << std::fixed;
        ss << std::setprecision(fmt.precision);
        ss << x;
        return ss.str();
    };

    template<typename T>
    std::string format_scalar(const std::complex<T>& z, const IOFormat& fmt) {
        std::ostringstream ss;
        if (fmt.scientific) ss << std::scientific;
        else ss << std::fixed;
        ss << std::setprecision(fmt.precision);
        if (fmt.complex_as_pair) {
            ss << "(" << z.real() << "," << z.imag() << ")";
        } else {
            ss << z.real();
            if (z.imag() >= 0) ss << " + " << z.imag() << "i";
            else ss << " - " << std::abs(z.imag()) << "i";
        };
        return ss.str();
    };

    template<typename TT>
    std::ostream& print(std::ostream& os, const Vector<TT>& vec, const IOFormat& fmt) {
        const size_t n = vec.size();
        std::vector<std::string> repr(n);
        size_t width = 0;
        for (size_t i = 0; i < n; ++i) {
            repr[i] = format_scalar(vec(i), fmt);
            width = std::max(width, repr[i].size());
        };
        os << "[";
        for (size_t i = 0; i < n; ++i) {
            os << std::setw(width) << repr[i];
            if (i + 1 < n) os << " ";
        };
        os << "]";
        return os;
    };

    template<typename TT, Layout LL>
    std::ostream& print(std::ostream& os, const Matrix<TT, LL>& mat, const IOFormat& fmt) {
        const size_t m = mat.rows(), n = mat.cols();

        std::vector<std::vector<std::string>> repr(m, std::vector<std::string>(n));
        std::vector<size_t> widths(n, 0);

        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                repr[i][j] = format_scalar(mat(i, j), fmt);
                widths[j] = std::max(widths[j], repr[i][j].size());
            };
        };

        os << "[\n";
        for (size_t i = 0; i < m; ++i) {
            os << "  [";
            for (size_t j = 0; j < n; ++j) {
                os << std::setw(widths[j]) << repr[i][j];
                if (j + 1 < n) os << " ";
            };
            os << "]";
            if (i + 1 < m) os << "\n";
        };
        os << "\n]";
        return os;
    };

    inline IOFormat default_format() { return IOFormat{}; };

    template<typename TT>
    std::ostream& operator<<(std::ostream& os, const Vector<TT>& vec) {
        return print(os, vec, default_format());
    };

    template<typename TT, Layout LL>
    std::ostream& operator<<(std::ostream& os, const Matrix<TT, LL>& mat) {
        return print(os, mat, default_format());
    };
};