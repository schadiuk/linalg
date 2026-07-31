#pragma once

#include <linalg/storage/vector.hpp>
#include <linalg/storage/matrix.hpp>

#include <string>
#include <sstream>
#include <iomanip>

namespace linalg {
    /// @brief Base formatter.
    /// @note Conains `int precision`, `bool scientific`, and `bool complex_as_pair` fields.    
    /// @note Defaults to plain 3-decimal `std::complex` tuples.
    struct IOFormat {
        int precision = 3;
        bool scientific = false;
        bool complex_as_pair = false; // False: a + bi, true: (a,b)
    };

    namespace detail {     
        /// @brief Process-wide default.
        inline IOFormat& default_format_storage() {
            static IOFormat fmt{};
            return fmt;
        };

        // Reserved `std::ios_base` extensible-storage slot indices (lazily allocated once per process via xalloc):

        inline int format_precision_idx() { static int idx = std::ios_base::xalloc(); return idx; };
        inline int format_flags_idx() { static int idx = std::ios_base::xalloc(); return idx; };

        constexpr long FMT_PRECISION_SET  = 1 << 0;
        constexpr long FMT_SCIENTIFIC_SET = 1 << 1;
        constexpr long FMT_SCIENTIFIC = 1 << 2;
        constexpr long FMT_COMPLEX_SET = 1 << 3;
        constexpr long FMT_COMPLEX_PAIR = 1 << 4;

        struct precision_manip { int value; };

        inline std::ostream& operator<<(std::ostream& os, precision_manip m) {
            os.iword(format_flags_idx()) |= FMT_PRECISION_SET;
            os.iword(format_precision_idx()) = m.value;
            return os;
        };
    };

    /// @brief Sets the process-wide default formatting used by `operator<<` on any stream without a per-stream override. 
    /// @param fmt The new process-wide default.
    inline void set_default_format(const IOFormat& fmt) { detail::default_format_storage() = fmt; };

    /// @brief Retrieves the current process-wide default format.
    inline IOFormat default_format() { return detail::default_format_storage(); };

    /// @brief Resolves the effective format for a given stream: per-stream overrides take precedence field-by-field over the process-wide default.
    /// @param os The stream whose format state is being queried.
    inline IOFormat stream_format(std::ios_base& os) {
        IOFormat fmt = default_format();
        const long flags = os.iword(detail::format_flags_idx());
        if (flags & detail::FMT_PRECISION_SET) fmt.precision = static_cast<int>(os.iword(detail::format_precision_idx()));
        if (flags & detail::FMT_SCIENTIFIC_SET) fmt.scientific = (flags & detail::FMT_SCIENTIFIC) != 0;
        if (flags & detail::FMT_COMPLEX_SET) fmt.complex_as_pair = (flags & detail::FMT_COMPLEX_PAIR) != 0;
        return fmt;
    };

    /// @brief Stream manipulator: sets this stream's `linalg` print precision (independent of `std::setprecision`).
    /// @param p Digits after the decimal point.
    inline detail::precision_manip setprecision(int p) { return detail::precision_manip{p}; };

    /// @brief Stream manipulator: switches this stream's `linalg` output to scientific notation.
    inline std::ostream& scientific(std::ostream& os) {
        long& f = os.iword(detail::format_flags_idx());
        f |= (detail::FMT_SCIENTIFIC_SET | detail::FMT_SCIENTIFIC);
        return os;
    };

    /// @brief Stream manipulator: switches this stream's `linalg` output to fixed-point notation.
    inline std::ostream& fixed(std::ostream& os) {
        long& f = os.iword(detail::format_flags_idx());
        f |= detail::FMT_SCIENTIFIC_SET;
        f &= ~detail::FMT_SCIENTIFIC;
        return os;
    };

    /// @brief Stream manipulator: prints complex entries on this stream as `(re, im)` tuples.
    inline std::ostream& complex_as_pair(std::ostream& os) {
        long& f = os.iword(detail::format_flags_idx());
        f |= (detail::FMT_COMPLEX_SET | detail::FMT_COMPLEX_PAIR);
        return os;
    };

    /// @brief Stream manipulator: prints complex entries on this stream as `a + bi`.
    inline std::ostream& complex_as_sum(std::ostream& os) {
        long& f = os.iword(detail::format_flags_idx());
        f |= detail::FMT_COMPLEX_SET;
        f &= ~detail::FMT_COMPLEX_PAIR;
        return os;
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

    /// @brief Prints using this stream's resolved format (per-stream override, else the process-wide default).
    template<typename TT>
    std::ostream& print(std::ostream& os, const Vector<TT>& vec) { return print(os, vec, stream_format(os)); };

    /// @brief Prints using this stream's resolved format (per-stream override, else the process-wide default).
    template<typename TT, Layout LL>
    std::ostream& print(std::ostream& os, const Matrix<TT, LL>& mat) { return print(os, mat, stream_format(os)); };

    template<typename TT>
    std::ostream& operator<<(std::ostream& os, const Vector<TT>& vec) {
        return print(os, vec, stream_format(os));
    };

    template<typename TT, Layout LL>
    std::ostream& operator<<(std::ostream& os, const Matrix<TT, LL>& mat) {
        return print(os, mat, stream_format(os));
    };
};