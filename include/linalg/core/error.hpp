#pragma once

#include <stdexcept>
#include <source_location>

namespace linalg::detail {
    class BoundsError : public std::out_of_range {
    public:
        explicit BoundsError(const char* expr, const std::source_location&    loc = std::source_location::current()):
            std::out_of_range(make_message(expr, loc)) {};
    private:
        static std::string make_message(const char* expr, const std::source_location& loc) {
            return std::string("Bounds check failed: (") + expr + ") "
                + "in " + loc.function_name()
                + " at " + loc.file_name()
                + ":" + std::to_string(loc.line());
        };
    };

    // Helper called by the macro
    [[noreturn]] inline void bounds_fail(
            const char* expr,
            const std::source_location& loc) {
        throw BoundsError(expr, loc);
    };

    // Marked [[likely]] on the fast path so the branch predictor and optimiser treat a passing check as free
    inline void bounds_check(bool cond, const char* expr,
        const std::source_location& loc = std::source_location::current()) {
        if (cond) [[likely]] return;
        else [[unlikely]] bounds_fail(expr, loc);
    };
};

// Bound-checking macro with zero overhead when NDEBUG is defined, and detailed error messages otherwise
#ifndef NDEBUG
#define BOUNDS_CHECK(cond) ::linalg::detail::bounds_check((cond), #cond, std::source_location::current())
#else
#define BOUNDS_CHECK(cond) ((void)0)
#endif