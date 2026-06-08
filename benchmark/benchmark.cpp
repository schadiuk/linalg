#include <linalg.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace linalg::bench {
    using Clock = std::chrono::high_resolution_clock;
    using Seconds = std::chrono::duration<double>;

    using Matrix_f = Matrix<float>;
    using Matrix_d = Matrix<double>;
    using Matrix_cd = Matrix<std::complex<double>>;
    using Vector_f = Vector<float>;
    using Vector_d = Vector<double>;
    using Vector_cd = Vector<std::complex<double>>;

    struct BenchResult {
        std::string section; // Section label for grouping.
        std::string name;
        size_t problem_size = 0; // Characteristic N.
        size_t threads = 0;
        double median_sec = 0;
        double iqr_sec = 0;
        double gflops = 0;
        double bandwidth_GBs = 0;
        std::string notes;

        static std::string csv_header() {
            return "section,name,N,threads,median_s,iqr_s,gflops,bandwidth_GBs,notes";
        };

        std::string to_csv() const {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6)
            << section << ','
            << name << ','
            << problem_size << ','
            << threads << ','
            << median_sec << ','
            << iqr_sec << ','
            << gflops << ','
            << bandwidth_GBs << ','
            << notes;
            return ss.str();
        };

        void print() const {
            std::cout
                << std::left << std::setw(50) << name
                << std::right << std::setw(7) << problem_size
                << "  thr=" << std::setw(2) << threads
                << std::fixed << std::setprecision(3)
                << "  t=" << std::setw(9) << median_sec * 1e3 << "ms"
                << "  iqr=" << std::setw(7) << iqr_sec * 1e3 << "ms";
            if (gflops > 0)
                std::cout << "  " << std::setw(8) << std::setprecision(2) << gflops << " GFLOP/s";
            if (bandwidth_GBs > 0)
                std::cout << "  " << std::setw(7) << std::setprecision(2) << bandwidth_GBs << " GB/s";
            if (!notes.empty())
                std::cout << "  [" << notes << "]";
            std::cout << '\n';
        };
    };

    // Returns {median_seconds, IQR_seconds} over `reps` timed calls after `warmup`.
    template<typename F>
    std::pair<double,double> time_stats(F&& fn, int reps = 11, int warmup = 2) {
        for (int i = 0; i < warmup; ++i) fn();
        std::vector<double> t(reps);
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            fn();
            t[i] = Seconds(Clock::now() - t0).count();
        };
        std::sort(t.begin(), t.end());
        double med = t[reps / 2];
        double iqr = (reps >= 4) ? t[3 * reps / 4] - t[reps / 4] : 0.0;
        return {med, iqr};
    };

    // Convenience: single-number median only (no IQR needed).
    template<typename F>
    double time_median(F&& fn, int reps = 11, int warmup = 2) {
        return time_stats(std::forward<F>(fn), reps, warmup).first;
    };

    // ── Problem-size constants ────────────────────────────────────────────────────
    // Cache tiers (doubles): L1 ~32 KB -> 4096 d, L2 ~256 KB -> 32768 d, L3 ~8 MB -> 1M d, DRAM beyond.

    static const std::vector<size_t> VEC_SIZES = {
        64, // register-resident
        512, // small L1
        4096, // L1 limit
        16384, // deep L2
        65536, // L2->L3
        262144, // L3
        1048576, // large L3 / near DRAM boundary
        4194304 // DRAM
    };

    static const std::vector<size_t> MAT_SIZES = {
        32, 64, 128, 256, 512, 1024, 2048, 4096
    };

    static const std::vector<size_t> SMALL_MAT = {
        32, 64, 128, 256, 512, 1024
    };

    inline size_t hw_threads() { return std::max(1u, std::thread::hardware_concurrency()); };

    // Helper: build an SPD (symmetric positive-definite) matrix A = B*B^T + n*I.
    inline Matrix_d make_spd(size_t n) {
        auto B = Matrix_d::random(n, n);
        Matrix_d A(n, n, 0.0);
        gemm(1.0, expr(B), expr(transpose(B)), 0.0, A);
        for (size_t i = 0; i < n; ++i) A(i, i) += static_cast<double>(n);
        return A;;
    }

    // Helper: build an upper-triangular matrix (non-unit diagonal) from a random matrix.
    inline Matrix_d make_upper_tri(size_t n) {
        auto A = Matrix_d::random(n, n);
        // Ensure well-conditioned diagonal.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) A(i, j) = 0.0;
            A(i, i) = 2.0 + std::abs(A(i, i));
        };
        return A;
    };

    inline Matrix_d make_lower_tri(size_t n) {
        auto A = Matrix_d::random(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) A(i, j) = 0.0;
            A(i, i) = 2.0 + std::abs(A(i, i));
        };
        return A;
    };
};