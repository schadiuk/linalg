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

    std::vector<BenchResult> bench_axpy(int reps = 11) {
        std::vector<BenchResult> out;
        const double alpha = 1.5;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ axpy(alpha, expr(x), y); }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "axpy<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0 * N / med / 1e9;
            r.bandwidth_GBs = 3.0 * N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        // float variant at the same sizes
        for (size_t N : VEC_SIZES) {
            auto x = Vector_f::random(N);
            auto y = Vector_f::random(N);
            auto [med, iqr] = time_stats([&]{ axpy(1.5f, expr(x), y); }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "axpy<float>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0 * N / med / 1e9;
            r.bandwidth_GBs = 3.0 * N * sizeof(float) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_axpby(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ axpby(1.5, expr(x), 0.9, y); }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "axpby<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 3.0 * N / med / 1e9;   // 2 muls + 1 add per element
            r.bandwidth_GBs = 3.0 * N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_dot(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{
                volatile auto s = dot(expr(x), expr(y)); (void)s;
            }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "dot<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0 * N / med / 1e9;
            r.bandwidth_GBs = 2.0 * N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_dotc(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_cd::random(N);
            auto y = Vector_cd::random(N);
            auto [med, iqr] = time_stats([&]{
                volatile auto s = dotc(expr(x), expr(y)); (void)s;
            }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "dotc<complex<double>>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 6.0 * N / med / 1e9;   // 4 real muls + 2 adds per complex pair
            r.bandwidth_GBs = 2.0 * N * sizeof(std::complex<double>) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_nrm2(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{
                volatile double v = nrm2(expr(x)); (void)v;
            }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "nrm2<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0 * N / med / 1e9;
            r.bandwidth_GBs = N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_asum(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{
                volatile double v = asum(expr(x)); (void)v;
            }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "asum<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0 * N / med / 1e9;
            r.bandwidth_GBs = N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_scal(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ scal(0.99, x); }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "scal<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = N / med / 1e9;
            r.bandwidth_GBs = 2.0 * N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_iamax(int reps = 13) {
        std::vector<BenchResult> out;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{
                volatile auto idx = iamax(expr(x)); (void)idx;
            }, reps);
            BenchResult r;
            r.section = "Level-1";  r.name = "iamax<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 0;
            r.bandwidth_GBs = N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_gemv(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : {32u,64u,128u,256u,512u,1024u,2048u,4096u}) {
            auto A = Matrix_d::random(N, N);
            auto x = Vector_d::random(N);
            auto y = Vector_d(N, 0.0);
            auto [med, iqr] = time_stats([&]{ gemv(1.0, expr(A), expr(x), 0.5, y); }, reps);
            double flops = 2.0 * N * N;
            double bytes = (double(N)*N + N + 2.0*N) * sizeof(double);
            BenchResult r;
            r.section = "Level-2";  r.name = "gemv<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    // trsv forward/backward substitution:  2 cases * 2 uplo * 2 trans = 8 variants. We benchmark the most practically important ones.
    std::vector<BenchResult> bench_trsv(int reps = 11) {
        std::vector<BenchResult> out;
        struct Case { char uplo, trans; std::string tag; };
        const std::vector<Case> cases = {
            {'U', 'N', "upper-notrans"},
            {'L', 'N', "lower-notrans"},
            {'U', 'T', "upper-trans"},
            {'L', 'T', "lower-trans"},
        };
        for (size_t N : SMALL_MAT) {
            for (const auto& c : cases) {
                auto A = (c.uplo == 'U') ? make_upper_tri(N) : make_lower_tri(N);
                auto b = Vector_d::random(N);
                auto x = b;
                auto [med, iqr] = time_stats([&]{
                    x = b;
                    trsv(c.uplo, c.trans, 'N', expr(A), x);
                }, reps);
                double flops = double(N) * N;  // N^2 FLOPs for triangular solve
                BenchResult r;
                r.section = "Level-2";
                r.name = "trsv<double> " + c.tag;
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                r.bandwidth_GBs = 0.5 * N * N * sizeof(double) / med / 1e9;
                out.push_back(r);
            };
        };
        return out;
    };

    std::vector<BenchResult> bench_trmv(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = make_upper_tri(N);
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ trmv('U', 'N', 'N', expr(A), x); }, reps);
            double flops = double(N) * N;
            BenchResult r;
            r.section = "Level-2";  r.name = "trmv<double> upper-notrans";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = 0.5 * N * N * sizeof(double) / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_symv(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = make_spd(N);
            auto x = Vector_d::random(N);
            auto y = Vector_d::zeros(N);
            auto [med, iqr] = time_stats([&]{ symv('U', 1.0, expr(A), expr(x), 0.5, y); }, reps);
            double flops = 2.0 * N * N;
            double bytes = (double(N)*N + N + 2.0*N) * sizeof(double);
            BenchResult r;
            r.section = "Level-2";  r.name = "symv<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_ger(int reps = 11) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_d::random(N, N);
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ ger(0.5, expr(x), expr(y), A); }, reps);
            double flops = 2.0 * N * N;
            double bytes = (double(N)*N + 2.0*N) * sizeof(double);
            BenchResult r;
            r.section = "Level-2";  r.name = "ger<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            out.push_back(r);
        };
        // gerc with complex scalars
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_cd::random(N, N);
            auto x = Vector_cd::random(N);
            auto y = Vector_cd::random(N);
            std::complex<double> alpha(0.5, 0.0);
            auto [med, iqr] = time_stats([&]{ gerc(alpha, expr(x), expr(y), A); }, reps);
            double flops = 8.0 * N * N; // 4 real muls + extras per complex product.
            BenchResult r;
            r.section = "Level-2";  r.name = "gerc<complex<double>>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = 0;
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_gemm_square(int reps = 7) {
        std::vector<BenchResult> out;
        for (size_t N : MAT_SIZES) {
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            gemm(0.5, expr(A), expr(B), 0.5, C); // Warm-up run.
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0 * double(N)*N*N;
            double bytes = 3.0 * double(N)*N * sizeof(double);
            BenchResult r;
            r.section = "Level-3";  r.name = "gemm_square<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    // Tall-skinny GEMM: fixed M=N=2048, vary K.
    std::vector<BenchResult> bench_gemm_tall_skinny(int reps = 9) {
        std::vector<BenchResult> out;
        const size_t M = 2048;
        for (size_t K : {1u, 4u, 16u, 64u, 256u, 512u, 2048u}) {
            auto A = Matrix_d::random(M, K);
            auto B = Matrix_d::random(K, M);
            auto C = Matrix_d::zeros(M, M);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0 * M * M * K;
            BenchResult r;
            r.section = "Level-3";
            r.name = "gemm_tall_skinny K=" + std::to_string(K);
            r.problem_size = M;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.notes = "M=N=" + std::to_string(M) + " K=" + std::to_string(K);
            out.push_back(r);
        };
        return out;
    };

    // Wide-short GEMM: fixed M=K=2048, vary N (complement to tall-skinny).
    std::vector<BenchResult> bench_gemm_wide_short(int reps = 9) {
        std::vector<BenchResult> out;
        const size_t M = 2048, K = 2048;
        for (size_t N_col : {1u, 4u, 16u, 64u, 256u, 512u}) {
            auto A = Matrix_d::random(M, K);
            auto B = Matrix_d::random(K, N_col);
            auto C = Matrix_d::zeros(M, N_col);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0 * M * N_col * K;
            BenchResult r;
            r.section = "Level-3";
            r.name = "gemm_wide_short N=" + std::to_string(N_col);
            r.problem_size = M;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.notes = "M=K=" + std::to_string(M) + " N=" + std::to_string(N_col);
            out.push_back(r);
        };
        return out;
    };

    // Transposed GEMM: NN, TN, NT, TT, HN at N=1024.
    std::vector<BenchResult> bench_gemm_transposed(int reps = 7) {
        std::vector<BenchResult> out;
        const size_t N = 1024;
        auto A = Matrix_d::random(N, N);
        auto B = Matrix_d::random(N, N);
        auto C = Matrix_d::zeros(N, N);
        double flops = 2.0 * double(N)*N*N;
        struct Case { std::string tag; std::function<void()> fn; };
        std::vector<Case> cases = {
            {"NN", [&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }},
            {"TN", [&]{ gemm(0.5, expr(transpose(A)), expr(B), 0.5, C); }},
            {"NT", [&]{ gemm(0.5, expr(A), expr(transpose(B)), 0.5, C); }},
            {"TT", [&]{ gemm(0.5, expr(transpose(A)), expr(transpose(B)), 0.5, C); }},
            {"HN", [&]{ gemm(0.5, expr(hermitian(A)), expr(B), 0.5, C); }},
        };
        for (auto& c : cases) {
            c.fn();
            auto [med, iqr] = time_stats(c.fn, reps);
            BenchResult r;
            r.section = "Level-3";
            r.name = "gemm_" + c.tag + "<double> N=" + std::to_string(N);
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    // trsm: vary number of RHS columns.
    std::vector<BenchResult> bench_trsm(int reps = 9) {
        std::vector<BenchResult> out;
        const size_t N = 512; // Triangular system size
        auto A = make_upper_tri(N);
        for (size_t nrhs : {1u, 4u, 16u, 64u, 256u, 512u}) {
            auto B = Matrix_d::random(N, nrhs);
            auto Bwork = B;
            trsm('L','U','N','N', 1.0, expr(A), Bwork);
            auto [med, iqr] = time_stats([&]{ Bwork = B; trsm('L','U','N','N', 1.0, expr(A), Bwork); }, reps);
            double flops = double(N)*N * nrhs; // ~N^2/2 * nrhs triangular ops.
            BenchResult r;
            r.section = "Level-3";
            r.name = "trsm<double> nrhs=" + std::to_string(nrhs);
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.notes = "N=" + std::to_string(N) + " nrhs=" + std::to_string(nrhs);
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_syrk(int reps = 9) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            size_t K = N / 4 + 1;
            auto A = Matrix_d::random(N, K);
            auto C = Matrix_d::zeros(N, N);
            syrk('U', 'N', 1.0, expr(A), 0.0, C);
            auto [med, iqr] = time_stats([&]{ syrk('U', 'N', 1.0, expr(A), 0.0, C); }, reps);
            // N*(N+1)/2 output elements, each needing K multiplications+additions.
            double flops = double(N) * (N + 1) * K;
            BenchResult r;
            r.section = "Level-3";  r.name = "syrk<double> K=N/4";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.notes = "K=" + std::to_string(K);
            out.push_back(r);
        };
        return out;
    };

    std::vector<BenchResult> bench_herk(int reps = 9) {
        std::vector<BenchResult> out;
        for (size_t N : {64u, 128u, 256u, 512u}) {
            size_t K = N / 4 + 1;
            auto A = Matrix_cd::random(N, K);
            auto C = Matrix_cd::zeros(N, N);
            herk('U', 'N', 1.0, expr(A), 0.0, C);
            auto [med, iqr] = time_stats([&]{ herk('U', 'N', 1.0, expr(A), 0.0, C); }, reps);
            double flops = 8.0 * double(N) * (N + 1) * K; // Complex FLOP count.
            BenchResult r;
            r.section = "Level-3";  r.name = "herk<complex<double>> K=N/4";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.notes = "K=" + std::to_string(K);
            out.push_back(r);
        };
        return out;
    };

    // LU factorisation only (excludes solve, det, inverse).
    std::vector<BenchResult> bench_lu_factor(int reps = 7) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_d::random(N, N);
            lu(A);
            auto [med, iqr] = time_stats([&]{ auto r = lu(A); (void)r; }, reps);
            // LU: ~(2/3)*N^3 FLOPs.
            double flops = (2.0 / 3.0) * double(N)*N*N;
            BenchResult r;
            r.section = "Decompositions";  r.name = "lu_factor<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    // LU solve: single RHS and multiple RHS.
    std::vector<BenchResult> bench_lu_solve(int reps = 9) {
        std::vector<BenchResult> out;
        // Single RHS
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_d::random(N, N);
            auto res = lu(A);
            auto b = Vector_d::random(N);
            auto x = b;
            auto [med, iqr] = time_stats([&]{ x = b; lu_solve(res, x); }, reps);
            double flops = 2.0 * double(N)*N; // Forward + backward substitution.
            BenchResult r;
            r.section = "Decompositions";  r.name = "lu_solve<double> single-rhs";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            out.push_back(r);
        };
        // Multiple RHS: fixed N=512, vary nrhs.
        {
            size_t N = 512;
            auto A = Matrix_d::random(N, N);
            auto res = lu(A);
            for (size_t nrhs : {1u, 4u, 16u, 64u, 256u}) {
                auto B = Matrix_d::random(N, nrhs);
                auto Bw = B;
                lu_solve(res, Bw);
                auto [med, iqr] = time_stats([&]{ Bw = B; lu_solve(res, Bw); }, reps);
                double flops = 2.0 * double(N)*N * nrhs;
                BenchResult r;
                r.section = "Decompositions";
                r.name = "lu_solve<double> nrhs=" + std::to_string(nrhs);
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                r.notes = "N=" + std::to_string(N);
                out.push_back(r);
            };
        }
        return out;
    };

    // LU determinant and inverse.
    std::vector<BenchResult> bench_lu_det_inv(int reps = 9) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_d::random(N, N);
            auto res = lu(A);
            {
                auto [med, iqr] = time_stats([&]{
                    volatile auto d = lu_det(res); (void)d;
                }, reps);
                BenchResult r;
                r.section = "Decompositions";  r.name = "lu_det<double>";
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med;  r.iqr_sec = iqr;
                r.notes = "given pre-factored LU";
                out.push_back(r);
            }
            {
                auto [med, iqr] = time_stats([&]{
                    auto inv = lu_inverse(res); (void)inv;
                }, reps);
                double flops = 2.0 * double(N)*N*N;  // trsm on identity.
                BenchResult r;
                r.section = "Decompositions";  r.name = "lu_inverse<double>";
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                r.notes = "given pre-factored LU";
                out.push_back(r);
            }
        };
        return out;
    };

    // Full LU pipeline: factor + single solve combined.
    std::vector<BenchResult> bench_lu_full(int reps = 7) {
        std::vector<BenchResult> out;
        for (size_t N : SMALL_MAT) {
            auto A = Matrix_d::random(N, N);
            auto b = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{
                auto x = b;
                auto res = lu(A);
                lu_solve(res, x);
            }, reps);
            double flops = (2.0/3.0 + 2.0) * double(N)*N*N;
            BenchResult r;
            r.section = "Decompositions";  r.name = "lu_full_pipeline<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    // QR factorisation – all modes
    std::vector<BenchResult> bench_qr(int reps = 7) {
        std::vector<BenchResult> out;
        using QRResult_d = QRResult<double, Layout::RowMajor>;
        struct QRCase { std::string tag; std::function<QRResult_d(const Matrix_d&)> fn; };
        std::vector<QRCase> modes = {
            {"reduced", [](const Matrix_d& A) -> QRResult_d { return qr_reduced(A); }},
            {"complete", [](const Matrix_d& A) -> QRResult_d { return qr_complete(A); }},
            {"R-only", [](const Matrix_d& A) -> QRResult_d { return qr_r(A); }},
            {"pivoted", [](const Matrix_d& A) -> QRResult_d { return qr_pivoted(A); }},
        };
        for (size_t N : {64u, 128u, 256u, 512u, 1024u}) {
            for (const auto& m : modes) {
                auto A = Matrix_d::random(N, N);
                m.fn(A); // Warm-up run.
                auto [med, iqr] = time_stats([&]{ auto r = m.fn(A); (void)r; }, reps);
                // QR: ~(4/3)*N^3 for Householder with Q accumulation; R-only ~(2/3)*N^3.
                double base = (m.tag == "R-only") ? 2.0/3.0 : 4.0/3.0;
                double flops = base * double(N)*N*N;
                BenchResult r;
                r.section = "Decompositions";
                r.name = "qr_" + m.tag + "<double>";
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                out.push_back(r);
            };
        };
        return out;
    };

    std::vector<BenchResult> bench_norms_vector(int reps = 11) {
        std::vector<BenchResult> out;
        struct NCase { std::string kind; };
        for (const std::string& kind : {"1","2","inf"}) {
            for (size_t N : VEC_SIZES) {
                auto x = Vector_d::random(N);
                auto [med, iqr] = time_stats([&]{
                    volatile double v = norm(expr(x), kind); (void)v;
                }, reps);
                BenchResult r;
                r.section = "Norms";
                r.name = "vec_norm_" + kind + "<double>";
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.bandwidth_GBs = N * sizeof(double) / med / 1e9;
                out.push_back(r);
            };
        };
        return out;
    };

    std::vector<BenchResult> bench_norms_matrix(int reps = 9) {
        std::vector<BenchResult> out;
        for (const std::string& kind : {"1","fro","inf"}) {
            for (size_t N : SMALL_MAT) {
                auto A = Matrix_d::random(N, N);
                auto [med, iqr] = time_stats([&]{
                    volatile double v = norm(expr(A), kind); (void)v;
                }, reps);
                BenchResult r;
                r.section = "Norms";
                r.name = "mat_norm_" + kind + "<double>";
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.bandwidth_GBs = double(N)*N * sizeof(double) / med / 1e9;
                out.push_back(r);
            };
        };
        return out;
    };

    // Layout comparison: RowMajor vs ColMajor at various GEMM sizes.
    std::vector<BenchResult> bench_layout(int reps = 9) {
        std::vector<BenchResult> out;
        for (size_t N : {128u, 256u, 512u, 1024u, 2048u}) {
            using MR = Matrix<double, Layout::RowMajor>;
            using MC = Matrix<double, Layout::ColMajor>;
            double flops = 2.0 * double(N)*N*N;
            {
                auto A = MR::random(N,N), B = MR::random(N,N), C = MR::zeros(N,N);
                gemm(0.5, expr(A), expr(B), 0.5, C);
                auto [med, iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "gemm_RowMajor N=" + std::to_string(N);
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                out.push_back(r);
            }
            {
                auto A = MC::random(N,N), B = MC::random(N,N), C = MC::zeros(N,N);
                gemm(0.5, expr(A), expr(B), 0.5, C);
                auto [med, iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "gemm_ColMajor N=" + std::to_string(N);
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med;  r.iqr_sec = iqr;
                r.gflops = flops / med / 1e9;
                out.push_back(r);
            }
        };
        return out;
    };

    // Dtype comparison: float / double / complex<double> at fixed N.
    std::vector<BenchResult> bench_dtype(int reps = 9) {
        std::vector<BenchResult> out;
        const size_t N = 512;
        // double
        {
            auto A = Matrix_d::random(N,N), B = Matrix_d::random(N,N), C = Matrix_d::zeros(N,N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
            BenchResult r;
            r.section = "Cross-cutting";  r.name = "gemm_dtype<double>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0*double(N)*N*N/med/1e9;
            out.push_back(r);
        }
        // float
        {
            auto A = Matrix_f::random(N,N), B = Matrix_f::random(N,N), C = Matrix_f::zeros(N,N);
            gemm(0.5f, expr(A), expr(B), 0.5f, C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5f,expr(A),expr(B),0.5f,C); }, reps);
            BenchResult r;
            r.section = "Cross-cutting";  r.name = "gemm_dtype<float>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0*double(N)*N*N/med/1e9;
            r.notes = "expect ~2x vs double (AVX2 width)";
            out.push_back(r);
        }
        // complex<double>
        {
            auto A = Matrix_cd::random(N,N), B = Matrix_cd::random(N,N), C = Matrix_cd::zeros(N,N);
            std::complex<double> a05(0.5,0.), b05(0.5,0.);
            gemm(a05, expr(A), expr(B), b05, C);
            auto [med,iqr] = time_stats([&]{ gemm(a05,expr(A),expr(B),b05,C); }, reps);
            BenchResult r;
            r.section = "Cross-cutting";  r.name = "gemm_dtype<complex<double>>";
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 8.0*double(N)*N*N/med/1e9;
            r.notes = "8 real FLOP/cmul";
            out.push_back(r);
        }
        return out;
    };

    // Expression-tree overhead: ET chain vs raw loop.
    std::vector<BenchResult> bench_expr_overhead(int reps = 13) {
        std::vector<BenchResult> out;
        for (size_t N : {1024u, 16384u, 65536u, 1u<<20}) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto z = Vector_d::random(N);
            Vector_d out_v(N);
            // Shallow: out = x + 2*y - z  (depth 3).
            auto [med_et, iqr_et] = time_stats([&]{
                out_v = expr(x) + 2.0 * expr(y) - expr(z);
            }, reps);
            // Raw loop baseline.
            auto [med_raw, iqr_raw] = time_stats([&]{
                for (size_t i = 0; i < N; ++i) out_v[i] = x[i] + 2.0*y[i] - z[i];
            }, reps);
            double bytes = 4.0 * N * sizeof(double);
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "et_chain_shallow ET N=" + std::to_string(N);
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med_et;  r.iqr_sec = iqr_et;
                r.gflops = 3.0 * N / med_et / 1e9;
                r.bandwidth_GBs = bytes / med_et / 1e9;
                out.push_back(r);
            }
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "et_chain_shallow raw N=" + std::to_string(N);
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med_raw;  r.iqr_sec = iqr_raw;
                r.gflops = 3.0 * N / med_raw / 1e9;
                r.bandwidth_GBs = bytes / med_raw / 1e9;
                r.notes = "raw loop baseline";
                out.push_back(r);
            }
        };
        // Deeper chain: out = sin(x) + cos(y) * z - exp(z)  (depth 7).
        for (size_t N : {16384u, 262144u}) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto z = Vector_d::random(N);
            Vector_d out_v(N);
            auto [med_et, iqr_et] = time_stats([&]{
                out_v = linalg::sin(expr(x)) + linalg::cos(expr(y)) * expr(z) - linalg::exp(expr(z));
            }, reps);
            auto [med_raw, iqr_raw] = time_stats([&]{
                for (size_t i = 0; i < N; ++i)
                    out_v[i] = std::sin(x[i]) + std::cos(y[i])*z[i] - std::exp(z[i]);
            }, reps);
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "et_chain_deep ET N=" + std::to_string(N);
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med_et;  r.iqr_sec = iqr_et;
                r.notes = "sin(x)+cos(y)*z-exp(z), depth 7";
                out.push_back(r);
            }
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "et_chain_deep raw N=" + std::to_string(N);
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med_raw;  r.iqr_sec = iqr_raw;
                r.notes = "raw loop baseline";
                out.push_back(r);
            }
        };
        return out;
    };

    // Aliasing detection cost: alias-free path vs aliased (forces temp buffer).
    std::vector<BenchResult> bench_aliasing(int reps = 15) {
        std::vector<BenchResult> out;
        for (size_t N : {64u, 256u, 1024u, 4096u}) {
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            double bytes = 3.0 * double(N)*N * sizeof(double);
            {
                auto [med, iqr] = time_stats([&]{ C = expr(A) + expr(B); }, reps);
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "alias_no-alias N=" + std::to_string(N);
                r.problem_size = N*N;  r.threads = 1;
                r.median_sec = med;  r.iqr_sec = iqr;
                r.bandwidth_GBs = bytes / med / 1e9;
                out.push_back(r);
            }
            {
                auto [med, iqr] = time_stats([&]{ A = expr(A) + expr(B); }, reps);
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "alias_aliased   N=" + std::to_string(N);
                r.problem_size = N*N;  r.threads = 1;
                r.median_sec = med;  r.iqr_sec = iqr;
                r.bandwidth_GBs = bytes / med / 1e9;
                r.notes = "temp alloc + extra copy";
                out.push_back(r);
            }
        };
        return out;
    };

    // Thread-pool dispatch overhead: tiny-N benchmark shows cost of parallel_for for problems so small that parallelism is counter-productive.
    std::vector<BenchResult> bench_threadpool_overhead(int reps = 25) {
        std::vector<BenchResult> out;
        // Sizes deliberately below PARALLEL_THRESHOLD_SIMPLE (65536) and below PARALLEL_THRESHOLD_COMPUTE (4096) so we can observe the serial fast-path vs the parallel path.
        for (size_t N : {8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 4096u, 16384u}) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            // scal is a single-array operation with the simplest kernel.
            auto [med_scal, iqr_scal] = time_stats([&]{ scal(0.99, x); }, reps);
            // axpy is a two-array operation.
            auto [med_axpy, iqr_axpy] = time_stats([&]{ axpy(1.5, expr(x), y); }, reps);
            // raw baseline for comparison.
            auto [med_raw, iqr_raw] = time_stats([&]{
                for (size_t i = 0; i < N; ++i) y[i] += 1.5 * x[i];
            }, reps);
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "threadpool_scal N=" + std::to_string(N);
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med_scal;  r.iqr_sec = iqr_scal;
                r.bandwidth_GBs = 2.0 * N * sizeof(double) / med_scal / 1e9;
                out.push_back(r);
            }
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "threadpool_axpy N=" + std::to_string(N);
                r.problem_size = N;  r.threads = hw_threads();
                r.median_sec = med_axpy;  r.iqr_sec = iqr_axpy;
                r.bandwidth_GBs = 3.0 * N * sizeof(double) / med_axpy / 1e9;
                out.push_back(r);
            }
            {
                BenchResult r;
                r.section = "Cross-cutting";
                r.name = "threadpool_raw  N=" + std::to_string(N);
                r.problem_size = N;  r.threads = 1;
                r.median_sec = med_raw;  r.iqr_sec = iqr_raw;
                r.bandwidth_GBs = 3.0 * N * sizeof(double) / med_raw / 1e9;
                r.notes = "single-thread raw loop";
                out.push_back(r);
            }
        };
        return out;
    };

    // Parallel-scaling: fixed-work GEMM at N=2048, + serial baseline at N=256.
    std::vector<BenchResult> bench_parallel_scaling(int reps = 7) {
        std::vector<BenchResult> out;
        // Serial baseline.
        {
            size_t N = 256;
            auto A = Matrix_d::random(N,N), B = Matrix_d::random(N,N), C = Matrix_d::zeros(N,N);
            gemm(0.5,expr(A),expr(B),0.5,C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
            BenchResult r;
            r.section = "Cross-cutting";  r.name = "parallel_scaling_serial_baseline";
            r.problem_size = N;  r.threads = 1;
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = 2.0*double(N)*N*N / med / 1e9;
            r.notes = "below parallel threshold";
            out.push_back(r);
        }
        for (size_t N : {512u, 1024u, 2048u, 4096u}) {
            auto A = Matrix_d::random(N,N), B = Matrix_d::random(N,N), C = Matrix_d::zeros(N,N);
            gemm(0.5,expr(A),expr(B),0.5,C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
            double flops = 2.0*double(N)*N*N;
            BenchResult r;
            r.section = "Cross-cutting";
            r.name = "parallel_scaling N=" + std::to_string(N);
            r.problem_size = N;  r.threads = hw_threads();
            r.median_sec = med;  r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            out.push_back(r);
        };
        return out;
    };

    void print_roofline(const std::vector<BenchResult>& all) {
        double peak_bw = 0, peak_gflops_d = 0, peak_gflops_f = 0;
        for (const auto& r : all) {
            if (r.name.find("axpy<double>") != std::string::npos && r.problem_size >= (1u<<22))
                peak_bw = std::max(peak_bw, r.bandwidth_GBs);
            if (r.name.find("gemm_square<double>") != std::string::npos && r.problem_size >= 2048)
                peak_gflops_d = std::max(peak_gflops_d, r.gflops);
            if (r.name.find("gemm_dtype<float>") != std::string::npos)
                peak_gflops_f = std::max(peak_gflops_f, r.gflops);
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n══ Roofline summary ════════════════════════════════\n";
        std::cout << "  Peak memory BW (axpy large N)    : " << std::setw(8) << peak_bw << " GB/s\n";
        std::cout << "  Peak GEMM double (N>=2048)        : " << std::setw(8) << peak_gflops_d << " GFLOP/s\n";
        std::cout << "  Peak GEMM float                  : " << std::setw(8) << peak_gflops_f << " GFLOP/s\n";
        if (peak_bw > 0 && peak_gflops_d > 0)
            std::cout << "  Ridge point (double)             : " << std::setw(8)
                    << peak_gflops_d / peak_bw << " FLOP/byte\n";
    };

    void print_section_peaks(const std::vector<BenchResult>& all) {
        // Gather unique sections in order of first appearance.
        std::vector<std::string> section_order;
        std::map<std::string, double> best_gf, best_bw;
        for (const auto& r : all) {
            if (best_gf.find(r.section) == best_gf.end()) {
                section_order.push_back(r.section);
                best_gf[r.section] = 0;
                best_bw[r.section] = 0;
            }
            best_gf[r.section] = std::max(best_gf[r.section], r.gflops);
            best_bw[r.section] = std::max(best_bw[r.section], r.bandwidth_GBs);
        }
        std::cout << "\n══ Per-section peaks ═══════════════════════════════\n";
        std::cout << std::left << std::setw(20) << "Section"
                << std::right << std::setw(14) << "Peak GFLOP/s"
                << std::setw(14) << "Peak BW GB/s" << '\n';
        std::cout << std::string(48, '-') << '\n';
        for (const auto& sec : section_order) {
            std::cout << std::left  << std::setw(20) << sec
                    << std::right << std::fixed << std::setprecision(2)
                    << std::setw(14) << best_gf[sec]
                    << std::setw(14) << best_bw[sec] << '\n';
        };
    };

    struct BenchSuite {
        std::vector<BenchResult> results;

        void emit_section(const char* label, std::vector<BenchResult> v) {
            std::cout << "\n── " << label << " ────────────────────────────────────\n";
            for (auto& r : v) {
                r.section = label;
                r.print();
                results.push_back(r);
            };
        };

        void run_all(bool emit_csv = true, const std::string& csv_path = "linalg_bench.csv") {
            {
                size_t N = 1024;
                auto A = Matrix_d::random(N,N), B = Matrix_d::random(N,N), C = Matrix_d::zeros(N,N);
                gemm(0.5,expr(A),expr(B),0.5,C);
                double t = time_median([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, 5, 3);
                std::cout << "── Warm-up shot  N=1024 DGEMM: "
                        << std::fixed << std::setprecision(3) << t*1e3 << " ms  ("
                        << std::setprecision(2)
                        << 2.0*double(N)*N*N / t / 1e9 << " GFLOP/s) ────\n";
            }

            emit_section("Level-1: axpy", bench_axpy());
            emit_section("Level-1: axpby", bench_axpby());
            emit_section("Level-1: dot", bench_dot());
            emit_section("Level-1: dotc", bench_dotc());
            emit_section("Level-1: nrm2", bench_nrm2());
            emit_section("Level-1: asum", bench_asum());
            emit_section("Level-1: scal", bench_scal());
            emit_section("Level-1: iamax", bench_iamax());

            emit_section("Level-2: gemv", bench_gemv());
            emit_section("Level-2: trsv", bench_trsv());
            emit_section("Level-2: trmv", bench_trmv());
            emit_section("Level-2: symv", bench_symv());
            emit_section("Level-2: ger", bench_ger());

            emit_section("Level-3: gemm square", bench_gemm_square());
            emit_section("Level-3: gemm tall-skinny", bench_gemm_tall_skinny());
            emit_section("Level-3: gemm wide-short", bench_gemm_wide_short());
            emit_section("Level-3: gemm transposed", bench_gemm_transposed());
            emit_section("Level-3: trsm", bench_trsm());
            emit_section("Level-3: syrk", bench_syrk());
            emit_section("Level-3: herk", bench_herk());

            emit_section("Decomp: LU factor", bench_lu_factor());
            emit_section("Decomp: LU solve", bench_lu_solve());
            emit_section("Decomp: LU det+inv", bench_lu_det_inv());
            emit_section("Decomp: LU full pipeline", bench_lu_full());
            emit_section("Decomp: QR", bench_qr());

            emit_section("Norms: vector", bench_norms_vector());
            emit_section("Norms: matrix", bench_norms_matrix());

            emit_section("Cross: layout comparison", bench_layout());
            emit_section("Cross: dtype comparison", bench_dtype());
            emit_section("Cross: expr-tree overhead", bench_expr_overhead());
            emit_section("Cross: aliasing detection", bench_aliasing());
            emit_section("Cross: threadpool overhead", bench_threadpool_overhead());
            emit_section("Cross: parallel scaling", bench_parallel_scaling());

            print_roofline(results);
            print_section_peaks(results);

            if (emit_csv) {
                std::cout << "\n── CSV ──────────────────────────────────────────────\n";
                std::cout << BenchResult::csv_header() << '\n';
                for (const auto& r : results) std::cout << r.to_csv() << '\n';
            };
        };
    };

};

int main(int argc, char** argv) {
    bool csv = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--csv") csv = true;

    std::cout << "linalg benchmark suite\n"
              << "Hardware threads : " << std::thread::hardware_concurrency() << '\n'
              << "sizeof(double)   : " << sizeof(double) << " B\n\n";

    linalg::bench::BenchSuite suite;
    suite.run_all(csv);
    return 0;
};

// g++ -std=c++20 -march=native -mtune=native -O3 -ffast-math -funroll-loops -ftree-vectorize -Iinclude benchmark/benchmark.cpp -o main