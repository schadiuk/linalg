#include <linalg.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace linalg::bench {
    using Clock = std::chrono::high_resolution_clock;
    using Seconds = std::chrono::duration<double>;
    using Matrix_d = Matrix<double>;
    using Vector_d = Vector<double>;
    using Matrix_cd = Matrix<std::complex<double>>;
    using Vector_cd = Vector<std::complex<double>>;
        
    struct BenchResult {
        std::string name;
        size_t problem_size; // characteristic N
        size_t threads;
        double median_sec;
        double iqr_sec; // inter-quartile range — spread indicator
        double gflops; // 0 if not computable
        double bandwidth_GBs; // 0 if not computable
        std::string  notes;
    
        // CSV header
        static std::string csv_header() {
            return "name,N,threads,median_s,iqr_s,gflops,bandwidth_GBs,notes";
        };
    
        std::string to_csv() const {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6)
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
                << std::left  << std::setw(42) << name
                << std::right << std::setw(7)  << problem_size
                << "  threads=" << std::setw(2) << threads
                << std::fixed << std::setprecision(3)
                << "  t="    << std::setw(8) << median_sec << "s"
                << "  iqr="  << std::setw(8) << iqr_sec << "s";
            if (gflops > 0)
                std::cout << "  " << std::setw(8) << gflops << " GFLOPS";
            if (bandwidth_GBs > 0)
                std::cout << "  " << std::setw(7) << bandwidth_GBs << " GB/s";
            if (!notes.empty())
                std::cout << "  [" << notes << "]";
            std::cout << '\n';
        };
    };
    
    // Runs "kernel" "reps" times, discards warmup_reps, returns median/IQR.
    template<typename F>
    double time_median(F&& kernel, int reps = 11, int warmup_reps = 2) {
        // Warmup: fills caches, initialises thread pool
        for (int i = 0; i < warmup_reps; ++i) kernel();
        std::vector<double> times(reps);
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            kernel();
            times[i] = Seconds(Clock::now() - t0).count();
        };
        std::sort(times.begin(), times.end());
        return times[reps / 2];
    };
    
    template<typename F>
    std::pair<double,double> time_stats(F&& kernel, int reps = 11, int warmup = 2) {
        for (int i = 0; i < warmup; ++i) kernel();
        std::vector<double> t(reps);
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            kernel();
            t[i] = Seconds(Clock::now() - t0).count();
        };
        std::sort(t.begin(), t.end());
        double med = t[reps/2];
        double iqr = t[3*reps/4] - t[reps/4];
        return {med, iqr};
    }
    
    // Cache-tier size constants
    // These define the "interesting" problem sizes that probe each cache level
    // Number of doubles that fit in each cache tier (single array)
    // L1 32 KB: 4096 doubles
    // L2 256 KB: 32768 doubles
    // L3 8 MB: 1048576 doubles
    // DRAM: beyond L3
    static const std::vector<size_t> VEC_SIZES = {
        64,          // register-resident
        512,         // small L1
        4096,        // L1 boundary  (~32 KB for double)
        16384,       // deep L2
        65536,       // L2->L3 boundary
        262144,      // L3
        1048576,     // large L3 / near DRAM boundary
        4194304,     // DRAM
    };
    
    // Matrix side lengths N such that N×N double matrix fits in each tier
    static const std::vector<size_t> MAT_SIZES = {
        32,    // fits registers/L1
        64,    // L1
        128,   // L2
        256,   // L2->L3
        512,   // L3
        1024,  // large L3
        2048,  // near DRAM
        4096,  // DRAM-pressure (128MB)
    };
    
    //  All are memory-bandwidth bound. Report GB/s, not GFLOPS.
    //  Bandwidth = bytes_read + bytes_written per iteration.
    
    // axpy: y = alpha*x + y
    // Reads x (8N bytes), reads+writes y (16N bytes) -> 24N bytes
    std::vector<BenchResult> bench_axpy(int reps = 11) {
        std::vector<BenchResult> results;
        const double alpha = 1.5;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ axpy(alpha, expr(x), y); }, reps);
            double bytes = double(N) * sizeof(double) * 3; // read x, read+write y
            BenchResult r;
            r.name = "axpy<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = (2.0 * N) / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            results.push_back(r);
        };
        return results;
    };

    // Reads x and y (16N bytes), pure reduction
    std::vector<BenchResult> bench_dot(int reps = 11) {
        std::vector<BenchResult> results;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ volatile auto s = dot(expr(x), expr(y)); (void)s; }, reps);
            double bytes = double(N) * sizeof(double) * 2;
            BenchResult r;
            r.name = "dot<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = (2.0 * N) / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            results.push_back(r);
        };
        return results;
    };

    // Reads x (8N bytes)
    std::vector<BenchResult> bench_norm(int reps = 11) {
        std::vector<BenchResult> results;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ volatile double n = norm(x); (void)n; }, reps);
            double bytes = double(N) * sizeof(double);
            BenchResult r;
            r.name = "norm<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = (2.0 * N) / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            results.push_back(r);
        };
        return results;
    };
    // scal: x *= alpha  (read+write x, 16N bytes)
    std::vector<BenchResult> bench_scal(int reps = 11) {
        std::vector<BenchResult> results;
        const double alpha = 0.99;
        for (size_t N : VEC_SIZES) {
            auto x = Vector_d::random(N);
            auto [med, iqr] = time_stats([&]{ scal(alpha, x); }, reps);
            double bytes = double(N) * sizeof(double) * 2;
            BenchResult r;
            r.name = "scal<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = double(N) / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            results.push_back(r);
        };;
        return results;
    }
    
    // gemv: y = alpha*A*x + beta*y
    // FLOPs = 2·M·N;  bytes = 8MN (read A) + 8N (read x) + 16M (r/w y)
    std::vector<BenchResult> bench_gemv(int reps = 11) {
        std::vector<BenchResult> results;
        const double alpha = 1.0, beta = 0.5;
        for (size_t N : {32u,64u,128u,256u,512u,1024u,2048u,4096u}) {
            auto A = Matrix_d::random(N, N);
            auto x = Vector_d::random(N);
            auto y = Vector_d(N, 0.0);
            auto [med, iqr] = time_stats([&]{ gemv(alpha, expr(A), expr(x), beta, y); }, reps);
            double flops = 2.0 * N * N;
            double bytes = double(N)*double(N) * sizeof(double)   // read A
                        + double(N) * sizeof(double)              // read x
                        + double(N) * sizeof(double) * 2;         // r/w y
            BenchResult r;
            r.name = "gemv<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9;
            results.push_back(r);
        };
        return results;
    };
        
    // Square DGEMM: C = alpha*A*B + beta*C
    std::vector<BenchResult> bench_gemm_square(int reps = 7) {
        std::vector<BenchResult> results;
        for (size_t N : MAT_SIZES) {
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0 * double(N)*double(N)*double(N);
            double bytes = 3.0 * double(N)*double(N) * sizeof(double);
            BenchResult r;
            r.name = "gemm_square<double>";
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = bytes / med / 1e9; // lower bound (cache-cold is higher)
            results.push_back(r);
        };
        return results;
    };
    
    // Non-square (tall-skinny) GEMM: A is M*K, B is K*N with K<<M
    std::vector<BenchResult> bench_gemm_tall_skinny(int reps = 9) {
        std::vector<BenchResult> results;
        // Fixed N=2048, vary K=1,4,16,64,256,2048
        const size_t M = 2048;
        for (size_t K : {1u, 4u, 16u, 64u, 256u, 512u, 2048u}) {
            size_t N_col = M;
            auto A = Matrix_d::random(M, K);
            auto B = Matrix_d::random(K, N_col);
            auto C = Matrix_d::zeros(M, N_col);
            gemm(0.5, expr(A), expr(B), 0.5, C); // warmup
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0 * double(M) * double(N_col) * double(K);
            BenchResult r;
            r.name = "gemm_tall_skinny K=" + std::to_string(K);
            r.problem_size = M;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = 0;
            r.notes = "M=N=" + std::to_string(M) + " K=" + std::to_string(K);
            results.push_back(r);
        };
        return results;
    };
    
    // Transposed GEMM  (A^T * B, A * B^T, A^T * B^T)
    // Non-contiguous access patterns expose stride penalties
    std::vector<BenchResult> bench_gemm_transposed(int reps = 7) {
        std::vector<BenchResult> results;
        const size_t N = 1024;
        auto A = Matrix_d::random(N, N);
        auto B = Matrix_d::random(N, N);
        auto C = Matrix_d::zeros(N, N);
        double flops = 2.0 * double(N)*double(N)*double(N);
        struct Case { std::string name; std::function<void()> fn; };
        std::vector<Case> cases = {
            {"gemm NN", [&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }},
            {"gemm TN", [&]{ gemm(0.5, expr(transpose(A)), expr(B), 0.5, C); }},
            {"gemm NT", [&]{ gemm(0.5, expr(A), expr(transpose(B)), 0.5, C); }},
            {"gemm TT", [&]{ gemm(0.5, expr(transpose(A)), expr(transpose(B)), 0.5, C); }},
            {"gemm HN", [&]{ gemm(0.5, expr(hermitian(A)), expr(B), 0.5, C); }},
        };
        for (auto& c : cases) {
            c.fn(); // warmup
            auto [med, iqr] = time_stats(c.fn, reps);
            BenchResult r;
            r.name = c.name + "<double> N=" + std::to_string(N);
            r.problem_size = N;
            r.threads = std::thread::hardware_concurrency();
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = 0;
            results.push_back(r);
        };
        return results;
    };
  
    // Strong-scaling: fixed work (N=2048 GEMM), measure time vs expected serial.
    // We compare against a N=256 serial-forced result (below PARALLEL_THRESHOLD)
    // extrapolated to N=2048 - this gives an upper bound on parallel efficiency
    std::vector<BenchResult> bench_parallel_scaling(int reps = 7) {
        std::vector<BenchResult> results;
        const size_t P = std::thread::hardware_concurrency();
        // Establish serial baseline at N=256 (below PARALLEL_THRESHOLD_COMPUTE)
        {
            size_t N = 256;
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            BenchResult r;
            r.name  = "gemm_scaling_serial_baseline";
            r.problem_size = N;
            r.threads = 1;
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = 2.0*double(N)*double(N)*double(N) / med / 1e9;
            r.bandwidth_GBs = 0;
            r.notes = "serial extrapolation base";
            results.push_back(r);
        };
    
        // Parallel measurements across problem sizes
        for (size_t N : {512u, 1024u, 2048u, 4096u}) {
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med, iqr] = time_stats([&]{ gemm(0.5, expr(A), expr(B), 0.5, C); }, reps);
            double flops = 2.0*double(N)*double(N)*double(N);
            BenchResult r;
            r.name = "gemm_parallel N=" + std::to_string(N);
            r.problem_size = N;
            r.threads = P;
            r.median_sec = med;
            r.iqr_sec = iqr;
            r.gflops = flops / med / 1e9;
            r.bandwidth_GBs = 0;
            results.push_back(r);
        };
        return results;
    };
    
    // Compares float, double, complex<double> at identical problem size    
    std::vector<BenchResult> bench_dtype_comparison(int reps = 9) {
        std::vector<BenchResult> results;
        const size_t N = 512;
        // double
        {
            auto A = Matrix_d::random(N,N);
            auto B = Matrix_d::random(N,N);
            auto C = Matrix_d::zeros(N,N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
            BenchResult r; r.name="gemm<double>"; r.problem_size=N;
            r.threads=std::thread::hardware_concurrency(); r.median_sec=med; r.iqr_sec=iqr;
            r.gflops=2.0*double(N)*double(N)*double(N)/med/1e9; r.bandwidth_GBs=0;
            results.push_back(r);
        };
        // float
        {
            using Mf = Matrix<float>;
            auto A = Mf::random(N,N); auto B = Mf::random(N,N); auto C = Mf::zeros(N,N);
            gemm(0.5f, expr(A), expr(B), 0.5f, C);
            auto [med,iqr] = time_stats([&]{ gemm(0.5f,expr(A),expr(B),0.5f,C); }, reps);
            BenchResult r; r.name="gemm<float>"; r.problem_size=N;
            r.threads=std::thread::hardware_concurrency(); r.median_sec=med; r.iqr_sec=iqr;
            r.gflops=2.0*double(N)*double(N)*double(N)/med/1e9; r.bandwidth_GBs=0;
            r.notes="expect ~2 times faster than double (AVX2 = 8 floats vs 4 doubles)";
            results.push_back(r);
        };
        // complex<double>
        {
            auto A = Matrix_cd::random(N,N); auto B = Matrix_cd::random(N,N);
            auto C = Matrix_cd::zeros(N,N);
            std::complex<double> a05(0.5,0.), b05(0.5,0.);
            gemm(a05, expr(A), expr(B), b05, C);
            auto [med,iqr] = time_stats([&]{ gemm(a05,expr(A),expr(B),b05,C); }, reps);
            // complex GEMM: 8 real FLOPs per element (Karatsuba or naive 4+2)
            double flops = 8.0*double(N)*double(N)*double(N);
            BenchResult r; r.name="gemm<complex<double>>"; r.problem_size=N;
            r.threads=std::thread::hardware_concurrency(); r.median_sec=med; r.iqr_sec=iqr;
            r.gflops=flops/med/1e9; r.bandwidth_GBs=0;
            r.notes="8 real FLOP per cmul";
            results.push_back(r);
        };
        return results;
    };
    
    // Compares RowMajor vs ColMajor at various sizes.
    // For cache-blocked gemm_direct the difference should be small.
    // For expression-template paths it reveals stride penalties.
    std::vector<BenchResult> bench_layout(int reps = 9) {
        std::vector<BenchResult> results;
        for (size_t N : {256u, 512u, 1024u, 2048u}) {
            using MR = Matrix<double, Layout::RowMajor>;
            using MC = Matrix<double, Layout::ColMajor>;
            {
                auto A=MR::random(N,N), B=MR::random(N,N), C=MR::zeros(N,N);
                gemm(0.5,expr(A),expr(B),0.5,C);
                auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
                BenchResult r; r.name="gemm RowMajor N="+std::to_string(N);
                r.problem_size=N; r.threads=std::thread::hardware_concurrency();
                r.median_sec=med; r.iqr_sec=iqr;
                r.gflops=2.0*double(N)*double(N)*double(N)/med/1e9;
                r.bandwidth_GBs=0; results.push_back(r);
            };
            {
                auto A=MC::random(N,N), B=MC::random(N,N), C=MC::zeros(N,N);
                gemm(0.5,expr(A),expr(B),0.5,C);
                auto [med,iqr] = time_stats([&]{ gemm(0.5,expr(A),expr(B),0.5,C); }, reps);
                BenchResult r; r.name="gemm ColMajor N="+std::to_string(N);
                r.problem_size=N; r.threads=std::thread::hardware_concurrency();
                r.median_sec=med; r.iqr_sec=iqr;
                r.gflops=2.0*double(N)*double(N)*double(N)/med/1e9;
                r.bandwidth_GBs=0; results.push_back(r);
            };
        };
        return results;
    };

    // Compares: Matrix assignment from an ET chain vs. a raw loop.
    std::vector<BenchResult> bench_expr_overhead(int reps = 11) {
        std::vector<BenchResult> results;
        for (size_t N : {1024u, 16384u, 65536u, 1u<<20}) {
            auto x = Vector_d::random(N);
            auto y = Vector_d::random(N);
            auto z = Vector_d::random(N);
            Vector_d out(N);
            // ET chain: out = x + 2.0*y - z
            auto [med_et, iqr_et] = time_stats([&]{
                out = expr(x) + 2.0 * expr(y) - expr(z);
            }, reps);
            // Raw loop baseline
            auto [med_raw, iqr_raw] = time_stats([&]{
                for (size_t i = 0; i < N; ++i) out[i] = x[i] + 2.0*y[i] - z[i];
            }, reps);
            double bytes = double(N) * sizeof(double) * 4; // read x,y,z + write out
            {
                BenchResult r; r.name = "expr_chain(x+2y-z) ET N="+std::to_string(N);
                r.problem_size=N; r.threads=1; r.median_sec=med_et; r.iqr_sec=iqr_et;
                r.gflops=3.0*double(N)/med_et/1e9; r.bandwidth_GBs=bytes/med_et/1e9;
                results.push_back(r);
            };
            {
                BenchResult r; r.name = "expr_chain(x+2y-z) raw N="+std::to_string(N);
                r.problem_size=N; r.threads=1; r.median_sec=med_raw; r.iqr_sec=iqr_raw;
                r.gflops=3.0*double(N)/med_raw/1e9; r.bandwidth_GBs=bytes/med_raw/1e9;
                r.notes="baseline: raw loop";
                results.push_back(r);
            };
        };
        return results;
    };
    
    // Aliasing detection cost
    std::vector<BenchResult> bench_aliasing_check(int reps = 15) {
        std::vector<BenchResult> results;
        for (size_t N : {64u, 256u, 1024u, 4096u}) {
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            // No aliasing: depends_on returns false, no temp allocation
            auto [med_na, iqr_na] = time_stats([&]{
                C = expr(A) + expr(B);
            }, reps);
            // Aliasing: assign back to A (forces temp buffer)
            auto [med_a, iqr_a] = time_stats([&]{
                A = expr(A) + expr(B);
            }, reps);
            double bytes = 3.0*double(N)*double(N)*sizeof(double);
            {
                BenchResult r; r.name="alias_check no-alias N="+std::to_string(N);
                r.problem_size=N*N; r.threads=1; r.median_sec=med_na; r.iqr_sec=iqr_na;
                r.gflops=0; r.bandwidth_GBs=bytes/med_na/1e9;
                results.push_back(r);
            };
            {
                BenchResult r; r.name="alias_check aliased  N="+std::to_string(N);
                r.problem_size=N*N; r.threads=1; r.median_sec=med_a; r.iqr_sec=iqr_a;
                r.gflops=0; r.bandwidth_GBs=bytes/med_a/1e9;
                r.notes="overhead = temp alloc + extra copy";
                results.push_back(r);
            };
        };
        return results;
    };
    
    void print_roofline_summary(const std::vector<BenchResult>& all_results) {
        // Peak BW: max bandwidth seen in axpy at large N
        double peak_bw = 0;
        double peak_gflops = 0;
    
        for (const auto& r : all_results) {
            if (r.name.find("axpy") != std::string::npos && r.problem_size >= 1<<20)
                peak_bw = std::max(peak_bw, r.bandwidth_GBs);
            if (r.name.find("gemm_square") != std::string::npos && r.problem_size >= 2048)
                peak_gflops = std::max(peak_gflops, r.gflops);
        };
        std::cout << std::fixed << std::setprecision(1);
        std::cout <<   "Peak memory bandwidth  : "
                << std::setw(7) << peak_bw      << " GB/s     \n";
        std::cout <<   "Peak GEMM (large N)    : "
                << std::setw(7) << peak_gflops  << " GFLOPS   \n";
        if (peak_bw > 0)
            std::cout << "Ridge point (FI)       : "
                    << std::setw(7) << peak_gflops/peak_bw << " FLOP/byte\n";
    };

    struct BenchSuite {
        std::vector<BenchResult> results;
        void run_all(bool csv_output = true) {
            auto run = [&](auto fn, const char* section) {
                std::cout << "\n── " << section << " ──────────────\n";
                auto v = fn();
                for (auto& r : v) { r.print(); results.push_back(r); }
            };
            run([]{return bench_axpy();}, "Level-1: axpy");
            run([]{return bench_dot();}, "Level-1: dot");
            run([]{return bench_norm();}, "Level-1: norm");
            run([]{return bench_scal();}, "Level-1: scal");
            run([]{return bench_gemv();}, "Level-2: gemv");
            run([]{return bench_gemm_square();}, "Level-3: gemm square");
            run([]{return bench_gemm_tall_skinny();}, "Level-3: gemm tall-skinny");
            run([]{return bench_gemm_transposed();}, "Level-3: gemm transposed");
            run([]{return bench_parallel_scaling();}, "Parallel scaling");
            run([]{return bench_dtype_comparison();}, "Dtype comparison");
            run([]{return bench_layout();}, "Layout comparison");
            run([]{return bench_expr_overhead();}, "Expression-tree overhead");
            run([]{return bench_aliasing_check();}, "Aliasing detection cost");
            print_roofline_summary(results);
            if (csv_output) emit_csv("linalg_bench.csv");
        };
    
        void emit_csv(const std::string& path) const {
            std::cout << "\n# CSV output — redirect to " << path << "\n";
            std::cout << BenchResult::csv_header() << '\n';
            for (const auto& r : results) std::cout << r.to_csv() << '\n';
        };
    
        // Mini benchmark for comparison
        void run_mini(size_t N = 4096, int reps = 3) {
            std::cout << "\n── Original mini benchmark (for comparison) ────\n";
            auto A = Matrix_d::random(N, N);
            auto B = Matrix_d::random(N, N);
            auto C = Matrix_d::zeros(N, N);
            gemm(0.5, expr(A), expr(B), 0.5, C);
            auto t0 = Clock::now();
            for (int i = 0; i < reps; ++i)
                gemm(0.5, expr(A), expr(B), 0.5, C);
            double dur = Seconds(Clock::now() - t0).count() / reps;
    
            double gflops = 2.0*double(N)*double(N)*double(N) / dur / 1e9;
            std::cout << std::fixed << std::setprecision(3)
                    << "N=" << N << "  t=" << dur << "s  " << gflops << " GFLOPS\n";
        };
    };
 
};

int main() {
    linalg::bench::BenchSuite suite;
    suite.run_mini();   // reproduce the existing result
    suite.run_all(false);         // full suite + CSV
    return 0;
};