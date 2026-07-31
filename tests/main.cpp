#include <harness.hpp>
#include <test_core.cpp>
#include <test_expr.cpp>
#include <test_vec_ops.cpp>
#include <test_mat_ops.cpp>
#include <test_constructors.cpp>
#include <test_level1.cpp>
#include <test_level2.cpp>
#include <test_level3.cpp>
#include <test_norms.cpp>
#include <test_lu.cpp>
#include <test_cholesky.cpp>
#include <test_qr.cpp>
#include <test_svd.cpp>
#include <test_schur.cpp>
#include <test_lstsq.cpp>
#include <test_stability.cpp>

#include <chrono>

int main() {
    auto t0 = std::chrono::steady_clock::now();

    #define STAGE(fn) do { std::cout << "[stage] " #fn << std::flush; fn(); std::cout << " done (checks=" << test::g_checks << ")\n" << std::flush; } while(0)
    STAGE(run_core_tests);
    STAGE(run_expr_tests);
    STAGE(run_vec_ops_tests);
    STAGE(run_mat_ops_tests);
    STAGE(run_constructors_tests);
    STAGE(run_level1_tests);
    STAGE(run_level2_tests);
    STAGE(run_level3_tests);
    STAGE(run_norms_tests);
    STAGE(run_lu_tests);
    STAGE(run_cholesky_tests);
    STAGE(run_qr_tests);
    STAGE(run_bidiag_svd_tests);
    STAGE(run_schur_eig_tests);
    STAGE(run_lstsq_tests);
    STAGE(run_numerical_stability_tests);

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n==============================\n";
    std::cout << "Checks run : " << test::g_checks << "\n";
    std::cout << "Failures   : " << test::g_failures << "\n";
    std::cout << "Time       : " << secs << "s\n";
    std::cout << (test::g_failures == 0 ? "ALL PASSED\n" : "FAILURES PRESENT\n");
    return test::g_failures == 0 ? 0 : 1;
};

// g++ -std=c++20 -march=native -mtune=native -O3 -ffast-math -funroll-loops -ftree-vectorize -Iinclude -Itests tests/main.cpp -o main