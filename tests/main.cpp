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

#include <chrono>
/*
void run_core_tests();
void run_expr_tests();
void run_vec_ops_tests();
void run_mat_ops_tests();
void run_constructors_tests();
void run_level1_tests();
void run_level2_tests();
void run_level3_tests();

void run_norms_tests();
void run_lu_tests();
void run_qr_tests();
void run_cholesky_tests();
void run_bidiag_svd_tests();
void run_schur_eig_tests();
void run_lstsq_tests();
void run_io_tests();
void run_numerical_stability_tests();
*/

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
/*    
    STAGE(run_lu_tests);
    STAGE(run_qr_tests);
    STAGE(run_cholesky_tests);
    STAGE(run_bidiag_svd_tests);
    STAGE(run_schur_eig_tests);
    STAGE(run_lstsq_tests);
    STAGE(run_io_tests);
    STAGE(run_numerical_stability_tests);
*/
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n==============================\n";
    std::cout << "Checks run : " << test::g_checks << "\n";
    std::cout << "Failures   : " << test::g_failures << "\n";
    std::cout << "Time       : " << secs << "s\n";
    std::cout << (test::g_failures == 0 ? "ALL PASSED\n" : "FAILURES PRESENT\n");
    return test::g_failures == 0 ? 0 : 1;
};