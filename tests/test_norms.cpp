#include <harness.hpp>

using namespace linalg;
using namespace test;

void t_vector_norms() {
    Vector<double> v{ 0.0, -3.0, 4.0, 0.0 };
    EXPECT_NEAR(norm_l0(expr(v)), 2.0, 1e-12);
    EXPECT_NEAR(norm_l1(expr(v)), 7.0, 1e-12);
    EXPECT_NEAR(norm_l2(expr(v)), 5.0, 1e-9);
    EXPECT_NEAR(norm_inf(expr(v)), 4.0, 1e-12);
    EXPECT_NEAR(norm_neg_inf(expr(v)), 0.0, 1e-12);

    EXPECT_NEAR(norm(expr(v), "0"), 2.0, 1e-12);
    EXPECT_NEAR(norm(expr(v), "1"), 7.0, 1e-12);
    EXPECT_NEAR(norm(expr(v), "2"), 5.0, 1e-9);
    EXPECT_NEAR(norm(expr(v), "fro"), 5.0, 1e-9);
    EXPECT_NEAR(norm(expr(v)), 5.0, 1e-9);
    EXPECT_NEAR(norm(expr(v), "inf"), 4.0, 1e-12);
    EXPECT_NEAR(norm(expr(v), "-inf"), 0.0, 1e-12);
    bool threw = false;
    try { norm(expr(v), "bogus"); } catch (const std::invalid_argument&) { threw = true; };
    EXPECT(threw);

    Vector<std::complex<double>> cv{ {3,4}, {0,0} };
    EXPECT_NEAR(norm_l2(expr(cv)), 5.0, 1e-9);
};

template<typename T, Layout L>
void t_matrix_norms() {
    Matrix<T, L> A(2, 2, T(0));
    A(0, 0) = T(1); A(0, 1) = T(-2); A(1, 0) = T(3); A(1, 1) = T(4);
    double l1 = norm_l1(expr(A));
    EXPECT_NEAR(l1, 6.0, 1e-9);
    double linf = norm_inf(expr(A));
    EXPECT_NEAR(linf, 7.0, 1e-9);
    double lninf = norm_neg_inf(expr(A));
    EXPECT_NEAR(lninf, 3.0, 1e-9);
    double fro = norm_fro(expr(A));
    EXPECT_NEAR(fro, std::sqrt(1.0 + 4.0 + 9.0 + 16.0), 1e-9);
    double l2 = norm_l2(expr(A));
    EXPECT(l2 > 0.0 && l2 <= fro + 1e-6);

    EXPECT_NEAR(norm(expr(A), "1"), l1, 1e-9);
    EXPECT_NEAR(norm(expr(A), "fro"), fro, 1e-9);
    EXPECT_NEAR(norm(expr(A)), fro, 1e-9);
    EXPECT_NEAR(norm(expr(A), "inf"), linf, 1e-9);
    EXPECT_NEAR(norm(expr(A), "-inf"), lninf, 1e-9);
    EXPECT_NEAR(norm(expr(A), "2"), l2, 1e-6);
    bool threw = false;
    try { norm(expr(A), "bogus"); } catch (const std::invalid_argument&) { threw = true; };
    EXPECT(threw);
    // Should approximate the largest singular value for a simple diagonal case.
    Matrix<T, L> D(3, 3, T(0));
    D(0, 0) = T(5); D(1, 1) = T(2); D(2, 2) = T(1);
    EXPECT_NEAR(norm_l2(expr(D)), 5.0, 1e-6);
};

void run_norms_tests() {
    RUN_TEST(t_vector_norms);
    RUN_TEST((t_matrix_norms<double, Layout::RowMajor>));
    RUN_TEST((t_matrix_norms<double, Layout::ColMajor>));
    RUN_TEST((t_matrix_norms<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_matrix_norms<std::complex<double>, Layout::ColMajor>));
};