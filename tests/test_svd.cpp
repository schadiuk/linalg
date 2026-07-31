#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_bidiag_impl(size_t m, size_t n) {
    XorShift64 rng(4000 + m * 100 + n);
    auto A = random_matrix<T, L>(m, n, rng);

    BidiagResult<T, L> res = bidiag(A, true);
    const size_t k = std::min(m, n);
    EXPECT(res.d.size() == k);
    EXPECT(res.e.size() == (k > 0 ? k - 1 : 0));
    EXPECT(res.U.rows() == m && res.U.cols() == k);
    EXPECT(res.V.rows() == n && res.V.cols() == k);
    EXPECT(is_orthonormal_cols(res.U, 1e-6));
    EXPECT(is_orthonormal_cols(res.V, 1e-6));

    Matrix<T, L> B(k, k, T(0));
    for (size_t i = 0; i < k; ++i) B(i, i) = T(res.d[i]);
    for (size_t i = 0; i + 1 < k; ++i) B(i, i + 1) = T(res.e[i]);
    Matrix<T, L> UB(m, k, T(0)), UBV(m, n, T(0));
    gemm(T(1), expr(res.U), expr(B), T(0), UB);
    gemm(T(1), expr(UB), hermitian(res.V), T(0), UBV);
    EXPECT(residual_small(A, UBV, 1e-6));

    BidiagResult<T, L> res2 = bidiag(A, false);
    EXPECT(res2.U.rows() == 0 && res2.V.rows() == 0);
    for (size_t i = 0; i < k; ++i) EXPECT_NEAR(res2.d[i], res.d[i], 1e-6);

    BidiagResult<T, L> res3 = bidiag<T, L>(expr(A), true);
    for (size_t i = 0; i < k; ++i) EXPECT_NEAR(res3.d[i], res.d[i], 1e-6);
};

template<typename T, Layout L>
void t_svd_impl(size_t m, size_t n) {
    XorShift64 rng(4500 + m * 100 + n);
    auto A = random_matrix<T, L>(m, n, rng);

    SVDResult<T, L> res = svd(A);
    const size_t k = std::min(m, n);
    EXPECT(res.s.size() == k);
    EXPECT(res.U.cols() == k && res.V.cols() == k);

    for (size_t i = 0; i < k; ++i) EXPECT(res.s[i] >= -1e-9);
    for (size_t i = 0; i + 1 < k; ++i) EXPECT(res.s[i] >= res.s[i + 1] - 1e-9);
    EXPECT(is_orthonormal_cols(res.U, 1e-6));
    EXPECT(is_orthonormal_cols(res.V, 1e-6));

    Matrix<T, L> S(k, k, T(0));
    for (size_t i = 0; i < k; ++i) S(i, i) = T(res.s[i]);
    Matrix<T, L> US(m, k, T(0)), USV(m, n, T(0));
    gemm(T(1), expr(res.U), expr(S), T(0), US);
    gemm(T(1), expr(US), hermitian(res.V), T(0), USV);
    EXPECT(residual_small(A, USV, 1e-6));

    SVDResult<T, L> res2 = svd<T, L>(expr(A));
    for (size_t i = 0; i < k; ++i) EXPECT_NEAR(res2.s[i], res.s[i], 1e-6);
};

void t_dqds() {
    // Diagonal matrix: singular values are the absolute diagonal entries.
    Vector<double> d{ 3.0, -1.0, 2.0, 5.0 };
    Vector<double> e{ 0.0, 0.0, 0.0 };
    Vector<double> s = dqds(d, e);
    EXPECT(s.size() == 4);
    std::vector<double> expected{ 5.0, 3.0, 2.0, 1.0 };
    for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(s[i], expected[i], 1e-8);

    Vector<double> d0(0), e0(0);
    Vector<double> s0 = dqds(d0, e0);
    EXPECT(s0.size() == 0);
    Vector<double> d1{ -7.0 }, e1(0);
    Vector<double> s1 = dqds(d1, e1);
    EXPECT(s1.size() == 1 && std::abs(s1[0] - 7.0) < 1e-9);

    // Cross-check against full bidiagonal SVD on a random bidiagonal matrix.
    XorShift64 rng(4600);
    const size_t n = 8;
    Vector<double> dd(n), ee(n - 1);
    for (size_t i = 0; i < n; ++i) dd[i] = rng.uniform(0.1, 5.0);
    for (size_t i = 0; i < n - 1; ++i) ee[i] = rng.uniform(-2.0, 2.0);
    Vector<double> sv = dqds(dd, ee);
    auto Bd = bidiagonal<double>(dd, ee, true);
    SVDResult<double, Layout::RowMajor> full = svd(Bd);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(sv[i], full.s[i], 1e-6);
};

void run_bidiag_svd_tests() {
    RUN_TEST((std::bind(t_bidiag_impl<double, Layout::RowMajor>, 8, 5)));
    RUN_TEST((std::bind(t_bidiag_impl<double, Layout::RowMajor>, 80, 70)));
    RUN_TEST((std::bind(t_bidiag_impl<double, Layout::RowMajor>, 40, 80))); // Wide -> A^H path.
    RUN_TEST((std::bind(t_bidiag_impl<double, Layout::ColMajor>, 8, 5)));
    RUN_TEST((std::bind(t_bidiag_impl<double, Layout::ColMajor>, 80, 70)));

    RUN_TEST((std::bind(t_bidiag_impl<std::complex<double>, Layout::RowMajor>, 8, 5)));
    RUN_TEST((std::bind(t_bidiag_impl<std::complex<double>, Layout::RowMajor>, 80, 70)));
    RUN_TEST((std::bind(t_bidiag_impl<std::complex<double>, Layout::RowMajor>, 40, 80)));
    RUN_TEST((std::bind(t_bidiag_impl<std::complex<double>, Layout::ColMajor>, 8, 5)));
    RUN_TEST((std::bind(t_bidiag_impl<std::complex<double>, Layout::ColMajor>, 80, 70)));

    RUN_TEST((std::bind(t_svd_impl<double, Layout::RowMajor>, 8, 5)));
    RUN_TEST((std::bind(t_svd_impl<double, Layout::RowMajor>, 80, 70)));
    RUN_TEST((std::bind(t_svd_impl<double, Layout::RowMajor>, 40, 80)));
    RUN_TEST((std::bind(t_svd_impl<double, Layout::ColMajor>, 40, 80)));
    RUN_TEST((std::bind(t_svd_impl<std::complex<double>, Layout::RowMajor>, 8, 5)));
    RUN_TEST((std::bind(t_svd_impl<std::complex<double>, Layout::RowMajor>, 50, 40)));
    RUN_TEST((std::bind(t_svd_impl<std::complex<double>, Layout::ColMajor>, 50, 40)));

    RUN_TEST(t_dqds);
};