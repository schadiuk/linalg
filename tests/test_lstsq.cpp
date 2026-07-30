#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_lstsq_overdetermined(const std::string& driver) {
    XorShift64 rng(6000);
    const size_t m = 10, n = 4;
    auto A = random_matrix<T, L>(m, n, rng);
    Vector<T> x_true = random_vector<T>(n, rng);
    Vector<T> b(m, T(0));
    for (size_t i = 0; i < m; ++i) { T s = T(0); for(size_t j = 0; j < n; ++j) s += A(i, j) * x_true[j]; b[i] = s; };

    LstsqVecResult<T> res = lstsq(A, b, driver);
    EXPECT(res.rank == static_cast<int>(n));
    EXPECT_NEAR(res.residual, 0.0, 1e-6);
    for (size_t j = 0; j < n; ++j) EXPECT_NEAR(std::abs(res.x[j] - x_true[j]), 0.0, 1e-5);

    LstsqVecResult<T> resQ = lstsq_qr(A, b);
    LstsqVecResult<T> resS = lstsq_svd(A, b);
    for (size_t j = 0; j < n; ++j) {
        EXPECT_NEAR(std::abs(resQ.x[j] - x_true[j]), 0.0, 1e-5);
        EXPECT_NEAR(std::abs(resS.x[j] - x_true[j]), 0.0, 1e-5);
    };

    Matrix<T, L> X_true = random_matrix<T, L>(n, 3, rng);
    Matrix<T, L> B(m, 3, T(0));
    gemm(T(1), expr(A), expr(X_true), T(0), B);
    LstsqMatResult<T, L> mres = lstsq(A, B, driver);
    EXPECT(residual_small(mres.X, X_true, 1e-5));
    for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(mres.residuals[j], 0.0, 1e-6);

    LstsqMatResult<T, L> mresQ = lstsq_qr(A, B);
    LstsqMatResult<T, L> mresS = lstsq_svd(A, B);
    EXPECT(residual_small(mresQ.X, X_true, 1e-5));
    EXPECT(residual_small(mresS.X, X_true, 1e-5));

    LstsqVecResult<T> resE = lstsq_qr<T, L>(expr(A), expr(b));
    EXPECT_NEAR(std::abs(resE.x[0] - x_true[0]), 0.0, 1e-5);
    LstsqVecResult<T> resE2 = lstsq_svd<T, L>(expr(A), expr(b));
    EXPECT_NEAR(std::abs(resE2.x[0] - x_true[0]), 0.0, 1e-5);
    LstsqMatResult<T, L> mresE = lstsq_qr<T, L>(expr(A), expr(B));
    EXPECT(residual_small(mresE.X, X_true, 1e-5));
    LstsqMatResult<T, L> mresE2 = lstsq_svd<T, L>(expr(A), expr(B));
    EXPECT(residual_small(mresE2.X, X_true, 1e-5));
    LstsqVecResult<T> resGen = lstsq<T, L>(expr(A), expr(b), driver);
    EXPECT_NEAR(std::abs(resGen.x[0] - x_true[0]), 0.0, 1e-5);
    LstsqMatResult<T, L> mresGen = lstsq<T, L>(expr(A), expr(B), driver);
    EXPECT(residual_small(mresGen.X, X_true, 1e-5));
};

void t_lstsq_bad_driver() {
    XorShift64 rng(6100);
    auto A = random_matrix<double, Layout::RowMajor>(4, 2, rng);
    auto b = random_vector<double>(4, rng);
    bool threw = false;
    try { lstsq(A, b, "bogus"); } catch (const std::invalid_argument&) { threw = true; };
    EXPECT(threw);
};

template<typename T, Layout L>
void t_pinv() {
    XorShift64 rng(6200);
    const size_t m = 8, n = 5;
    auto A = random_matrix<T, L>(m, n, rng);
    Matrix<T, L> P = pinv(A);
    EXPECT(P.rows() == n && P.cols() == m);

    Matrix<T, L> AP(m, m, T(0)), APA(m, n, T(0));
    gemm(T(1), expr(A), expr(P), T(0), AP);
    gemm(T(1), expr(AP), expr(A), T(0), APA);
    EXPECT(residual_small(APA, A, 1e-5));

    Matrix<T, L> Pe = pinv<T, L>(expr(A));
    EXPECT(residual_small(Pe, P, 1e-9));
};

void run_lstsq_tests() {
    RUN_TEST((std::bind(t_lstsq_overdetermined<double, Layout::RowMajor>, "qr")));
    RUN_TEST((std::bind(t_lstsq_overdetermined<double, Layout::RowMajor>, "svd")));
    RUN_TEST((std::bind(t_lstsq_overdetermined<double, Layout::ColMajor>, "svd")));
    RUN_TEST((std::bind(t_lstsq_overdetermined<std::complex<double>, Layout::RowMajor>, "qr")));
    RUN_TEST((std::bind(t_lstsq_overdetermined<std::complex<double>, Layout::RowMajor>, "svd")));
    RUN_TEST((std::bind(t_lstsq_overdetermined<std::complex<double>, Layout::ColMajor>, "svd")));
    RUN_TEST(t_lstsq_bad_driver);
    RUN_TEST((t_pinv<double, Layout::RowMajor>));
    RUN_TEST((t_pinv<std::complex<double>, Layout::ColMajor>));
};