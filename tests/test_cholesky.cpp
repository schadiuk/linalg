#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_cholesky_impl(size_t n) {
    XorShift64 rng(3000 + n);
    auto A = random_hpd<T, L>(n, rng);
    CholeskyResult<T, L> resL = potrf(A, 'L');
    EXPECT(resL.uplo == 'L');
    Matrix<T, L> LLt(n, n, T(0));
    gemm(T(1), expr(resL.factor), hermitian(resL.factor), T(0), LLt);
    EXPECT(residual_small(A, LLt, 1e-6));

    CholeskyResult<T, L> resU = potrf(A, 'U');
    EXPECT(resU.uplo == 'U');
    Matrix<T, L> UtU(n, n, T(0));
    gemm(T(1), hermitian(resU.factor), expr(resU.factor), T(0), UtU);
    EXPECT(residual_small(A, UtU, 1e-6));

    Vector<T> x_true = random_vector<T>(n, rng);
    Vector<T> b(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for(size_t j = 0; j < n; ++j) s += A(i, j) * x_true[j]; b[i] = s; };
    Vector<T> sol = b;
    potrs(resL, sol);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol[i] - x_true[i]), 0.0, 1e-6);

    Vector<T> solU = b;
    potrs(resU, solU);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(solU[i] - x_true[i]), 0.0, 1e-6);

    Matrix<T, L> X_true = random_matrix<T, L>(n, 3, rng);
    Matrix<T, L> B(n, 3, T(0));
    gemm(T(1), expr(A), expr(X_true), T(0), B);
    Matrix<T, L> Bsol = B;
    potrs(resL, Bsol);
    EXPECT(residual_small(Bsol, X_true, 1e-6));

    Matrix<T, L> Inv = potri(resL);
    Matrix<T, L> Prod(n, n, T(0));
    gemm(T(1), expr(A), expr(Inv), T(0), Prod);
    EXPECT(residual_small(Prod, Matrix<T,L>::identity(n), 1e-5));

    double ld = cholesky_logdet(resL);
    double d = cholesky_det(resL);
    EXPECT(d > 0.0);
    if (std::isfinite(d)) {
        EXPECT_NEAR(std::exp(ld), d, 1e-6 * std::max(1.0, d));
    } else {
        EXPECT(ld > 700.0); // Consistent with d overflowing exp().
    };

    CholeskyResult<T, L> resE = potrf<T, L>(expr(A));
    EXPECT(residual_small(resE.factor, resL.factor, 1e-9));
};

template<typename T, Layout L>
void t_cholesky_not_pd() {
    const size_t n = 3;
    Matrix<T, L> A(n, n, T(0));
    A(0,0) = T(1); A(1,1) = T(-2); A(2,2) = T(3);
    bool threw = false;
    try { potrf(A, 'L'); } catch (const std::runtime_error&) { threw = true; };
    EXPECT(threw);
};

void run_cholesky_tests() {
    RUN_TEST((std::bind(t_cholesky_impl<double, Layout::RowMajor>, 5)));
    RUN_TEST((std::bind(t_cholesky_impl<double, Layout::RowMajor>, 128)));
    RUN_TEST((std::bind(t_cholesky_impl<double, Layout::ColMajor>, 64)));
    RUN_TEST((std::bind(t_cholesky_impl<std::complex<double>, Layout::RowMajor>, 5)));
    RUN_TEST((std::bind(t_cholesky_impl<std::complex<double>, Layout::RowMajor>, 140)));
    RUN_TEST((std::bind(t_cholesky_impl<std::complex<double>, Layout::ColMajor>, 5)));
    RUN_TEST((t_cholesky_not_pd<double, Layout::RowMajor>));
    RUN_TEST((t_cholesky_not_pd<std::complex<double>, Layout::ColMajor>));
};