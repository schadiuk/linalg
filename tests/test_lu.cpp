#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_lu_impl(size_t n) {
    XorShift64 rng(1000 + n);
    auto A = random_matrix<T, L>(n, n, rng);
    for (size_t i = 0; i < n; ++i) A(i,i) += T(static_cast<double>(n)); // Well-conditioned.

    LUResult<T, L> res = lu(A);
    EXPECT(res.P.rows() == n && res.L.rows() == n && res.U.cols() == n);

    Matrix<T, L> PA(n, n, T(0)), LU_(n, n, T(0));
    gemm(T(1), expr(res.P), expr(A), T(0), PA);
    gemm(T(1), expr(res.L), expr(res.U), T(0), LU_);
    EXPECT(residual_small(PA, LU_, 1e-6));

    Vector<T> x_true = random_vector<T>(n, rng);
    Vector<T> b(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for(size_t j = 0; j < n; ++j) s += A(i, j) * x_true[j]; b[i] = s; };
    Vector<T> sol = b;
    lu_solve(res, sol);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol[i] - x_true[i]), 0.0, 1e-6);

    // Matrix (multi-RHS) solve.
    Matrix<T, L> X_true = random_matrix<T, L>(n, 3, rng);
    Matrix<T, L> B(n, 3, T(0));
    gemm(T(1), expr(A), expr(X_true), T(0), B);
    Matrix<T, L> Bsol = B;
    lu_solve(res, Bsol);
    EXPECT(residual_small(Bsol, X_true, 1e-6));

    T d = lu_det(res);
    EXPECT(std::abs(d) > 0.0); // Well-conditioned diag-dominant matrix is nonsingular.

    // Inverse: A * inv(A) == I.
    Matrix<T, L> Inv = lu_inverse(res);
    Matrix<T, L> Prod(n, n, T(0));
    gemm(T(1), expr(A), expr(Inv), T(0), Prod);
    EXPECT(residual_small(Prod, Matrix<T,L>::identity(n), 1e-5));

    LUResult<T, L> res2 = lu<T, L>(expr(A));
    EXPECT(residual_small(res2.U, res.U, 1e-9));
};

template<typename T, Layout L>
void t_lu_singular() {
    const size_t n = 4;
    Matrix<T, L> A(n, n, T(0));
    XorShift64 rng(555);
    for (size_t j = 0; j < n; ++j) A(0, j) = rand_scalar<T>(rng);
    for (size_t j = 0; j < n; ++j) A(1, j) = rand_scalar<T>(rng);
    for (size_t j = 0; j < n; ++j) A(2, j) = T(2) * A(0, j);
    for (size_t j = 0; j < n; ++j) A(3, j) = rand_scalar<T>(rng);
    LUResult<T, L> res = lu(A);
    T d = lu_det(res);
    EXPECT_NEAR(std::abs(d), 0.0, 1e-6);
};

void run_lu_tests() {
    // Sizes straddling LU_BLOCK = 64.
    RUN_TEST((std::bind(t_lu_impl<double, Layout::RowMajor>, 5)));
    RUN_TEST((std::bind(t_lu_impl<double, Layout::RowMajor>, 80)));
    RUN_TEST((std::bind(t_lu_impl<double, Layout::ColMajor>, 5)));
    RUN_TEST((std::bind(t_lu_impl<double, Layout::ColMajor>, 80)));
    RUN_TEST((std::bind(t_lu_impl<std::complex<double>, Layout::RowMajor>, 5)));
    RUN_TEST((std::bind(t_lu_impl<std::complex<double>, Layout::RowMajor>, 70)));
    RUN_TEST((std::bind(t_lu_impl<std::complex<double>, Layout::ColMajor>, 5)));
    RUN_TEST((std::bind(t_lu_impl<std::complex<double>, Layout::ColMajor>, 70)));
    RUN_TEST((t_lu_singular<double, Layout::RowMajor>));
    RUN_TEST((t_lu_singular<std::complex<double>, Layout::ColMajor>));
};