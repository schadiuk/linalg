#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_gemv() {
    XorShift64 rng(21);
    auto A = random_matrix<T, L>(5, 4, rng);
    auto x = random_vector<T>(4, rng);
    Vector<T> y = random_vector<T>(5, rng);
    Vector<T> y_ref = y;
    for (size_t i = 0; i < 5; ++i) {
        T s = T(0);
        for (size_t j = 0; j < 4; ++j) s += A(i,j) * x[j];
        y_ref[i] = T(2) * s + T(3) * y_ref[i];
    };
    gemv<T, L>(T(2), expr(A), expr(x), T(3), y);
    for (size_t i = 0; i < 5; ++i) EXPECT_NEAR(std::abs(y[i] - y_ref[i]), 0.0, 1e-9);

    Vector<T> y2 = random_vector<T>(5, rng);
    Vector<T> y2_ref = y2;
    for (size_t i = 0; i < 5; ++i) {
        T s = T(0);
        for (size_t j = 0; j < 4; ++j) s += A(i,j) * x[j];
        y2_ref[i] = T(1) * s + T(0) * y2_ref[i];
    };
    auto y2v = view(y2);
    gemv<T, L>(T(1), expr(A), expr(x), T(0), y2v);
    for (size_t i = 0; i < 5; ++i) EXPECT_NEAR(std::abs(y2[i] - y2_ref[i]), 0.0, 1e-9);
};

template<typename T, Layout L>
void t_ger_gerc() {
    XorShift64 rng(22);
    auto x = random_vector<T>(3, rng);
    auto y = random_vector<T>(4, rng);
    Matrix<T, L> A = random_matrix<T, L>(3, 4, rng);
    Matrix<T, L> A_ref = A;
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) A_ref(i,j) += T(2) * x[i] * y[j];
    ger(T(2), expr(x), expr(y), A);
    EXPECT(residual_small(A, A_ref, 1e-9));

    Matrix<T, L> B = random_matrix<T, L>(3, 4, rng);
    Matrix<T, L> B_ref = B;
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) B_ref(i,j) += T(2) * x[i] * conj(y[j]);
    gerc(T(2), expr(x), expr(y), B);
    EXPECT(residual_small(B, B_ref, 1e-9));

    // MatrixView overloads.
    Matrix<T, L> C = random_matrix<T, L>(3, 4, rng);
    Matrix<T, L> C_ref = C;
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) C_ref(i,j) += x[i] * y[j];
    auto Cv = view(C);
    ger(T(1), expr(x), expr(y), Cv);
    EXPECT(residual_small(C, C_ref, 1e-9));

    Matrix<T, L> D = random_matrix<T, L>(3, 4, rng);
    Matrix<T, L> D_ref = D;
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 4; ++j) D_ref(i,j) += x[i] * conj(y[j]);
    auto Dv = view(D);
    gerc(T(1), expr(x), expr(y), Dv);
    EXPECT(residual_small(D, D_ref, 1e-9));
};

template<typename T>
void t_trsv() {
    XorShift64 rng(23);
    const size_t n = 5;
    Matrix<T, Layout::RowMajor> L_(n, n, T(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) L_(i,j) = rand_scalar<T>(rng);
        L_(i,i) += T(static_cast<double>(n)); // Diag dominance for better conditioning.
    };
    Vector<T> x_true = random_vector<T>(n, rng);
    Vector<T> b(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t j = 0; j < n; ++j) s += L_(i, j) * x_true[j]; b[i] = s; };
    Vector<T> sol = b;
    trsv('L', 'N', 'N', expr(L_), sol);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol[i] - x_true[i]), 0.0, 1e-6);

    // Transposed solve: L^T * x = b.
    Vector<T> b2(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t j = 0; j < n; ++j) s += L_(j, i) * x_true[j]; b2[i] = s; };
    Vector<T> sol2 = b2;
    trsv('L', 'T', 'N', expr(L_), sol2);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol2[i] - x_true[i]), 0.0, 1e-6);

    // Conjugate-transposed solve.
    Vector<T> b3(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t j = 0; j < n; ++j) s += conj(L_(j, i)) * x_true[j]; b3[i] = s; };
    Vector<T> sol3 = b3;
    trsv('L', 'C', 'N', expr(L_), sol3);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol3[i] - x_true[i]), 0.0, 1e-6);

    // Upper triangular, unit diagonal.
    Matrix<T, Layout::RowMajor> U_(n, n, T(0));
    for (size_t i = 0; i < n; ++i) { U_(i, i) = T(1); for (size_t j = i + 1; j < n; ++j) U_(i, j) = rand_scalar<T>(rng) * T(0.1); };
    Vector<T> b4(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t j = 0; j<n; ++j) s += U_(i,j) * x_true[j]; b4[i] = s; };
    Vector<T> sol4 = b4;
    trsv('U', 'N', 'U', expr(U_), sol4);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(sol4[i]-x_true[i]), 0.0, 1e-6);
};

template<typename T, Layout L>
void t_trmv() {
    XorShift64 rng(24);
    const size_t n = 5;
    Matrix<T, L> Up(n, n, T(0));
    for (size_t i = 0; i < n; ++i) for (size_t j = i; j < n; ++j) Up(i,j) = rand_scalar<T>(rng);
    Vector<T> x = random_vector<T>(n, rng);
    Vector<T> x_ref(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t j = i; j < n; ++j) s += Up(i, j) * x[j]; x_ref[i] = s; };
    Vector<T> x1 = x;
    trmv('U', 'N', 'N', expr(Up), x1);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(x1[i] - x_ref[i]), 0.0, 1e-9);

    // View overload.
    Vector<T> x1b = x;
    auto x1bv = view(x1b);
    trmv('U', 'N', 'N', expr(Up), x1bv);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(x1b[i] - x_ref[i]), 0.0, 1e-9);

    // Transposed / conj-transposed / unit-diag variants exercised via detail path already covered structurally; spot check trans.
    Vector<T> x_ref_t(n, T(0));
    for (size_t i = 0; i < n; ++i) { T s = T(0); for (size_t k = 0; k <= i; ++k) s += Up(k, i) * x[k]; x_ref_t[i] = s; };
    Vector<T> x2 = x;
    trmv('U', 'T', 'N', expr(Up), x2);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(x2[i] - x_ref_t[i]), 0.0, 1e-9);
};

template<typename T, Layout L>
void t_symv_hemv() {
    XorShift64 rng(25);
    const size_t n = 4;
    // Build full symmetric/hermitian matrix, store only the triangle used by symv/hemv.
    Matrix<T, L> Full = random_matrix<T, L>(n, n, rng);
    Matrix<T, L> Sym(n, n, T(0));
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Sym(i, j) = Full(i, j) + Full(j, i); // symmetric
    Vector<T> x = random_vector<T>(n, rng);
    Vector<T> y(n, T(0));
    Vector<T> y_ref(n, T(0));
    for (size_t i = 0; i < n; ++i){ T s = T(0); for(size_t j = 0; j < n; ++j) s += Sym(i, j) * x[j]; y_ref[i] = T(2) * s; };
    symv('L', T(2), expr(Sym), expr(x), T(0), y);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(y[i] - y_ref[i]), 0.0, 1e-8);

    // Hermitian:
    Matrix<T, L> Herm(n, n, T(0));
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Herm(i, j) = Full(i, j) + conj(Full(j, i));
    Vector<T> yh(n, T(0));
    Vector<T> yh_ref(n, T(0));
    for (size_t i = 0; i < n; ++i){ T s = T(0); for(size_t j = 0; j < n; ++j) s += Herm(i, j) * x[j]; yh_ref[i] = s; };
    hemv('L', T(1), expr(Herm), expr(x), T(0), yh);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(yh[i] - yh_ref[i]), 0.0, 1e-8);

    // Upper-storage variants.
    Vector<T> yU(n, T(0));
    symv('U', T(1), expr(Sym), expr(x), T(0), yU);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(yU[i] - y_ref[i] / T(2)), 0.0, 1e-8);
};

template<typename T, Layout L>
void t_vgem() {
    XorShift64 rng(26);
    auto A = random_matrix<T, L>(4, 3, rng);
    auto x = random_vector<T>(4, rng);
    Vector<T> y(3, T(0));
    Vector<T> y_ref(3, T(0));
    for (size_t j = 0; j < 3; ++j) { T s = T(0); for (size_t i = 0; i < 4; ++i) s += x[i] * A(i,j); y_ref[j] = s; };
    vgem(expr(x), expr(A), y);
    for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(std::abs(y[j] - y_ref[j]), 0.0, 1e-9);

    Vector<T> y2 = expr(x) * expr(A);
    for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(std::abs(y2[j] - y_ref[j]), 0.0, 1e-9);
};

void run_level2_tests() {
    RUN_TEST((t_gemv<double, Layout::RowMajor>));
    RUN_TEST((t_gemv<double, Layout::ColMajor>));
    RUN_TEST((t_gemv<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_gemv<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_ger_gerc<double, Layout::RowMajor>));
    RUN_TEST((t_ger_gerc<double, Layout::ColMajor>));
    RUN_TEST((t_ger_gerc<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_ger_gerc<std::complex<double>, Layout::ColMajor>));
    RUN_TEST(t_trsv<double>);
    RUN_TEST(t_trsv<std::complex<double>>);
    RUN_TEST((t_trmv<double, Layout::RowMajor>));
    RUN_TEST((t_trmv<double, Layout::ColMajor>));
    RUN_TEST((t_trmv<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_trmv<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_symv_hemv<double, Layout::RowMajor>));
    RUN_TEST((t_symv_hemv<double, Layout::ColMajor>));
    RUN_TEST((t_symv_hemv<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_symv_hemv<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_vgem<double, Layout::RowMajor>));
    RUN_TEST((t_vgem<std::complex<double>, Layout::ColMajor>));
};