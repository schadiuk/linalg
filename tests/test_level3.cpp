#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_gemm() {
    XorShift64 rng(31);
    // Sizes straddling the internal blocking thresholds.
    for (size_t n : { size_t(3), size_t(40), size_t(80) }) {
        auto A = random_matrix<T, L>(n, n, rng);
        auto B = random_matrix<T, L>(n, n, rng);
        Matrix<T, L> C = random_matrix<T, L>(n, n, rng);
        Matrix<T, L> C_ref(n, n, T(0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                T s = T(0);
                for (size_t k = 0; k < n; ++k) s += A(i, k) * B(k, j);
                C_ref(i,j) = T(1.5) * s + T(0.5) * C(i,j);
            };
        gemm(T(1.5), expr(A), expr(B), T(0.5), C);
        EXPECT(residual_small(C, C_ref, 1e-7));
    };

    // via GemmExpr construction/assignment paths (Matrix constructor and operator= overloads).
    auto A = random_matrix<T, L>(6, 6, rng);
    auto B = random_matrix<T, L>(6, 6, rng);
    Matrix<T, L> P = expr(A) * expr(B);
    Matrix<T, L> P_ref(6, 6, T(0));
    gemm(T(1), expr(A), expr(B), T(0), P_ref);
    EXPECT(residual_small(P, P_ref, 1e-9));

    Matrix<T, L> Q(6, 6, T(0));
    Q = expr(A) * expr(B); // operator= from GemmExpr.
    EXPECT(residual_small(Q, P_ref, 1e-9));

    // Accumulate paths: C0 + A * B and A * B + C0 (Matrix ctor/assign hooks).
    Matrix<T, L> C0 = random_matrix<T, L>(6, 6, rng);
    Matrix<T, L> S1 = expr(C0) + (expr(A) * expr(B));
    Matrix<T, L> S1_ref = C0;
    gemm(T(1), expr(A), expr(B), T(1), S1_ref);
    EXPECT(residual_small(S1, S1_ref, 1e-8));

    Matrix<T, L> S2 = (expr(A) * expr(B)) + expr(C0);
    EXPECT(residual_small(S2, S1_ref, 1e-8));

    Matrix<T, L> D1 = expr(C0) - (expr(A) * expr(B));
    Matrix<T, L> D1_ref = C0;
    gemm(T(-1), expr(A), expr(B), T(1), D1_ref);
    EXPECT(residual_small(D1, D1_ref, 1e-8));

    Matrix<T, L> D2 = (expr(A) * expr(B)) - expr(C0);
    Matrix<T, L> D2_ref(6, 6, T(0));
    gemm(T(1), expr(A), expr(B), T(0), D2_ref);
    gemm(T(-1), expr(C0), Matrix<T,L>::identity(6), T(1), D2_ref);
    EXPECT(residual_small(D2, D2_ref, 1e-8));

    Matrix<T, L> S1b(6, 6, T(0)); S1b = C0;
    S1b = expr(C0) + (expr(A) * expr(B));
    EXPECT(residual_small(S1b, S1_ref, 1e-8));
    Matrix<T, L> S2b(6, 6, T(0));
    S2b = (expr(A) * expr(B)) + expr(C0);
    EXPECT(residual_small(S2b, S1_ref, 1e-8));
    Matrix<T, L> D1b(6,6,T(0));
    D1b = expr(C0) - (expr(A) * expr(B));
    EXPECT(residual_small(D1b, D1_ref, 1e-8));
    Matrix<T, L> D2b(6, 6,T(0));
    D2b = (expr(A) * expr(B)) - expr(C0);
    EXPECT(residual_small(D2b, D2_ref, 1e-8));

    // Aliased GEMM: C = A*B where C shares storage with an operand (exercises depends_on / temp materialisation).
    Matrix<T, L> Aself = random_matrix<T, L>(5, 5, rng);
    Matrix<T, L> Aself_copy = Aself;
    Aself = expr(Aself) * expr(Aself); // self-aliased
    Matrix<T, L> Aself_ref(5,5,T(0));
    gemm(T(1), expr(Aself_copy), expr(Aself_copy), T(0), Aself_ref);
    EXPECT(residual_small(Aself, Aself_ref, 1e-8));
};

template<typename T, Layout L>
void t_trsm() {
    XorShift64 rng(32);
    for (size_t n : { size_t(5), size_t(70) }) {
        Matrix<T, L> Lo(n, n, T(0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) Lo(i, j) = rand_scalar<T>(rng);
            Lo(i,i) += T(static_cast<double>(n));
        };
        Matrix<T, L> X_true = random_matrix<T, L>(n, 4, rng);
        Matrix<T, L> Bm(n, 4, T(0));
        gemm(T(1), expr(Lo), expr(X_true), T(0), Bm);
        Matrix<T, L> sol = Bm;
        trsm('L', 'L', 'N', 'N', T(1), expr(Lo), sol);
        EXPECT(residual_small(sol, X_true, 1e-6));

        Matrix<T, L> Xr_true = random_matrix<T, L>(4, n, rng);
        Matrix<T, L> Br(4, n, T(0));
        gemm(T(1), expr(Xr_true), hermitian(Lo), T(0), Br);
        Matrix<T, L> solr = Br;
        trsm('R', 'L', 'C', 'N', T(1), expr(Lo), solr);
        EXPECT(residual_small(solr, Xr_true, 1e-6));
    };

    const size_t n = 6;
    Matrix<T, L> Lo(n, n, T(0));
    for (size_t i = 0; i < n; ++i) { for (size_t j = 0; j <= i; ++j) Lo(i, j)=rand_scalar<T>(rng); Lo(i, i) += T(6.0); };
    Matrix<T, L> X_true = random_matrix<T, L>(n, 3, rng);
    Matrix<T, L> Bm(n, 3, T(0));
    gemm(T(1), expr(Lo), expr(X_true), T(0), Bm);
    auto Bv = view(Bm);
    trsm('L', 'L', 'N', 'N', T(1), expr(Lo), Bv);
    EXPECT(residual_small(Bm, X_true, 1e-6));
};

template<typename T, Layout L>
void t_syrk_herk() {
    XorShift64 rng(33);
    const size_t n = 5, k = 4;
    auto A = random_matrix<T, L>(n, k, rng);
    Matrix<T, L> C(n, n, T(0));
    syrk('L', 'N', T(1), expr(A), T(0), C);

    Matrix<T, L> Full(n, n, T(0));
    gemm(T(1), expr(A), transpose(A), T(0), Full);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) EXPECT_NEAR(std::abs(C(i, j) - Full(i, j)), 0.0, 1e-8);

    // View overload.
    Matrix<T, L> Cv_mat(n, n, T(0));
    auto Cv = view(Cv_mat);
    syrk('L', 'N', T(1), expr(A), T(0), Cv);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) EXPECT_NEAR(std::abs(Cv_mat(i, j) - Full(i, j)), 0.0, 1e-8);

    using R = detail::real_type_t<T>;
    Matrix<T, L> H(n, n, T(0));
    herk('L', 'N', R(1), expr(A), R(0), H);
    Matrix<T, L> FullH(n, n, T(0));
    gemm(T(1), expr(A), hermitian(A), T(0), FullH);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) EXPECT_NEAR(std::abs(H(i, j) - FullH(i, j)), 0.0, 1e-8);

    Matrix<T, L> Hv_mat(n, n, T(0));
    auto Hv = view(Hv_mat);
    herk('L', 'N', R(1), expr(A), R(0), Hv);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) EXPECT_NEAR(std::abs(Hv_mat(i, j) - FullH(i, j)), 0.0, 1e-8);

    // Upper-storage + transposed variant: C = A^T * A.
    auto A2 = random_matrix<T, L>(k, n, rng);
    Matrix<T, L> Cu(n, n, T(0));
    syrk('U', 'T', T(1), expr(A2), T(0), Cu);
    Matrix<T, L> FullU(n, n, T(0));
    gemm(T(1), transpose(A2), expr(A2), T(0), FullU);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i; j < n; ++j) EXPECT_NEAR(std::abs(Cu(i, j) - FullU(i, j)), 0.0, 1e-8);
};

void run_level3_tests() {
    RUN_TEST((t_gemm<double, Layout::RowMajor>));
    RUN_TEST((t_gemm<double, Layout::ColMajor>));
    RUN_TEST((t_gemm<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_gemm<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_trsm<double, Layout::RowMajor>));
    RUN_TEST((t_trsm<double, Layout::ColMajor>));
    RUN_TEST((t_trsm<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_trsm<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_syrk_herk<double, Layout::RowMajor>));
    RUN_TEST((t_syrk_herk<double, Layout::ColMajor>));
    RUN_TEST((t_syrk_herk<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_syrk_herk<std::complex<double>, Layout::ColMajor>));
};