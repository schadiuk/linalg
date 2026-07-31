#include <harness.hpp>

using namespace linalg;
using namespace test;

void t_schur_complex(size_t n) {
    XorShift64 rng(5000 + n);
    Matrix<DefaultScalar, Layout::RowMajor> A = random_matrix<DefaultScalar, Layout::RowMajor>(n, n, rng);

    SchurResult<Layout::RowMajor> res = schur(A, true, true);
    EXPECT(res.Q.rows() == n && res.T.rows() == n);
    EXPECT(is_orthonormal_cols(res.Q, 1e-6));
    // T must be upper triangular.
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < i; ++j) EXPECT_NEAR(std::abs(res.T(i,j)), 0.0, 1e-6);

    Matrix<DefaultScalar, Layout::RowMajor> QT(n, n, DefaultScalar(0)), QTQ(n, n, DefaultScalar(0));
    gemm(DefaultScalar(1), expr(res.Q), expr(res.T), DefaultScalar(0), QT);
    gemm(DefaultScalar(1), expr(QT), hermitian(res.Q), DefaultScalar(0), QTQ);
    EXPECT(residual_small(A, QTQ, 1e-5));
    EXPECT(res.eigvals.size() == n);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(res.eigvals[i] - res.T(i,i)), 0.0, 1e-9);

    SchurResult<Layout::RowMajor> res_nv = schur(A, false, true);
    EXPECT(res_nv.Q.rows() == 0);
    // Eigenvalue multiset should match (order may differ due to balancing/deflation order); compare sorted by real part then imag.
    std::vector<DefaultScalar> e1(res.eigvals.begin(), res.eigvals.end());
    std::vector<DefaultScalar> e2(res_nv.eigvals.begin(), res_nv.eigvals.end());
    auto cmp = [](const DefaultScalar& a, const DefaultScalar& b) {
        if (a.real() != b.real()) return a.real() < b.real();
        return a.imag() < b.imag();
    };
    std::sort(e1.begin(), e1.end(), cmp);
    std::sort(e2.begin(), e2.end(), cmp);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(e1[i]-e2[i]), 0.0, 1e-4);

    // No-balance path:
    SchurResult<Layout::RowMajor> res_nb = schur(A, true, false);
    EXPECT(!res_nb.balanced);
    EXPECT(!res.balanced);

    // Balancing engages when vectors are not requested.
    SchurResult<Layout::RowMajor> res_novec_bal = schur(A, false, true);
    EXPECT(res_novec_bal.balanced);
    EXPECT(res_novec_bal.balance_scale.size() == n);
    EXPECT(res_novec_bal.balance_perm.size() == n);

    // eigenvalues() convenience wrapper.
    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(expr(A));
    std::vector<DefaultScalar> e3(ev.begin(), ev.end());
    std::sort(e3.begin(), e3.end(), cmp);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(std::abs(e1[i]-e3[i]), 0.0, 1e-4);
};

void t_schur_real_input() {
    const size_t n = 32;
    Matrix<double, Layout::RowMajor> Ar(n, n, 0.0);
    XorShift64 rng(5100);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Ar(i,j) = rng.uniform(-1,1);

    SchurResult<Layout::RowMajor> res = schur(Ar, true, true);
    EXPECT(res.eigvals.size() == n);

    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(Ar);
    EXPECT(ev.size() == n);

    SchurResult<Layout::RowMajor> res2 = schur<Layout::RowMajor>(expr(Ar), true, true);
    EXPECT(res2.eigvals.size() == n);
};

void t_eig_right_left() {
    const size_t n = 32;
    XorShift64 rng(5200);
    Matrix<DefaultScalar, Layout::RowMajor> A = random_matrix<DefaultScalar, Layout::RowMajor>(n, n, rng);
    SchurResult<Layout::RowMajor> sres = schur(A, true, true);

    EigResult<DefaultScalar, Layout::RowMajor> er = eig(sres.T, sres.Q, true, false);
    EXPECT(er.VR.rows() == n && er.VR.cols() == n);
    for (size_t i = 0; i < n; ++i) {
        Vector<DefaultScalar> vi(n);
        for (size_t r = 0; r < n; ++r) vi[r] = er.VR(r, i);
        Vector<DefaultScalar> Av(n, DefaultScalar(0));
        gemv(DefaultScalar(1), expr(A), expr(vi), DefaultScalar(0), Av);
        double nrm = nrm2(expr(vi));
        double resid = 0.0;
        for (size_t r = 0; r < n; ++r) resid += std::norm(Av[r] - er.eigenvalues[i] * vi[r]);
        EXPECT(std::sqrt(resid) <= 1e-5 * std::max(1.0, nrm));
    };

    EigResult<DefaultScalar, Layout::RowMajor> el = eig(sres.T, sres.Q, false, true);
    EXPECT(el.VL.rows() == n && el.VL.cols() == n);
    for (size_t i = 0; i < n; ++i) {
        Vector<DefaultScalar> wi(n);
        for (size_t r = 0; r < n; ++r) wi[r] = el.VL(r, i);
        Vector<DefaultScalar> wA(n, DefaultScalar(0));
        vgem(expr(wi), expr(A), wA);
        Vector<DefaultScalar> AHw(n, DefaultScalar(0));
        gemv(DefaultScalar(1), hermitian(A), expr(wi), DefaultScalar(0), AHw);
        double resid = 0.0;
        for (size_t r = 0; r < n; ++r) resid += std::norm(AHw[r] - conj(el.eigenvalues[i]) * wi[r]);
        double nrm = nrm2(expr(wi));
        EXPECT(std::sqrt(resid) <= 1e-5 * std::max(1.0, nrm));
    };

    // ColMajor instantiation of eig() (exercises the layout-copy branch).
    Matrix<DefaultScalar, Layout::ColMajor> Ac(n, n);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Ac(i,j) = A(i,j);
    SchurResult<Layout::ColMajor> sresC = schur(Ac, true, true);
    EigResult<DefaultScalar, Layout::ColMajor> erC = eig(sresC.T, sresC.Q, true, true);
    EXPECT(erC.VR.rows() == n && erC.VL.rows() == n);
};

void run_schur_eig_tests() {
    RUN_TEST((std::bind(t_schur_complex, 4)));
    RUN_TEST((std::bind(t_schur_complex, 8)));
    RUN_TEST((std::bind(t_schur_complex, 63)));
    RUN_TEST((std::bind(t_schur_complex, 65)));
    RUN_TEST(t_schur_real_input);
    RUN_TEST(t_eig_right_left);
};