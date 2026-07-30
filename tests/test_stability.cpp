#include <harness.hpp>
#include <numbers>

using namespace linalg;
using namespace test;

/*
Hilbert matrix: canonical severely ill-conditioned SPD matrix. Cholesky/LU remain formally valid (H is exactly SPD
in infinite precision) but achievable solution accuracy must degrade gracefully with n rather than exploding into NaN/Inf.
*/
void t_hilbert_conditioning() {
    XorShift64 rng(9001);
    double prev_cond = 0.0;
    for (size_t n : { size_t(3), size_t(6), size_t(9) }) {
        auto H = hilbert<double>(n);
        double c = cond2(H);
        EXPECT(std::isfinite(c));
        EXPECT(c > prev_cond);
        prev_cond = c;

        // Cholesky must still succeed (H is SPD) and reproduce H to a tolerance that scales with conditioning.
        CholeskyResult<double, Layout::RowMajor> res = potrf(H, 'L');
        Matrix<double, Layout::RowMajor> LLt(n, n, 0.0);
        gemm(1.0, expr(res.factor), hermitian(res.factor), 0.0, LLt);
        EXPECT(all_finite(LLt));
        EXPECT(residual_small(H, LLt, 1e-6));

        Vector<double> x_true = random_vector<double>(n, rng);
        Vector<double> b(n, 0.0);
        for (size_t i = 0; i < n; ++i) { double s=0.0; for(size_t j = 0; j < n; ++j) s += H(i, j) * x_true[j]; b[i] = s; };
        Vector<double> sol = b;
        potrs(res, sol);
        EXPECT(all_finite(sol));
        double err = 0.0, nrm = 0.0;
        for (size_t i = 0; i < n; ++i) { err += std::norm(sol[i]-x_true[i]); nrm += std::norm(x_true[i]); };
        double rel_err = std::sqrt(err) / std::max(1e-300, std::sqrt(nrm));
        // Generous backward-stability bound: forward error <~ cond(H) * eps, with slack.
        EXPECT(rel_err <= 1e3 * c * std::numeric_limits<double>::epsilon() + 1e-6);
    };
};

// Kahan matrix: upper-triangular, exponentially decaying diagonal: test for rank-revealing (pivoted) QR.

void t_kahan_qr() {
    const size_t n = 12;
    const double theta = 0.3;
    auto K = kahan<double>(n, theta, 0.0);
    EXPECT(all_finite(K));

    QRResult<double, Layout::RowMajor> res = qr_pivoted(K, 1e-12);
    EXPECT(res.pivoted);
    EXPECT(all_finite(res.R));
    for (size_t i = 0; i + 1 < n; ++i) EXPECT(std::abs(res.R(i,i)) >= std::abs(res.R(i+1,i+1)) - 1e-9);

    // A tight tolerance should reveal a rank strictly less than n.
    QRResult<double, Layout::RowMajor> res_tight = qr_pivoted(K, 1e-3);
    EXPECT(res_tight.rank < static_cast<int>(n));
    EXPECT(res_tight.rank > 0);

    // Reconstruction must still hold to tight tolerance regardless of rank estimate.
    Matrix<double, Layout::RowMajor> P = perm_matrix(res);
    Matrix<double, Layout::RowMajor> AP(n, n, 0.0), QR_(n, n, 0.0);
    gemm(1.0, expr(K), expr(P), 0.0, AP);
    gemm(1.0, expr(res.Q), expr(res.R), 0.0, QR_);
    EXPECT(residual_small(AP, QR_, 1e-8));
};

/*
Vandermonde matrix from closely-clustered nodes: a standard ill-conditioned least-squares stress test.
QR- and SVD-based lstsq must agree with each other even though the raw normal-equations approach would catastrophically fail.
This is a cross-validation-style stability check rather than a comparison against a hand-computed reference.
*/
void t_vandermonde_lstsq() {
    const size_t n = 8;
    Vector<double> nodes(n);
    for (size_t i = 0; i < n; ++i) nodes[i] = 1.0 + static_cast<double>(i) * 0.01; // Tightly clustered.
    auto V = vandermonde<double>(nodes);
    double c = cond2(V);
    EXPECT(std::isfinite(c));
    EXPECT(c > 1e6);

    XorShift64 rng(9002);
    Vector<double> x_true = random_vector<double>(n, rng);
    Vector<double> b(n, 0.0);
    for (size_t i = 0; i < n; ++i) { double s=0.0; for(size_t j=0;j<n;++j) s+=V(i,j)*x_true[j]; b[i]=s; };

    LstsqVecResult<double> resQ = lstsq_qr(V, b);
    LstsqVecResult<double> resS = lstsq_svd(V, b);
    EXPECT(all_finite(resQ.x));
    EXPECT(all_finite(resS.x));

    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) diff += std::norm(resQ.x[i] - resS.x[i]);
    diff = std::sqrt(diff);
    EXPECT(diff <= 1e-6 * c);
};

/*
Wilkinson matrix: symmetric tridiagonal with nearly-coincident eigenvalue pairs: notoriously hard for naive eigenvalue algorithms to separate.
All eigenvalues must come out real (matrix is symmetric) and the multiset must be symmetric about the matrix's central value.
*/
void t_wilkinson_eig() {
    const size_t n = 21; // The size at which the famous near-degenerate pair appears.
    auto W = wilkinson<double>(n);
    EXPECT(all_finite(W));

    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(W);
    EXPECT(ev.size() == n);
    EXPECT(all_finite(ev));
    std::vector<double> re(n);
    double sum = 0.0, tr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        // Symmetric tridiagonal real input -> eigenvalues must be (numerically) real.
        EXPECT_NEAR(ev[i].imag(), 0.0, 1e-5);
        re[i] = ev[i].real();
        sum += re[i];
        tr += W(i, i);
    };
    EXPECT_NEAR(sum, tr, 1e-6 * std::max(1.0, std::abs(tr))); // sum(eigenvalues) == trace

    // Wilkinson's matrix has at least one pair of eigenvalues that agree to many decimal places without being exactly equal.
    std::sort(re.begin(), re.end());
    double min_gap = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < n; ++i) min_gap = std::min(min_gap, re[i + 1] - re[i]);
    EXPECT(min_gap >= 0.0);
    EXPECT(min_gap < 1e-6);
};

// Frank matrix: known to have an entirely real, positive spectrum with unit product (matching its unit determinant).
void t_frank() {
    const size_t n = 16;
    auto F = frank<double>(n);
    EXPECT(all_finite(F));

    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(F);
    EXPECT(all_finite(ev));
    std::vector<double> re(n);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(ev[i].imag(), 0.0, 1e-4);
        EXPECT(ev[i].real() > 0.0); // Frank matrix spectrum is strictly positive
        re[i] = ev[i].real();
    };
    std::sort(re.begin(), re.end());
    double prod = 1.0;
    for (double v : re) prod *= v;
    EXPECT_NEAR(prod, 1.0, 1e-3);
    // Cross-check the product of eigenvalues against det via LU.
    LUResult<double, Layout::RowMajor> lures = lu(F);
    double d = lu_det(lures);
    EXPECT_NEAR(d, 1.0, 1e-4);
};

// Companion matrix: eigenvalues are exactly the roots of the generating polynomial.
void t_companion() {
    // Defines monic polynomial with real roots {1, 2, 3, -1}:
    Vector<double> p{ -6.0, 5.0, 5.0, -5.0 };
    auto C = companion<double>(p);
    EXPECT(all_finite(C));

    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(C);
    std::vector<double> re(4);
    for (size_t i = 0; i < 4; ++i) { EXPECT_NEAR(ev[i].imag(), 0.0, 1e-6); re[i] = ev[i].real(); };
    std::sort(re.begin(), re.end());
    std::vector<double> expected{ -1.0, 1.0, 2.0, 3.0 };
    for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(re[i], expected[i], 1e-6);
};

// Circulant matrix: eigenvalues have the closed-form DFT expression.
void t_circulant() {
    const size_t n = 6;
    Vector<double> c{ 1.0, 2.0, -1.0, 0.5, 3.0, -2.0 };
    auto Cm = circulant<double>(c);
    EXPECT(all_finite(Cm));

    Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(Cm);
    std::vector<std::complex<double>> got(ev.begin(), ev.end());

    std::vector<std::complex<double>> expected(n);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> s(0.0, 0.0);
        for (size_t j = 0; j < n; ++j) {
            double angle = -2.0 * std::numbers::pi * static_cast<double>(j * k) / static_cast<double>(n);
            s += c[j] * std::complex<double>(std::cos(angle), std::sin(angle));
        };
        expected[k] = s;
    };
    auto cmp = [](const std::complex<double>& a, const std::complex<double>& b) {
        if (a.real() != b.real()) return a.real() < b.real();
        return a.imag() < b.imag();
    };
    std::sort(got.begin(), got.end(), cmp);
    std::sort(expected.begin(), expected.end(), cmp);
    for (size_t k = 0; k < n; ++k) EXPECT_NEAR(std::abs(got[k] - expected[k]), 0.0, 1e-6);
};

/*
Hadamard matrix: perfectly conditioned (unit cond2 up to roundoff), orthogonal-up-to-scaling matrix.
This is the "best case" stability sanity check, contrasting with the ill-conditioned cases above.
*/
void t_hadamard() {
    const size_t n = 16;
    auto Hd = hadamard<double>(n);
    double c = cond2(Hd);
    EXPECT_NEAR(c, 1.0, 1e-9);

    SVDResult<double, Layout::RowMajor> res = svd(Hd);
    for (size_t i = 0; i < n; ++i) EXPECT_NEAR(res.s[i], std::sqrt(static_cast<double>(n)), 1e-9);
};

// Pascal matrix: SPD with determinant exactly 1.
void t_pascal_unit_det() {
    for (size_t n : { size_t(4), size_t(8), size_t(16) }) {
        auto Pm = pascal<double>(n);
        EXPECT(all_finite(Pm));
        CholeskyResult<double, Layout::RowMajor> res = potrf(Pm, 'L');
        EXPECT(all_finite(res.factor));
        double d = cholesky_det(res);
        EXPECT(std::isfinite(d));
        EXPECT_NEAR(d, 1.0, 1e-4 * std::max(1.0, d));
    };
};

// Moler matrix: SPD-by-construction but with exactly one tiny eigenvalue.
void t_moler_near_singular() {
    const size_t n = 10;
    auto M = moler<double>(n);
    EXPECT(all_finite(M));

    SVDResult<double, Layout::RowMajor> res = svd(M);
    EXPECT(all_finite(res.s));
    // Moler's smallest singular value is orders of magnitude below the largest.
    EXPECT(res.s[n - 1] <= 1e-6 * res.s[0]);

    Matrix<double, Layout::RowMajor> P = pinv(M);
    EXPECT(all_finite(P));
    Matrix<double, Layout::RowMajor> MP(n, n, 0.0), MPM(n, n, 0.0);
    gemm(1.0, expr(M), expr(P), 0.0, MP);
    gemm(1.0, expr(MP), expr(M), 0.0, MPM);
    EXPECT(residual_small(MPM, M, 1e-4));

    LstsqVecResult<double> res_svd = lstsq_svd(M, Vector<double>(n, 1.0), res.s[0] * 1e-3);
    EXPECT(all_finite(res_svd.x));
    EXPECT(res_svd.rank < static_cast<int>(n)); // Rank deficiency correctly detected.
};

/*
Lehmer matrix: SPD and well-conditioned (cond grows only linearly with n): part of a "control group" contrasting with Hilbert, confirming that a benign
structured matrix achieves near-machine-precision accuracy throughout.
*/
void t_lehmer() {
    const size_t n = 20;
    auto Lm = lehmer<double>(n);
    double c = cond2(Lm);
    EXPECT(std::isfinite(c));
    EXPECT(c < 1e4 * static_cast<double>(n * n));

    CholeskyResult<double, Layout::RowMajor> res = potrf(Lm, 'L');
    Matrix<double, Layout::RowMajor> Inv = potri(res);
    Matrix<double, Layout::RowMajor> Prod(n, n, 0.0);
    gemm(1.0, expr(Lm), expr(Inv), 0.0, Prod);
    EXPECT(all_finite(Prod));
    EXPECT(residual_small(Prod, Matrix<double,Layout::RowMajor>::identity(n), 1e-8));
};

/*
Redheffer matrix: non-symmetric, moderately ill-conditioned matrix.
Its determinant equals the number-theoretic Mertens function M(n).
*/
void t_redheffer_det() {
    // M(n) for n = 1...10:
    std::vector<double> mertens{ 1,0,-1,-1,-2,-1,-2,-2,-2,-1 };
    for (size_t n = 1; n <= 10; ++n) {
        auto R = redheffer<double>(n);
        LUResult<double, Layout::RowMajor> res = lu(R);
        double d = lu_det(res);
        EXPECT(std::isfinite(d));
        EXPECT_NEAR(d, mertens[n - 1], 1e-6);
    };
};

/*
Toeplitz built from a Kahan-decayed generating vector: confirms the SVD/bidiagonalisation stack degrades gracefully.
Should produce no NaN/Inf, singular values still sorted and non-negative) on a structured, ill-conditioned, but non-symmetric input.
*/
void t_toeplitz_kahan_gen() {
    const size_t n = 10;
    Vector<double> col(n), row(n);
    col[0] = row[0] = 1.0;
    for (size_t i = 1; i < n; ++i) { col[i] = std::pow(0.3, static_cast<double>(i)); row[i] = std::pow(0.5, static_cast<double>(i)); };
    auto Tp = toeplitz<double>(col, row);
    EXPECT(all_finite(Tp));

    SVDResult<double, Layout::RowMajor> res = svd(Tp);
    EXPECT(all_finite(res.s));
    for (size_t i = 0; i < res.s.size(); ++i) EXPECT(res.s[i] >= -1e-9);
    for (size_t i = 0; i + 1 < res.s.size(); ++i) EXPECT(res.s[i] >= res.s[i+1] - 1e-9);
    EXPECT(all_finite(res.U));
    EXPECT(all_finite(res.V));
};

// Pei matrix: closed-form spectrum.
void t_pei() {
    for (double alpha : { 5.0, 1e-4 }) {
        const size_t n = 10;
        auto Pm = pei<double>(n, alpha);
        EXPECT(all_finite(Pm));

        double c = cond2(Pm);
        double c_expected = (alpha + static_cast<double>(n)) / alpha;
        EXPECT_NEAR(c, c_expected, 1e-4 * c_expected);

        // Eigenvalues via the general Schur/eig path must match the exact closed form.
        Vector<std::complex<double>> ev = eigenvalues<Layout::RowMajor>(Pm);
        std::vector<double> re(n);
        for (size_t i = 0; i < n; ++i) { EXPECT_NEAR(ev[i].imag(), 0.0, 1e-5); re[i] = ev[i].real(); };
        std::sort(re.begin(), re.end());
        for (size_t i = 0; i + 1 < n; ++i) EXPECT_NEAR(re[i], alpha, 1e-5 * std::max(1.0, alpha));
        EXPECT_NEAR(re[n - 1], alpha + static_cast<double>(n), 1e-5 * (alpha + n));

        // SPD for alpha > 0 regardless of conditioning -- Cholesky must succeed and stay finite.
        CholeskyResult<double, Layout::RowMajor> res = potrf(Pm, 'L');
        EXPECT(all_finite(res.factor));
        Matrix<double, Layout::RowMajor> LLt(n, n, 0.0);
        gemm(1.0, expr(res.factor), hermitian(res.factor), 0.0, LLt);
        EXPECT(residual_small(Pm, LLt, 1e-6));
    };
};

/*
Grcar matrix: example of a highly non-normal banded Toeplitz matrix.
No simple closed-form spectrum exists, so check  algebraic invariants that must hold regardless: 
trace == sum(eigenvalues), det == prod(eigenvalues)
*/
void t_grcar() {
    const size_t n = 16;
    auto G = grcar<double>(n, 3);
    EXPECT(all_finite(G));

    SchurResult<Layout::RowMajor> sres = schur(G, true, true);
    EXPECT(all_finite(sres.T));
    EXPECT(all_finite(sres.Q));
    EXPECT(is_orthonormal_cols(sres.Q, 1e-6));

    double tr = 0.0;
    for (size_t i = 0; i < n; ++i) tr += G(i, i);
    std::complex<double> eig_sum(0.0, 0.0);
    for (size_t i = 0; i < n; ++i) eig_sum += sres.eigvals[i];
    EXPECT_NEAR(eig_sum.real(), tr, 1e-6 * std::max(1.0, std::abs(tr)));
    EXPECT_NEAR(eig_sum.imag(), 0.0, 1e-6);

    LUResult<double, Layout::RowMajor> lures = lu(G);
    double det_lu = lu_det(lures);
    std::complex<double> det_eig(1.0, 0.0);
    for (size_t i = 0; i < n; ++i) det_eig *= sres.eigvals[i];
    EXPECT_NEAR(det_eig.real(), det_lu, 1e-4 * std::max(1.0, std::abs(det_lu)));
    EXPECT_NEAR(det_eig.imag(), 0.0, 1e-4 * std::max(1.0, std::abs(det_lu)));

    Matrix<std::complex<double>, Layout::RowMajor> Gc(n, n);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) Gc(i,j) = std::complex<double>(G(i,j), 0.0);
    EigResult<std::complex<double>, Layout::RowMajor> er = eig(sres.T, sres.Q, true, false);
    EXPECT(all_finite(er.VR));
    for (size_t i = 0; i < n; ++i) {
        Vector<std::complex<double>> vi(n);
        for (size_t r = 0; r < n; ++r) vi[r] = er.VR(r, i);
        Vector<std::complex<double>> Av(n, std::complex<double>(0,0));
        gemv(std::complex<double>(1), expr(Gc), expr(vi), std::complex<double>(0), Av);
        double resid = 0.0;
        for (size_t r = 0; r < n; ++r) resid += std::norm(Av[r] - er.eigenvalues[i] * vi[r]);
        EXPECT(std::isfinite(resid));
        EXPECT(std::sqrt(resid) <= 1e-3); // Loose: eigenvector sensitivity is the whole point of Grcar.
    };
};

// Lotkin matrix: a Hilbert matrix with its first row forced to all-ones: breaks Hilbert's symmetry while making the conditioning worse.
void t_lotkin() {
    const size_t n = 8;
    auto Lo = lotkin<double>(n);
    EXPECT(all_finite(Lo));
    for (size_t j = 0; j < n; ++j) EXPECT_NEAR(Lo(0, j), 1.0, 1e-12);

    SVDResult<double, Layout::RowMajor> res = svd(Lo);
    EXPECT(all_finite(res.s));
    for (size_t i = 0; i + 1 < res.s.size(); ++i) EXPECT(res.s[i] >= res.s[i+1] - 1e-9);
    double c = cond2(Lo);
    // Lotkin is known to be at least as ill-conditioned as Hilbert of the same size.
    auto Hb = hilbert<double>(n);
    double cH = cond2(Hb);
    EXPECT(c >= cH * 0.5);

    XorShift64 rng(9010);
    Vector<double> x_true = random_vector<double>(n, rng);
    Vector<double> b(n, 0.0);
    for (size_t i = 0; i < n; ++i) { double s=0.0; for(size_t j=0;j<n;++j) s+=Lo(i,j)*x_true[j]; b[i]=s; };
    LstsqVecResult<double> resQ = lstsq_qr(Lo, b);
    LstsqVecResult<double> resS = lstsq_svd(Lo, b);
    EXPECT(all_finite(resQ.x));
    EXPECT(all_finite(resS.x));
    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) diff += std::norm(resQ.x[i] - resS.x[i]);
    EXPECT(std::sqrt(diff) <= 1e-5 * c); // Two independent drivers must still agree, scaled by conditioning.
};

void run_numerical_stability_tests() {
    RUN_TEST(t_hilbert_conditioning);
    RUN_TEST(t_kahan_qr);
    RUN_TEST(t_vandermonde_lstsq);
    RUN_TEST(t_wilkinson_eig);
    RUN_TEST(t_frank);
    RUN_TEST(t_companion);
    RUN_TEST(t_circulant);
    RUN_TEST(t_hadamard);
    RUN_TEST(t_pascal_unit_det);
    RUN_TEST(t_moler_near_singular);
    RUN_TEST(t_lehmer);
    RUN_TEST(t_redheffer_det);
    RUN_TEST(t_toeplitz_kahan_gen);
    RUN_TEST(t_pei);
    RUN_TEST(t_grcar);
    RUN_TEST(t_lotkin);
};