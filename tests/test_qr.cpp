#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_qr_reduced_complete_r(size_t m, size_t n) {
    XorShift64 rng(2000 + m * 100 + n);
    auto A = random_matrix<T, L>(m, n, rng);

    QRResult<T, L> red = qr_reduced(A);
    EXPECT(red.Q.cols() == std::min(m, n));
    EXPECT(is_orthonormal_cols(red.Q, 1e-6));
    Matrix<T, L> QR_(m, n, T(0));
    gemm(T(1), expr(red.Q), expr(red.R), T(0), QR_);
    EXPECT(residual_small(A, QR_, 1e-6));

    QRResult<T, L> comp = qr_complete(A);
    EXPECT(comp.Q.cols() == m);
    EXPECT(is_orthonormal_cols(comp.Q, 1e-6));
    Matrix<T, L> QRc(m, n, T(0));
    gemm(T(1), expr(comp.Q), expr(comp.R), T(0), QRc);
    EXPECT(residual_small(A, QRc, 1e-6));

    QRResult<T, L> ronly = qr_r(A);
    EXPECT(ronly.Q.rows() == 0); // Q not computed in R-only mode.
    EXPECT(residual_small(ronly.R, red.R, 1e-6));

    // Generic qr() with explicit args.
    QRResult<T, L> g = qr(A, QRMode::Reduced, false, -1.0);
    EXPECT(residual_small(g.R, red.R, 1e-6));

    // MatExpr overload.
    QRResult<T, L> ge = qr<T, L>(expr(A));
    EXPECT(residual_small(ge.R, red.R, 1e-6));
};

template<typename T, Layout L>
void t_qr_pivoted_rank_deficient() {
    const size_t m = 6, n = 4;
    XorShift64 rng(2100);
    Matrix<T, L> A(m, n, T(0));
    for (size_t i = 0; i < m; ++i) A(i, 0) = rand_scalar<T>(rng);
    for (size_t i = 0; i < m; ++i) A(i, 1) = rand_scalar<T>(rng);
    for (size_t i = 0; i < m; ++i) A(i, 2) = T(2) * A(i,0) - T(1) * A(i,1); // Linearly dependent column.
    for (size_t i = 0; i < m; ++i) A(i, 3) = rand_scalar<T>(rng);

    QRResult<T, L> res = qr_pivoted(A);
    EXPECT(res.pivoted);
    EXPECT(res.rank >= 0 && res.rank <= static_cast<int>(n));
    EXPECT(res.rank <= 3); // Rank-deficient by construction.
    
    // A * P == Q * R  (piv_to_P convention).
    Matrix<T, L> P = perm_matrix(res);
    Matrix<T, L> AP(m, n, T(0)), QR_(m, n, T(0));
    gemm(T(1), expr(A), expr(P), T(0), AP);
    gemm(T(1), expr(res.Q), expr(res.R), T(0), QR_);
    EXPECT(residual_small(AP, QR_, 1e-6));

    // perm_matrix() throws for a non-pivoted result.
    QRResult<T, L> nores = qr_reduced(A);
    bool threw = false;
    try { perm_matrix(nores); } catch (const std::logic_error&) { threw = true; };
    EXPECT(threw);

    // Explicit tol argument path.
    QRResult<T, L> res_tol = qr_pivoted(A, 1e-8);
    EXPECT(res_tol.pivoted);
};

void run_qr_tests() {
    // Straddle QR_BLOCK=64 and the blocked/unblocked dispatch boundary; tall & wide shapes.
    RUN_TEST((std::bind(t_qr_reduced_complete_r<double, Layout::RowMajor>, 6, 4)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<double, Layout::RowMajor>, 4, 6)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<double, Layout::RowMajor>, 80, 70)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<double, Layout::ColMajor>, 70, 80)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<std::complex<double>, Layout::RowMajor>, 6, 4)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<std::complex<double>, Layout::RowMajor>, 70, 65)));
    RUN_TEST((std::bind(t_qr_reduced_complete_r<std::complex<double>, Layout::ColMajor>, 65, 70)));
    RUN_TEST((t_qr_pivoted_rank_deficient<double, Layout::RowMajor>));
    RUN_TEST((t_qr_pivoted_rank_deficient<std::complex<double>, Layout::ColMajor>));
};