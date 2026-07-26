#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_mat_unary_math() {
    Matrix<T, L> A(2, 2, T(0));
    A(0,0) = T(1); A(0,1) = T(4); A(1,0) = T(9); A(1,1) = T(16);
    Matrix<T, L> s = sqrt(expr(A));
    EXPECT_NEAR(std::abs(s(1,1) - T(4)), 0.0, tol<T>());
    Matrix<T, L> e = exp(expr(A));
    Matrix<T, L> lg = log(expr(e));
    EXPECT_NEAR(std::abs(lg(0,0) - A(0,0)), 0.0, 1e-6);
    Matrix<T, L> ab = abs(expr(A));
    Matrix<T, L> sn = sin(expr(A)), cs = cos(expr(A)), tn = tan(expr(A));
    Matrix<T, L> sh = sinh(expr(A)), ch = cosh(expr(A)), th = tanh(expr(A));
    Matrix<T, L> pw = pow(expr(A), 2.0);
    EXPECT_NEAR(std::abs(pw(0,1) - T(16)), 0.0, 1e-6);
    Matrix<T, L> rl = real(expr(A)), im = imag(expr(A)), cj = conj(expr(A));
    Matrix<DefaultScalar, L> cp = cplx(expr(A));
    (void)ab; (void)sn; (void)cs; (void)tn; (void)sh; (void)ch; (void)th; (void)rl; (void)im; (void)cp; (void)cj;

    if constexpr (!linalg::detail::is_complex_v<T>) {
        Matrix<T, L> half(2, 2, T(0.5));
        Matrix<T, L> asn = asin(expr(half));
        Matrix<T, L> acs = acos(expr(half));
        Matrix<T, L> atn = atan(expr(A));
        Matrix<T, L> fl = floor(expr(A)), ce = ceil(expr(A)), rd = round(expr(A));
        (void)asn; (void)acs; (void)atn; (void)fl; (void)ce; (void)rd;
    };
};

template<typename T, Layout L>
void t_mat_reductions_diag() {
    Matrix<T, L> A(3, 3, T(0));
    for (size_t i = 0; i < 3; ++i) for (size_t j = 0; j < 3; ++j) A(i,j) = static_cast<T>(i * 3 + j + 1);
    auto s = sum(expr(A));
    EXPECT_NEAR(std::abs(s - T(45)), 0.0, tol<T>());
    auto m = mean(expr(A));
    EXPECT_NEAR(std::abs(m - T(5)), 0.0, tol<T>());
    double var = variance(expr(A));
    EXPECT(var >= 0.0);
    double sd = stddev(expr(A));
    EXPECT_NEAR(sd, std::sqrt(var), 1e-9);

    auto tr = trace(expr(A));
    EXPECT_NEAR(std::abs(tr - (A(0,0)+A(1,1)+A(2,2))), 0.0, tol<T>());

    // diag extraction / construction round trip.
    Vector<T> dvec = diag(expr(A));
    EXPECT(dvec.size() == 3 && dvec[1] == A(1,1));
    Vector<T> dvec1 = diag(expr(A), 1);
    EXPECT(dvec1.size() == 2 && dvec1[0] == A(0,1));
    Vector<T> dvecm1 = diag(expr(A), -1);
    EXPECT(dvecm1.size() == 2 && dvecm1[0] == A(1,0));

    Vector<T> dv{ T(1), T(2), T(3) };
    Matrix<T, L> Dm = diag<VecRef<T>, L>(expr(dv));
    EXPECT(Dm.rows() == 3 && Dm(0,0) == T(1) && Dm(0,1) == T(0));
    Matrix<T, L> Dm1 = diag<VecRef<T>, L>(expr(dv), 1);
    EXPECT(Dm1.rows() == 4 && Dm1(0,1) == T(1));

    // triu/tril, flatten (matrix_expr already tested triu/tril free functions; flatten lives in matrix_ops).
    Vector<T> flat = flatten(expr(A));
    EXPECT(flat.size() == 9 && flat[0] == A(0,0) && flat[8] == A(2,2));
};

void run_mat_ops_tests() {
    RUN_TEST((t_mat_unary_math<double, Layout::RowMajor>));
    RUN_TEST((t_mat_unary_math<double, Layout::ColMajor>));
    RUN_TEST((t_mat_unary_math<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_mat_unary_math<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_mat_reductions_diag<double, Layout::RowMajor>));
    RUN_TEST((t_mat_reductions_diag<double, Layout::ColMajor>));
    RUN_TEST((t_mat_reductions_diag<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_mat_reductions_diag<std::complex<double>, Layout::ColMajor>));
};