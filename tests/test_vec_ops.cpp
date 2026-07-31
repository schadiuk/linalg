#include <harness.hpp>

using namespace linalg;
using namespace test;

void t_linspace_arange() {
    Vector<double> ls = linspace<double>(0.0, 10.0, 5);
    EXPECT(ls.size() == 5);
    EXPECT_NEAR(ls[0], 0.0, 1e-12);
    EXPECT_NEAR(ls[4], 10.0, 1e-12);
    EXPECT_NEAR(ls[2], 5.0, 1e-12);

    Vector<double> lsn = linspace<double>(0.0, 10.0, 5, false);
    EXPECT_NEAR(lsn[4], 8.0, 1e-9);

    Vector<double> ls1 = linspace<double>(3.0, 7.0, 1);
    EXPECT(ls1.size() == 1 && ls1[0] == 3.0);

    Vector<double> ar = arange<double>(0.0, 5.0, 1.0);
    EXPECT(ar.size() == 5);
    EXPECT_NEAR(ar[4], 4.0, 1e-12);

    Vector<int> ari = arange<int>(0, 10, 2);
    EXPECT(ari.size() == 5 && ari[4] == 8);

    Vector<int> ar1 = arange<int>(5);
    EXPECT(ar1.size() == 5 && ar1[0] == 0 && ar1[4] == 4);
};

template<typename T>
void t_unary_math() {
    Vector<T> v{ T(1), T(4), T(9) };
    Vector<T> s = sqrt(expr(v));
    EXPECT_NEAR(std::abs(s[1] - T(2)), 0.0, tol<T>());
    Vector<T> e = exp(expr(v));
    EXPECT(e.size() == 3);
    Vector<T> lg = log(expr(e));
    EXPECT_NEAR(std::abs(lg[0] - v[0]), 0.0, 1e-6);
    Vector<T> ab = abs(expr(v));
    EXPECT_NEAR(std::abs(ab[0] - std::abs(v[0])), 0.0, tol<T>());
    Vector<T> sn = sin(expr(v)), cs = cos(expr(v));
    Vector<T> tn = tan(expr(v));
    Vector<T> sh = sinh(expr(v)), ch = cosh(expr(v)), th = tanh(expr(v));
    Vector<T> pw = pow(expr(v), 2.0);
    EXPECT_NEAR(std::abs(pw[1] - T(16)), 0.0, 1e-6);
    Vector<T> rl = real(expr(v));
    Vector<T> im = imag(expr(v));
    Vector<DefaultScalar> cp = cplx(expr(v));
    Vector<T> cj = conj(expr(v));
    (void)sn; (void)cs; (void)tn; (void)sh; (void)ch; (void)th; (void)rl; (void)im; (void)cp; (void)cj;

    if constexpr (!linalg::detail::is_complex_v<T>) {
        Vector<T> half{ T(0.5), T(0.5), T(0.5) };
        Vector<T> asn = asin(expr(half));
        Vector<T> acs = acos(expr(half));
        Vector<T> atn = atan(expr(v));
        Vector<T> fl = floor(expr(v));
        Vector<T> ce = ceil(expr(v));
        Vector<T> rd = round(expr(v));
        EXPECT(fl.size() == 3 && ce.size() == 3 && rd.size() == 3);
        (void)asn; (void)acs; (void)atn;
    };
};

template<typename T>
void t_reductions() {
    Vector<T> v{ T(1), T(2), T(3), T(4) };
    auto s = sum(expr(v));
    EXPECT_NEAR(std::abs(s - T(10)), 0.0, tol<T>());
    auto m = mean(expr(v));
    EXPECT_NEAR(std::abs(m - T(2.5)), 0.0, tol<T>());
    double var = variance(expr(v));
    EXPECT_NEAR(var, 1.25, 1e-9);
    double sd = stddev(expr(v));
    EXPECT_NEAR(sd, std::sqrt(1.25), 1e-9);
};

void run_vec_ops_tests() {
    RUN_TEST(t_linspace_arange);
    RUN_TEST(t_unary_math<double>);
    RUN_TEST(t_unary_math<std::complex<double>>);
    RUN_TEST(t_reductions<double>);
    RUN_TEST(t_reductions<std::complex<double>>);
};