#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T>
void t_axpy_axpby_scal() {
    XorShift64 rng(7);
    auto x = random_vector<T>(50, rng);
    auto y = random_vector<T>(50, rng);
    Vector<T> y_ref = y;
    for (size_t i = 0; i < 50; ++i) y_ref[i] += T(2) * x[i];
    axpy(T(2), expr(x), y);
    for (size_t i = 0; i < 50; ++i) EXPECT_NEAR(std::abs(y[i] - y_ref[i]), 0.0, tol<T>());

    // axpy into a VectorView.
    Vector<T> y2 = random_vector<T>(50, rng);
    Vector<T> y2_ref = y2;
    for (size_t i = 0; i < 50; ++i) y2_ref[i] += T(3) * x[i];
    auto y2v = view(y2);
    axpy(T(3), expr(x), y2v);
    for (size_t i = 0; i < 50; ++i) EXPECT_NEAR(std::abs(y2[i] - y2_ref[i]), 0.0, tol<T>());

    Vector<T> y3 = random_vector<T>(50, rng);
    Vector<T> y3_ref;
    y3_ref = y3;
    for (size_t i = 0; i < 50; ++i) y3_ref[i] = T(2) * x[i] + T(3) * y3[i];
    axpby(T(2), expr(x), T(3), y3);
    for (size_t i = 0; i < 50; ++i) EXPECT_NEAR(std::abs(y3[i] - y3_ref[i]), 0.0, tol<T>());

    Vector<T> y4 = random_vector<T>(50, rng);
    Vector<T> y4_ref = y4;
    for (size_t i = 0; i < 50; ++i) y4_ref[i] = T(2) * x[i] + T(3) * y4[i];
    auto y4v = view(y4);
    axpby(T(2), expr(x), T(3), y4v);
    for (size_t i = 0; i < 50; ++i) EXPECT_NEAR(std::abs(y4[i] - y4_ref[i]), 0.0, tol<T>());

    Vector<T> sv = random_vector<T>(20, rng);
    Vector<T> sv_ref = sv;
    for (auto& e : sv_ref) e *= T(4);
    scal(T(4), sv);
    for (size_t i = 0; i < 20; ++i) EXPECT_NEAR(std::abs(sv[i] - sv_ref[i]), 0.0, tol<T>());

    Vector<T> sv2 = random_vector<T>(20, rng);
    Vector<T> sv2_ref = sv2;
    for (auto& e : sv2_ref) e *= T(5);
    auto sv2v = view(sv2);
    scal(T(5), sv2v);
    for (size_t i = 0; i < 20; ++i) EXPECT_NEAR(std::abs(sv2[i] - sv2_ref[i]), 0.0, tol<T>());

    Matrix<T, Layout::RowMajor> M = random_matrix<T, Layout::RowMajor>(4, 4, rng);
    Matrix<T, Layout::RowMajor> M_ref = M;
    for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j) M_ref(i,j) *= T(6);
    scal(T(6), M);
    EXPECT(residual_small(M, M_ref, tol<T>()));

    Matrix<T, Layout::RowMajor> M2 = random_matrix<T, Layout::RowMajor>(4, 4, rng);
    Matrix<T, Layout::RowMajor> M2_ref = M2;
    for (size_t i = 0; i < 4; ++i) for (size_t j = 0; j < 4; ++j) M2_ref(i,j) *= T(7);
    auto M2v = view(M2);
    scal(T(7), M2v);
    EXPECT(residual_small(M2, M2_ref, tol<T>()));
};

template<typename T>
void t_copy_swap() {
    XorShift64 rng(11);
    auto x = random_vector<T>(10, rng);
    Vector<T> y(10, T(0));
    copy(x, y);
    for (size_t i = 0; i < 10; ++i) EXPECT(x[i] == y[i]);

    VectorView<T, false> xview(x);
    Vector<T> y2(10, T(0));
    copy(xview, y2);
    for (size_t i = 0; i < 10; ++i) EXPECT(x[i] == y2[i]);

    Matrix<T, Layout::RowMajor> A = random_matrix<T, Layout::RowMajor>(3, 3, rng);
    Matrix<T, Layout::RowMajor> B(3, 3, T(0));
    copy(A, B);
    EXPECT(residual_small(A, B, tol<T>()));

    Vector<T> y3(10, T(0));
    copy(expr(x), y3);
    for (size_t i = 0; i < 10; ++i) EXPECT(x[i] == y3[i]);

    Vector<T> y4(10, T(0));
    auto y4v = view(y4);
    copy(expr(x), y4v);
    for (size_t i = 0; i < 10; ++i) EXPECT(x[i] == y4[i]);

    Matrix<T, Layout::RowMajor> C(3, 3, T(0));
    copy(expr(A), C);
    EXPECT(residual_small(A, C, tol<T>()));

    Matrix<T, Layout::RowMajor> D(3, 3, T(0));
    auto Dv = view(D);
    copy(expr(A), Dv);
    EXPECT(residual_small(A, D, tol<T>()));

    Vector<T> a{ T(1), T(2) }, b{ T(3), T(4) };
    swap(a, b);
    EXPECT(a[0] == T(3) && b[0] == T(1));

    Vector<T> c{ T(5), T(6) }, d{ T(7), T(8) };
    auto cv = view(c), dv = view(d);
    swap(cv, dv);
    EXPECT(c[0] == T(7) && d[0] == T(5));

    Vector<T> e{ T(9), T(10) };
    Vector<T> f{ T(11), T(12) };
    auto fv = view(f);
    swap(e, fv);
    EXPECT(e[0] == T(11) && f[0] == T(9));

    Vector<T> g{ T(13), T(14) };
    Vector<T> h{ T(15), T(16) };
    auto gv = view(g);
    swap(gv, h);
    EXPECT(g[0] == T(15) && h[0] == T(13));
};

template<typename T>
void t_reductions_dots() {
    Vector<T> v{ T(3), T(-4) };
    if constexpr (!linalg::detail::is_complex_v<T>) {
        EXPECT_NEAR(asum(expr(v)), 7.0, 1e-12);
    } else {
        EXPECT_NEAR(asum(expr(v)), std::abs(std::real(v[0])) + std::abs(std::imag(v[0])) + std::abs(std::real(v[1])) + std::abs(std::imag(v[1])), 1e-12);
    };
    EXPECT(iamax(expr(v)) == 1);
    EXPECT(iamin(expr(v)) == 0);

    Vector<T> a{ T(1), T(2), T(3) }, b{ T(4), T(5), T(6) };
    auto d = dot(expr(a), expr(b));
    T d_ref = T(0); for (size_t i = 0; i < 3; ++i) d_ref += a[i] * b[i];
    EXPECT_NEAR(std::abs(d - d_ref), 0.0, tol<T>());

    auto dc = dotc(expr(a), expr(b));
    T dc_ref = T(0); for (size_t i = 0; i < 3; ++i) dc_ref += conj(a[i])*b[i];
    EXPECT_NEAR(std::abs(dc - dc_ref), 0.0, tol<T>());

    double n2 = nrm2(expr(a));
    double n2_ref = std::sqrt(1.0 + 4.0 + 9.0);
    EXPECT_NEAR(n2, n2_ref, 1e-9);
};

void t_rotg_rot_real() {
    double a = 3.0, b = 4.0, c, s;
    rotg(a, b, c, s);
    EXPECT_NEAR(a, 5.0, 1e-9);
    EXPECT_NEAR(b, 0.0, 1e-9);
    EXPECT_NEAR(c * c + s * s, 1.0, 1e-9);

    // zero case.
    double az = 0.0, bz = 0.0, cz, sz;
    rotg(az, bz, cz, sz);
    EXPECT_NEAR(cz, 1.0, 1e-12);
    EXPECT_NEAR(sz, 0.0, 1e-12);

    Vector<double> x{ 1.0, 2.0, 3.0 };
    Vector<double> y{ 4.0, 5.0, 6.0 };
    double cc = 0.6, ss = 0.8;
    Vector<double> x0 = x, y0 = y;
    rot(x, y, cc, ss);
    EXPECT_NEAR(x[0], cc * x0[0] + ss * y0[0], 1e-9);
    EXPECT_NEAR(y[0], -ss * x0[0] + cc * y0[0], 1e-9);

    Vector<double> xv1 = x0, yv1 = y0;
    auto xv1w = view(xv1), yv1w = view(yv1);
    rot(xv1w, yv1w, cc, ss);
    EXPECT_NEAR(xv1[0], cc * x0[0] + ss * y0[0], 1e-9);

    Vector<double> xv2 = x0;
    Vector<double> yv2 = y0;
    auto yv2w = view(yv2);
    rot(xv2, yv2w, cc, ss);
    EXPECT_NEAR(xv2[0], cc * x0[0] + ss * y0[0], 1e-9);

    Vector<double> xv3 = x0;
    auto xv3w = view(xv3);
    Vector<double> yv3 = y0;
    rot(xv3w, yv3, cc, ss);
    EXPECT_NEAR(xv3[0], cc * x0[0] + ss * y0[0], 1e-9);
};

void t_rotg_rot_complex() {
    using C = std::complex<double>;
    C a = C(3, 4), b = C(1, -2);
    double c; C s;
    rotg(a, b, c, s);
    EXPECT_NEAR(c*c + std::norm(s), 1.0, 1e-9);
    EXPECT_NEAR(std::abs(b), 0.0, 1e-9);

    C az = C(0,0), bz = C(2, -1);
    double cz; C sz;
    rotg(az, bz, cz, sz);
    EXPECT_NEAR(cz, 0.0, 1e-12);

    Vector<C> x{ C(1, 1), C(2, 0), C(0, 3) };
    Vector<C> y{ C(4, -1), C(0, 5), C(1, 1) };
    double cc = 0.6; C ss(0.48, 0.64);
    Vector<C> x0 = x, y0 = y;
    rot(x, y, cc, ss);
    C csconj = conj(ss);
    EXPECT_NEAR(std::abs(x[0] - (cc * x0[0] + ss * y0[0])), 0.0, 1e-9);
    EXPECT_NEAR(std::abs(y[0] - (-csconj * x0[0] + cc * y0[0])), 0.0, 1e-9);

    Vector<C> xv = x0, yv = y0;
    auto xvw = view(xv), yvw = view(yv);
    rot(xvw, yvw, cc, ss);
    EXPECT_NEAR(std::abs(xv[0] - (cc * x0[0] + ss * y0[0])), 0.0, 1e-9);
};

void run_level1_tests() {
    RUN_TEST(t_axpy_axpby_scal<double>);
    RUN_TEST(t_axpy_axpby_scal<std::complex<double>>);
    RUN_TEST(t_copy_swap<double>);
    RUN_TEST(t_copy_swap<std::complex<double>>);
    RUN_TEST(t_reductions_dots<double>);
    RUN_TEST(t_reductions_dots<std::complex<double>>);
    RUN_TEST(t_rotg_rot_real);
    RUN_TEST(t_rotg_rot_complex);
};