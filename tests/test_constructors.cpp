#include <harness.hpp>

using namespace linalg;
using namespace test;

void t_constructors() {
    Vector<double> v{ 1.0, 2.0, 3.0 };

    auto C = circulant<double>(v);
    EXPECT(C.rows() == 3 && C.cols() == 3);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 3.0, 1e-12);

    Vector<double> c{ 1.0, 2.0, 3.0 };
    Vector<double> r{ 1.0, 5.0, 6.0 };
    auto Tp = toeplitz<double>(c, r);
    EXPECT(Tp.rows() == 3 && Tp.cols() == 3);
    EXPECT_NEAR(Tp(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Tp(0, 1), 5.0, 1e-12);
    EXPECT_NEAR(Tp(1, 0), 2.0, 1e-12);
    auto TpSym = toeplitz<double>(c);
    EXPECT_NEAR(TpSym(0,1), TpSym(1,0), 1e-12);

    Vector<double> hc{ 1.0, 2.0, 3.0 };
    Vector<double> hr{ 3.0, 4.0, 5.0 };
    auto Hk = hankel<double>(hc, hr);
    EXPECT_NEAR(Hk(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Hk(2, 2), 5.0, 1e-12);

    Vector<double> p{ 6.0, -11.0, 6.0 }; // x^3 - 6*x^2 + 11*x - 6.
    auto Comp = companion<double>(p);
    EXPECT(Comp.rows() == 3);
    EXPECT_NEAR(Comp(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(Comp(0, 2), -6.0, 1e-12);

    auto Pas = pascal<double>(4);
    EXPECT_NEAR(Pas(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Pas(3, 3), 20.0, 1e-9); // C(6, 3) = 20.

    auto Kh = kahan<double>(4);
    EXPECT(Kh.rows() == 4);

    auto Hb = hilbert<double>(4);
    EXPECT_NEAR(Hb(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Hb(1, 1), 1.0/3.0, 1e-12);

    auto Lm = lehmer<double>(4);
    EXPECT_NEAR(Lm(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Lm(0, 1), 0.5, 1e-12);

    auto W = wilkinson<double>(5);
    EXPECT(W.rows() == 5);
    EXPECT_NEAR(W(0, 1), 1.0, 1e-12);

    auto Fr = frank<double>(4);
    EXPECT_NEAR(Fr(0, 0), 4.0, 1e-12);

    auto Rd = redheffer<double>(4);
    EXPECT_NEAR(Rd(0, 0), 1.0, 1e-12);

    auto Hd = hadamard<double>(4);
    EXPECT(Hd.rows() == 4);

    Matrix<double, Layout::RowMajor> HHt(4, 4, 0.0);
    gemm(1.0, expr(Hd), transpose(Hd), 0.0, HHt);
    EXPECT_NEAR(HHt(0, 0), 4.0, 1e-9);
    EXPECT_NEAR(HHt(0, 1), 0.0, 1e-9);

    Vector<double> dl{ 1.0, 1.0 }, dm{ 2.0, 2.0, 2.0 }, du{ 3.0, 3.0 };
    auto Tri = tridiagonal<double>(dl, dm, du);
    EXPECT_NEAR(Tri(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(Tri(1, 1), 2.0, 1e-12);
    EXPECT_NEAR(Tri(1, 2), 3.0, 1e-12);

    Vector<double> bd{ 1.0, 2.0, 3.0 }, be{ 0.5, 0.5 };
    auto Bi = bidiagonal<double>(bd, be, true);
    EXPECT_NEAR(Bi(0, 1), 0.5, 1e-12);
    auto BiL = bidiagonal<double>(bd, be, false);
    EXPECT_NEAR(BiL(1, 0), 0.5, 1e-12);

    Vector<double> ad{ 1.0, 2.0 }, ac{ 3.0, 4.0 }, ar{ 5.0, 6.0 };
    auto Aw = arrowhead<double>(ad, ac, ar, 9.0);
    EXPECT(Aw.rows() == 3);
    EXPECT_NEAR(Aw(2, 2), 9.0, 1e-12);
    EXPECT_NEAR(Aw(0, 2), 3.0, 1e-12);
    EXPECT_NEAR(Aw(2, 0), 5.0, 1e-12);

    auto Mo = moler<double>(4);
    EXPECT_NEAR(Mo(0, 0), 1.0, 1e-12);

    auto Gr = grcar<double>(5, 2);
    EXPECT_NEAR(Gr(1, 0), -1.0, 1e-12);

    auto Lo = lotkin<double>(4);
    for (size_t j = 0; j < 4; ++j) EXPECT_NEAR(Lo(0, j), 1.0, 1e-12);

    auto Cl = clement<double>(4, false);
    EXPECT_NEAR(Cl(0, 1), 1.0, 1e-12);
    auto ClS = clement<double>(4, true);
    EXPECT(ClS.rows() == 4);

    auto Pe = pei<double>(4, 2.0);
    EXPECT_NEAR(Pe(0, 0), 3.0, 1e-12);
    EXPECT_NEAR(Pe(0, 1), 1.0, 1e-12);

    Vector<double> xv{ 1.0, 2.0, 3.0 };
    auto Vd = vandermonde<double>(xv);
    EXPECT_NEAR(Vd(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(Vd(1, 1), 2.0, 1e-12);
    EXPECT_NEAR(Vd(1, 2), 4.0, 1e-12);
    auto Vd4 = vandermonde<double>(xv, 4);
    EXPECT(Vd4.cols() == 4);

    Vector<double> alpha{ 0.0, 1.0, 2.0 };
    auto Vg = vandermonde_gen<double>(xv, alpha);
    EXPECT_NEAR(Vg(2, 2), 9.0, 1e-9);

    Vector<double> ox{ 1.0, 2.0 }, oy{ 3.0, 4.0, 5.0 };
    auto Ou = outer<double>(ox, oy);
    EXPECT_NEAR(Ou(0, 0), 3.0, 1e-12);
    EXPECT_NEAR(Ou(1, 2), 10.0, 1e-12);

    Vector<double> dx{ 1.0, 5.0, 2.0 };
    auto Ds = distance<double>(dx);
    EXPECT_NEAR(Ds(0, 1), 4.0, 1e-12);
    Vector<double> dy{ 0.0, 10.0 };
    auto Ds2 = distance<double>(dx, dy);
    EXPECT_NEAR(Ds2(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(Ds2(1, 1), 5.0, 1e-12);
};

void run_constructors_tests() {
    RUN_TEST(t_constructors);
};