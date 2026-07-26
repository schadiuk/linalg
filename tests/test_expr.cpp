#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T>
void t_vec_expr() {
    Vector<T> a{ T(1), T(2), T(3) };
    Vector<T> b{ T(4), T(5), T(6) };
    Vector<T> s = expr(a) + expr(b);
    EXPECT(s[0] == T(5) && s[1] == T(7) && s[2] == T(9));
    Vector<T> d = expr(a) - expr(b);
    EXPECT(d[0] == T(-3));
    Vector<T> m = expr(a) * expr(b);
    EXPECT(m[1] == T(10));
    Vector<T> q = expr(a) / expr(b);
    EXPECT(std::abs(q[0] - T(1) / T(4)) < tol<T>());
    Vector<T> sc = T(2) * expr(a);
    EXPECT(sc[2] == T(6));
    Vector<T> sc2 = expr(a) * T(2);
    EXPECT(sc2[2] == T(6));
    Vector<T> sdiv = expr(a) / T(2);
    EXPECT(std::abs(sdiv[2] - T(3) / T(2)) < tol<T>());
    Vector<T> rdiv = T(6) / expr(a);
    EXPECT(std::abs(rdiv[0] - T(6)) < tol<T>());

    // Vector-matrix products via operator*.
    Matrix<T, Layout::RowMajor> M(3, 3, T(0));
    for (size_t i = 0; i < 3; ++i) M(i, i) = T(2);
    Vector<T> mv = expr(M) * expr(a); // GemvExpr
    EXPECT(mv[0] == T(2));
    Vector<T> vm = expr(a) * expr(M); // VgemExpr
    EXPECT(vm[0] == T(2));

    // depends_on utility on an expression.
    auto sumExpr = expr(a) + expr(b);
    EXPECT(sumExpr.depends_on(a.data(), sizeof(T)));
};

template<typename T, Layout L>
void t_mat_expr() {
    Matrix<T, L> A(2, 2, T(0)); 
    A(0, 0) = T(1); A(0, 1) = T(2);
    A(1, 0) = T(3); A(1, 1) = T(4);

    Matrix<T, L> B(2, 2, T(0));
    B(0, 0) = T(5); B(0, 1) = T(6); 
    B(1, 0) = T(7); B(1, 1) = T(8);

    Matrix<T, L> S = expr(A) + expr(B);
    EXPECT(S(0,0) == T(6) && S(1,1) == T(12));
    Matrix<T, L> D = expr(A) - expr(B);
    EXPECT(D(0,0) == T(-4));
    Matrix<T, L> Prod = expr(A) * expr(B); // GemmExpr
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    EXPECT_NEAR(std::abs(Prod(0,0) - T(19)), 0.0, tol<T>());
    EXPECT_NEAR(std::abs(Prod(0,1) - T(22)), 0.0, tol<T>());
    EXPECT_NEAR(std::abs(Prod(1,0) - T(43)), 0.0, tol<T>());
    EXPECT_NEAR(std::abs(Prod(1,1) - T(50)), 0.0, tol<T>());

    Matrix<T, L> Sc = T(2) * expr(A);
    EXPECT(Sc(1,1) == T(8));
    Matrix<T, L> Sc2 = expr(A) * T(2);
    EXPECT(Sc2(1,1) == T(8));
    Matrix<T, L> Sdiv = expr(A) / T(2);
    EXPECT(std::abs(Sdiv(1,1) - T(2)) < tol<T>());
    Matrix<T, L> Rdiv = T(8) / expr(A);
    EXPECT(std::abs(Rdiv(0,0) - T(8)) < tol<T>());

    Matrix<T, L> Had = elementwise_multiply(expr(A), expr(B));
    EXPECT(Had(0,0) == T(5) && Had(1,1) == T(32));
    Matrix<T, L> HadDiv = elementwise_divide(expr(A), expr(B));
    EXPECT(std::abs(HadDiv(0,0) - T(1)/T(5)) < tol<T>());

    // triu / tril extraction.
    Matrix<T, L> Sq(3,3,T(0));
    for (size_t i=0;i<3;++i) for(size_t j=0;j<3;++j) Sq(i,j)=static_cast<T>(i*3+j+1);
    Matrix<T, L> U = triu(expr(Sq));
    EXPECT(U(1,0) == T(0) && U(0,1) == Sq(0,1));
    Matrix<T, L> Lo = tril(expr(Sq));
    EXPECT(Lo(0,1) == T(0) && Lo(1,0) == Sq(1,0));
    Matrix<T, L> U1 = triu(expr(Sq), 1);
    EXPECT(U1(0,0) == T(0) && U1(0,1) == Sq(0,1));
    Matrix<T, L> Lm1 = tril(expr(Sq), -1);
    EXPECT(Lm1(0,0) == T(0) && Lm1(1,0) == Sq(1,0));

    MatRef<T, L> ref(A);
    EXPECT(ref(0,1) == A(0,1));
    EXPECT(ref.depends_on(A.data(), sizeof(T)));

    // Regression test: expr(MatrixView<T,L,Trans,Conj,true>&), the mutable-view overload.

    auto Vw = view(A);
    auto ew = expr(Vw);
    EXPECT(ew(1,0) == A(1,0));
    ew(0,1) = T(77);
    EXPECT(A(0,1) == T(77));

    const Matrix<T,L>& CA = A;
    auto CVw = view(CA);
    auto cew = expr(CVw);
    EXPECT(cew(1,0) == A(1,0));

    const auto& VwConst = Vw;
    auto ewc = expr(VwConst);
    EXPECT(ewc(1,0) == A(1,0));
};

void run_expr_tests() {
    RUN_TEST(t_vec_expr<double>);
    RUN_TEST(t_vec_expr<std::complex<double>>);
    RUN_TEST((t_mat_expr<double, Layout::RowMajor>));
    RUN_TEST((t_mat_expr<double, Layout::ColMajor>));
    RUN_TEST((t_mat_expr<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_mat_expr<std::complex<double>, Layout::ColMajor>));
};