#include <harness.hpp>

using namespace linalg;
using namespace test;

template<typename T, Layout L>
void t_vector_basics() {
    XorShift64 rng(0x1234567890ABCDEFULL);
    Vector<T> v(5, T(0));
    EXPECT(v.size() == 5);
    for (size_t i = 0; i < 5; ++i) v[i] = static_cast<T>(i + 1);
    EXPECT(v[0] == T(1) && v[4] == T(5));
    EXPECT(v.at(2) == T(3));
    Vector<T> ones = Vector<T>::ones(4);
    for (size_t i = 0; i < 4; ++i) EXPECT(ones[i] == T(1));
    Vector<T> zeros = Vector<T>::zeros(4);
    for (size_t i = 0; i < 4; ++i) EXPECT(zeros[i] == T(0));
    Vector<T> r = Vector<T>::random(10);
    EXPECT(r.size() == 10);

    Vector<T> il{ T(1), T(2), T(3) };
    EXPECT(il.size() == 3 && il[1] == T(2));
    // swap: pointer exchange.
    Vector<T> a(3, T(1)), b(3, T(2));
    a.swap(b);
    EXPECT(a[0] == T(2) && b[0] == T(1));
    // begin/end iteration.
    size_t cnt = 0; for (auto it = v.begin(); it != v.end(); ++it) ++cnt;
    EXPECT(cnt == v.size());
    // depends_on: self should always alias.
    EXPECT(v.depends_on(v.data(), sizeof(T)));
};

template<typename T, Layout L>
void t_matrix_basics() {
    Matrix<T, L> A(3, 4, T(0));
    EXPECT(A.rows() == 3 && A.cols() == 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j) A(i, j) = static_cast<T>(i * 4 + j + 1);
    EXPECT(A(0, 0) == T(1) && A(2, 3) == T(12));
    EXPECT(A.at(1, 1) == T(6));

    auto I = Matrix<T, L>::identity(4);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) EXPECT(I(i, j) == ((i == j) ? T(1) : T(0)));

    auto O = Matrix<T, L>::ones(2, 2);
    for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) EXPECT(O(i, j) == T(1));

    auto Z = Matrix<T, L>::zeros(2, 2);
    for (size_t i = 0; i < 2; ++i) for (size_t j = 0; j < 2; ++j) EXPECT(Z(i, j) == T(0));

    auto R = Matrix<T, L>::random(5, 5);
    EXPECT(R.rows() == 5 && R.cols() == 5);

    // row/col extraction.
    Vector<T> row1 = A.row(1);
    EXPECT(row1.size() == 4 && row1[0] == A(1, 0));
    Vector<T> row1_sub = A.row(1, 1, 2);
    EXPECT(row1_sub.size() == 2 && row1_sub[0] == A(1, 1));
    Vector<T> col2 = A.col(2);
    EXPECT(col2.size() == 3 && col2[0] == A(0, 2));
    Vector<T> col2_sub = A.col(2, 1, 2);
    EXPECT(col2_sub.size() == 2 && col2_sub[0] == A(1, 2));

    // reshape (row-major flat traversal invariant preserved logically only when total matches).
    Matrix<T, L> B(2, 6, T(0));
    for (size_t i = 0; i < 12; ++i) B.data()[i] = static_cast<T>(i);
    B.reshape(3, 4);
    EXPECT(B.rows() == 3 && B.cols() == 4);

    // to_array / array-constructor round trip.
    std::array<std::array<T, 2>, 2> arr{{ {T(1), T(2)}, {T(3), T(4)} }};
    Matrix<T, L> M(arr);
    EXPECT(M(0,0)==T(1) && M(0,1)==T(2) && M(1,0)==T(3) && M(1,1)==T(4));
    auto arr2 = M.template to_array<2,2>();
    EXPECT(arr2[1][1] == T(4));

    // C-array assignment operator.
    Matrix<T, L> M2(2, 2, T(0));
    T carr[2][2] = { {T(5), T(6)}, {T(7), T(8)} };
    M2 = carr;
    EXPECT(M2(1,1) == T(8));

    EXPECT(A.depends_on(A.data(), sizeof(T)));
};

template<typename T, Layout L>
void t_views() {
    XorShift64 rng(42);
    auto A = random_matrix<T, L>(4, 4, rng);
    // Mutable view mirrors underlying storage.
    auto Vw = view(A);
    EXPECT(Vw.rows() == 4 && Vw.cols() == 4);
    Vw(0, 0) = T(99);
    EXPECT(A(0, 0) == T(99));
    // Const view.
    const Matrix<T, L>& CA = A;
    auto CVw = view(CA);
    EXPECT(CVw(0, 0) == T(99));

    auto Tv = transpose(A);
    EXPECT(Tv.rows() == 4 && Tv.cols() == 4);
    EXPECT(Tv(1, 0) == A(0, 1));
    Tv(2, 1) = T(7);
    EXPECT(A(1, 2) == T(7));
    auto TvConst = transpose(CA);
    EXPECT(TvConst(0, 1) == A(1, 0));

    auto Hv = hermitian(A);
    EXPECT(std::abs(Hv(0, 1) - conj(A(1, 0))) < tol<T>());

    EXPECT(std::abs(Hv.at(0,1) - conj(A(1,0))) < tol<T>());

    Matrix<T, L> B(4, 4, T(0));
    auto Bv = view(B);
    Bv = expr(A);
    EXPECT(residual_small(A, B, tol<T>()));

    auto v = random_vector<T>(6, rng);
    auto vv = view(v);
    EXPECT(vv.size() == 6);
    vv(0) = T(123);
    EXPECT(v[0] == T(123));
    const Vector<T>& cv = v;
    auto cvv = view(cv);
    EXPECT(cvv(0) == T(123));
};

void run_core_tests() {
    RUN_TEST((t_vector_basics<double, Layout::RowMajor>));
    RUN_TEST((t_vector_basics<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_matrix_basics<double, Layout::RowMajor>));
    RUN_TEST((t_matrix_basics<double, Layout::ColMajor>));
    RUN_TEST((t_matrix_basics<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_matrix_basics<std::complex<double>, Layout::ColMajor>));
    RUN_TEST((t_views<double, Layout::RowMajor>));
    RUN_TEST((t_views<double, Layout::ColMajor>));
    RUN_TEST((t_views<std::complex<double>, Layout::RowMajor>));
    RUN_TEST((t_views<std::complex<double>, Layout::ColMajor>));
};