#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Publicly-visible least-squares result for a single RHS.
    template<typename T>
    struct LstsqVecResult {
        Vector<T> x; // Solution vector (length n).
        double residual;
        int rank; // Numerical rank of LHS matrix.
    };

    // Publicly-visible least-squares result for multpile RHS.
    template<typename T, Layout L>
    struct LstsqMatResult {
        Matrix<T, L> X; // Solution matrix (n * nrhs).
        Vector<double> residuals;
        int rank; // Numerical rank of LHS matrix.
    };

    namespace detail {
        // Solve R[0:r, 0:r] * sol[0:r] = rhs[0:r] by back-substitution (in-place).
        template<typename T, Layout L>
        LINALG_INLINE void triu_solve_vec(const Matrix<T, L>& R, Vector<T>& sol, int r) {
            const size_t n = sol.size();
            const size_t ur = static_cast<size_t>(r);
            for (size_t i = ur; i < n; ++i) sol[i] = T(0);
            for (int ii = r - 1; ii >= 0; --ii) {
                const size_t i = static_cast<size_t>(ii);
                T s = sol[i];
                for (size_t j = i + 1; j < ur; ++j) s -= static_cast<T>(R(i, j)) * sol[j];
                sol[i] = s / static_cast<T>(R(i, i));
            };
        };

        // Solve R[0:r, 0:r] * X[0:r, :] = RHS[0:r, :] column-wise.
        template<typename T, Layout L>
        LINALG_INLINE void triu_solve_mat(const Matrix<T, L>& R, Matrix<T, L>& RHS, int r) {
            const size_t nrhs = RHS.cols();
            const size_t n = RHS.rows();
            const size_t ur = static_cast<size_t>(r);
            for (size_t i = ur; i < n; ++i)
                for (size_t j = 0; j < nrhs; ++j)
                    RHS(i, j) = T(0);
            for (size_t col = 0; col < nrhs; ++col) {
                for (int ii = r - 1; ii >= 0; --ii) {
                    const size_t i = static_cast<size_t>(ii);
                    T s = RHS(i, col);
                    for (size_t j = i + 1; j < ur; ++j) s -= static_cast<T>(R(i, j)) * RHS(j, col);
                    RHS(i, col) = s / static_cast<T>(R(i, i));
                };
            };
        };
    };

    template<typename T, Layout L>
    LstsqVecResult<T> lstsq(const Matrix<T, L>& A, const Vector<T>& b, double tol = -1.0) {
        const size_t m = A.rows();
        const size_t n = A.cols();
        BOUNDS_CHECK(b.size() == m);
        // Pivoted QR.
        QRResult<T, L> res = qr_pivoted(A, tol);
        const int r = res.rank;
        const size_t k = res.Q.cols(); // min(m, n) for reduced QR.
        // c = Q^H * b
        Vector<T> c(k, T(0));
        gemv(T(1), hermitian(res.Q), expr(b), T(0), c);
        // Triangular solve on c[0:r] with result in xp (of length n).
        Vector<T> xp(n, T(0));
        for (size_t i = 0; i < static_cast<size_t>(r) && i < k; ++i) xp[i] = c[i];
        detail::triu_solve_vec(res.R, xp, r);

        Vector<T> x(n, T(0));
        for (size_t j = 0; j < n; ++j) x[res.piv[j]] = xp[j];
        // Residual: |b - Q * c|^2 = |b|^2 - |c|^2.
        double nb2 = 0.0, nc2 = 0.0;
        for (size_t i = 0; i < m; ++i) nb2 += std::norm(b[i]);
        for (size_t i = 0; i < k; ++i) nc2 += std::norm(c[i]);
        const double residual = (nb2 > nc2) ? nb2 - nc2 : 0.0;

        return { std::move(x), residual, r };
    };
};