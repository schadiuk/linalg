#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Publicly-visible least-squares result for a single RHS.
    template<typename T>
    struct LstsqVecResult {
        Vector<T> x;        // Solution vector (length n).
        double residual;
        int rank;           // Numerical rank of LHS matrix.
    };

    // Publicly-visible least-squares result for multiple RHS.
    template<typename T, Layout L>
    struct LstsqMatResult {
        Matrix<T, L> X;            // Solution matrix (n * nrhs).
        Vector<double> residuals;
        int rank;                  // Numerical rank of LHS matrix.
    };

    namespace detail {
        LINALG_INLINE double auto_tol(size_t m, size_t n, const Vector<double>& s) {
            const double smax = (s.size() > 0) ? s[0] : 0.0;
            return static_cast<double>(std::max(m, n)) * std::numeric_limits<double>::epsilon() * smax;
        };

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

    /// @brief Least-squares solution of `A * x = b` via QR decomposition.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param tol Rank-detection tolerance.
    /// @return `LstsqVecResult` object: solution `x`, `residual`, numerical `rank` estimate.
    template<typename T, Layout L>
    LstsqVecResult<T> lstsq_qr(const Matrix<T, L>& A, const Vector<T>& b, double tol = -1.0) {
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

    /// @brief Least-squares solution of `A * X = B` via QR decomposition.
    /// @param A System's matrix.
    /// @param B RHS matrix.
    /// @param tol Rank-detection tolerance.
    /// @return `LstsqMatResult` object: solution `X`, per-column `residuals` vector, numerical `rank` estimate.
    template<typename T, Layout L>
    LstsqMatResult<T, L> lstsq_qr(const Matrix<T, L>& A, const Matrix<T, L>& B, double tol = -1.0) {
        const size_t m = A.rows();
        const size_t n = A.cols();
        const size_t nrhs = B.cols();
        BOUNDS_CHECK(B.rows() == m);

        QRResult<T, L> res = qr_pivoted(A, tol);
        const int r = res.rank;
        const size_t k = res.Q.cols();

        Matrix<T, L> C(k, nrhs, T(0));
        gemm(T(1), hermitian(res.Q), expr(B), T(0), C);

        Matrix<T, L> Xp(n, nrhs, T(0));
        for (size_t i = 0; i < static_cast<size_t>(r) && i < k; ++i)
            for (size_t j = 0; j < nrhs; ++j) Xp(i, j) = C(i, j);
        detail::triu_solve_mat(res.R, Xp, r);
        // Undo column permutation:
        Matrix<T, L> X(n, nrhs, T(0));
        for (size_t j = 0; j < n; ++j)
            for (size_t col = 0; col < nrhs; ++col) X(res.piv[j], col) = Xp(j, col);
        // Per-column residuals:
        Vector<double> residuals(nrhs, 0.0);
        for (size_t j = 0; j < nrhs; ++j) {  // Note: fixed loop var (was using n)
            double nb2 = 0.0, nc2 = 0.0;
            for (size_t i = 0; i < m; ++i) nb2 += std::norm(B(i, j));
            for (size_t i = 0; i < k; ++i) nc2 += std::norm(C(i, j));
            residuals[j] = (nb2 > nc2) ? nb2 - nc2 : 0.0;
        };

        return { std::move(X), std::move(residuals), r };
    };

    /// @brief Least-squares solution of `A * X = B` via QR decomposition.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param tol Rank-detection tolerance.
    /// @return `LstsqVecResult` object.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqVecResult<T> lstsq_qr(const MatExpr<EA>& A, const VecExpr<EB>& b, double tol = -1.0) {
        return lstsq_qr(Matrix<T, L>(A), Vector<T>(b), tol);
    };
 
    /// @brief Least-squares solution of `A * X = B` via QR decomposition.
    /// @param A System's matrix.
    /// @param B RHS matrix.
    /// @param tol Rank-detection tolerance.
    /// @return `LstsqMatResult` object.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqMatResult<T, L> lstsq_qr(const MatExpr<EA>& A, const MatExpr<EB>& B, double tol = -1.0) {
        return lstsq_qr(Matrix<T, L>(A), Matrix<T, L>(B), tol);
    };

    /// @brief SVD-based least-squares solution of `A * x = b`.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param tol Singular-value cutoff for rank detection.
    /// @return `LstsqVecResult` object.
    template<typename T, Layout L>
    LstsqVecResult<T> lstsq_svd(const Matrix<T, L>& A, const Vector<T>& b, double tol = -1.0) {
        const size_t m = A.rows(), n = A.cols();
        BOUNDS_CHECK(b.size() == m);
        SVDResult<T, L> res = svd(A);
        const size_t k = res.s.size();
        if (tol < 0.0) tol = detail::auto_tol(m, n, res.s);

        Vector<T> c(k, T(0));
        gemv(T(1), hermitian(res.U), expr(b), T(0), c);

        Vector<T> y(k, T(0));
        int rank = 0;
        for (size_t i = 0; i < k; ++i) {
            if (res.s[i] <= tol) continue;
            y[i] = c[i] / static_cast<T>(res.s[i]);
            ++rank;
        };

        Vector<T> x(n, T(0));
        gemv(T(1), expr(res.V), expr(y), T(0), x);

        const double nb = nrm2(expr(b)), nc = nrm2(expr(c));
        const double residual = (nb * nb > nc * nc) ? nb * nb - nc * nc : 0.0;
 
        return { std::move(x), residual, rank };
    };

    /// @brief SVD-based least-squares solution of `A * X = B`.
    /// @param A System's matrix.
    /// @param B RHS matrix.
    /// @param tol Singular-value cutoff for rank detection.
    /// @return `LstsqMatResult` object: minimum-norm solution `X`, per-column `residuals`, numerical `rank`.
    template<typename T, Layout L>
    LstsqMatResult<T, L> lstsq_svd(const Matrix<T, L>& A, const Matrix<T, L>& B, double tol = -1.0) {
        const size_t m = A.rows(), n = A.cols();
        const size_t nrhs = B.cols();
        BOUNDS_CHECK(B.rows() == m);
        SVDResult<T, L> res = svd(A);
        const size_t k = res.s.size();
        if (tol < 0.0) tol = detail::auto_tol(m, n, res.s);

        Matrix<T, L> C(k, nrhs, T(0));
        gemm(T(1), hermitian(res.U), expr(B), T(0), C);

        std::vector<T> inv_s(k, T(0));
        int rank = 0;
        for (size_t i = 0; i < k; ++i)
            if (res.s[i] > tol) { inv_s[i] = static_cast<T>(1.0 / res.s[i]); ++rank; };

        Matrix<T, L> Y(k, nrhs, T(0));
        parallel_for(k, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (nrhs + 1)), [&](size_t is, size_t ie) {
            for (size_t i = is; i < ie; ++i) {
                if (inv_s[i] == T(0)) continue;
                LINALG_VECTORIZE
                for (size_t j = 0; j < nrhs; ++j) Y(i, j) = C(i, j) * inv_s[i];
            };
        });

        Matrix<T, L> X(n, nrhs, T(0));
        gemm(T(1), expr(res.V), expr(Y), T(0), X);

        Vector<double> residuals(nrhs, 0.0);
        parallel_for(nrhs, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (m + k + 1)), [&](size_t js, size_t je) {
            for (size_t j = js; j < je; ++j) {
                double nb2 = 0.0, nc2 = 0.0;
                LINALG_VECTORIZE
                for (size_t i = 0; i < m; ++i) nb2 += std::norm(B(i, j));
                LINALG_VECTORIZE
                for (size_t i = 0; i < k; ++i) nc2 += std::norm(C(i, j));
                residuals[j] = (nb2 > nc2) ? nb2 - nc2 : 0.0;
            };
        });
 
        return { std::move(X), std::move(residuals), rank };
    };
 
    /// @brief SVD-based least-squares solution of `A * x = b`.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param tol Singular-value cutoff for rank detection.
    /// @return `LstsqVecResult` object.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqVecResult<T> lstsq_svd(const MatExpr<EA>& A, const VecExpr<EB>& b, double tol = -1.0) {
        return lstsq_svd(Matrix<T, L>(A), Vector<T>(b), tol);
    };

    /// @brief SVD-based least-squares solution of `A * X = B`.
    /// @param A System's matrix.
    /// @param B RHS matrix.
    /// @param tol Singular-value cutoff for rank detection.
    /// @return `LstsqMatResult` object: minimum-norm solution `X`, per-column `residuals`, numerical `rank`.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqMatResult<T, L> lstsq_svd(const MatExpr<EA>& A, const MatExpr<EB>& B, double tol = -1.0) {
        return lstsq_svd(Matrix<T, L>(A), Matrix<T, L>(B), tol);
    };

    /// @brief Least-squares solution of `A * x = b`.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param driver LS algorithm (`qr` or `svd`)
    /// @param tol Tolerance.
    /// @return Corresponding `LstsqVecResult` structure.
    template<typename T, Layout L>
    LstsqVecResult<T> lstsq(const Matrix<T, L>& A, const Vector<T>& b, std::string driver = "svd", double tol = -1.0) {
        if (driver == "qr") return lstsq_qr(A, b, tol);
        else if (driver == "svd") return lstsq_svd(A, b, tol);
        else throw std::invalid_argument("Unrecognised driver: '" + driver + "' .");
    };

    /// @brief Least-squares solution of `A * X = B`.
    /// @param A System's matrix.
    /// @param b RHS matrix.
    /// @param driver LS algorithm (`qr` or `svd`)
    /// @param tol Tolerance.
    /// @return Corresponding `LstsqMatResult` structure.
    template<typename T, Layout L>
    LstsqMatResult<T, L> lstsq(const Matrix<T, L>& A, const Matrix<T, L>& B, std::string driver = "svd", double tol = -1.0) {
        if (driver == "qr") return lstsq_qr(A, B, tol);
        else if (driver == "svd") return lstsq_svd(A, B, tol);
        else throw std::invalid_argument("Unrecognised driver: '" + driver + "' .");
    };

    /// @brief Least-squares solution of `A * x = b`.
    /// @param A System's matrix.
    /// @param b RHS vector.
    /// @param driver LS algorithm (`qr` or `svd`)
    /// @param tol Tolerance.
    /// @return Corresponding `LstsqVecResult` structure.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqVecResult<T> lstsq(const MatExpr<EA>& A, const VecExpr<EB>& b, std::string driver = "svd", double tol = -1.0) {
        return lstsq(Matrix<T, L>(A), Vector<T>(b), driver, tol);
    };

    /// @brief Least-squares solution of `A * X = B`.
    /// @param A System's matrix.
    /// @param b RHS matrix.
    /// @param driver LS algorithm (`qr` or `svd`)
    /// @param tol Tolerance.
    /// @return Corresponding `LstsqMatResult` structure.
    template<typename T, Layout L, typename EA, typename EB>
    LstsqMatResult<T, L> lstsq(const MatExpr<EA>& A, const MatExpr<EB>& B, std::string driver = "svd", double tol = -1.0) {
        return lstsq(Matrix<T, L>(A), Matrix<T, L>(B), driver, tol);
    };

    /// @brief Moore-Penrose pseudoinverse.
    /// @param A `m * n` input matrix.
    /// @param tol Tolerance (setting negative value triggers auto-tolerance).
    /// @return `n * m` pseudoinverse matrix.
    template<typename T, Layout L>
    Matrix<T, L> pinv(const Matrix<T, L>& A, double tol = -1.0) {
        const size_t m = A.rows(), n = A.cols();
        SVDResult<T, L> res = svd(A);
        const size_t k = res.s.size();
        if (tol < 0.0) tol = detail::auto_tol(m, n, res.s);

        std::vector<T> inv_s(k, T(0));
        for (size_t i = 0; i < k; ++i)
            if (res.s[i] > tol) inv_s[i] = static_cast<T>(1.0 / res.s[i]);

        Matrix<T, L> Vs(n, k, T(0));
        parallel_for(n, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (k + 1)), [&](size_t rs, size_t re) {
            for (size_t r = rs; r < re; ++r) {
                LINALG_VECTORIZE
                for (size_t i = 0; i < k; ++i) Vs(r, i) = res.V(r, i) * inv_s[i];
            };
        });

        Matrix<T, L> P(n, m, T(0));
        gemm(T(1), expr(Vs), hermitian(res.U), T(0), P);
        return P;
    };

    /// @brief Moore-Penrose pseudoinverse.
    /// @param A `m * n` input matrix.
    /// @param tol Tolerance (setting negative value triggers auto-tolerance).
    /// @return `n * m` pseudoinverse matrix.
    template<typename T, Layout L, typename E>
    Matrix<T, L> pinv(const MatExpr<E>& e, double tol = -1.0) { return pinv(Matrix<T, L>(e), tol); };
};