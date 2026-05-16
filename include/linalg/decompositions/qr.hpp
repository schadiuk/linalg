#pragma once

#include <linalg/blas/level3.hpp>
#include <linalg/norms/matrix_norms.hpp>

namespace linalg {
    enum class QRMode {Reduced, Complete, R};

    // Publicly-visible decomposition package
    template<typename T, Layout LL>
    struct QRResult {
        Matrix<T, LL> P;
        Matrix<T, LL> Q;
        Matrix<T, LL> R;
        Vector<size_t> piv;
        int rank;
        bool pivoted;
    };

    namespace detail {
        constexpr size_t QR_BLOCK = 64;
        // Returns pair (v, beta): (I - beta*v*v^H)*x = norm(x)*sign(x[0])*e_0
        template<typename T>
        std::pair<Vector<T>, double> householder_reflector(const Vector<T>& x) {
            const size_t n = x.size();
            Vector<T> v = x;
            double sigma = 0.0;
            for (size_t i = 1; i < n; ++i) sigma += std::norm(x[i]);
            const double xnorm = std::sqrt(std::norm(x[0]) + sigma);
            if (xnorm == 0.0) return {v, 0.0};
            T alpha;
            if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
                using R = typename T::value_type;
                const double x0_abs = std::abs(x[0]);
                const T phase = (x0_abs == 0.0) ? T(1) : x[0] / static_cast<R>(x0_abs);
                alpha = -phase * static_cast<R>(xnorm);
            } else {
                alpha = (x[0] >= T(0)) ? -static_cast<T>(xnorm) : static_cast<T>(xnorm);
            };
            v[0] = x[0] - alpha;
            const double vHv  = std::norm(v[0]) + sigma;
            const double beta = (vHv == 0.0) ? 0.0 : 2.0 / vHv;
            return {v, beta};
        };

        template<typename T, Layout L>
        double col_norm_sq(const Matrix<T, L>& W, size_t col, size_t row0) {
            double s = 0.0;
            for (size_t i = row0; i < W.rows(); ++i) s += std::norm(W(i, col));
            return s;
        };

        template<typename T, Layout L>
        void apply_householder_left(Matrix<T, L>& W, const Vector<T>& v, double beta, size_t k, size_t len, size_t ncols) {
            if (beta == 0.0 || ncols == 0 || len == 0) return;
            const T b = static_cast<T>(beta);
            const size_t lda = W.stride();
            T* const W_base = W.data() + k * lda + k;
            // Materialise conj(v) for the transposed GEMV
            Vector<T> cv(len);
            for (size_t i = 0; i < len; ++i) cv[i] = linalg::conj(v[i]);
            // Step 1: w = W_sub^T * cv (layout-flipped raw GEMV)
            Vector<T> w(ncols, T(0));
            if constexpr (L == Layout::RowMajor)
                kernels::gemv_kernel_col(T(1), W_base, lda, cv.data(), size_t(1), T(0), w.data(), ncols, len);
            else
                kernels::gemv_kernel_row(T(1), W_base, lda, cv.data(), size_t(1), T(0), w.data(), ncols, len);
            // Step 2: W_sub -= beta * v * w^T  (GER; conj already absorbed in w via cv)
            MatrixView<T, L, false, false, true> W_sub(W_base, len, ncols, lda);
            ger(-b, v, w, W_sub);
        };

        template<typename T, Layout L>
        void apply_householder_right(Matrix<T, L>& Q, const Vector<T>& v, double beta, size_t k, size_t len) {
            if (beta == 0.0 || len == 0) return;
            const size_t m = Q.rows();
            if (m == 0) return;
            const T b = static_cast<T>(beta);
            const size_t lda = Q.stride();
            T* const Q_base = Q.data() + (L == Layout::RowMajor ? k : k * lda);
            // Step 1: w = Q_sub * v
            Vector<T> w(m, T(0));
            if constexpr (L == Layout::RowMajor)
                kernels::gemv_kernel_row(T(1), Q_base, lda, v.data(), size_t(1), T(0), w.data(), m, len);
            else
                kernels::gemv_kernel_col(T(1), Q_base, lda, v.data(), size_t(1), T(0), w.data(), m, len);
            // Step 2: Q_sub -= beta * w * v^H  (GERC)
            MatrixView<T, L, false, false, true> Q_sub(Q_base, m, len, lda);
            gerc(-b, w, v, Q_sub);
        };

        // larft: given n Householder vectors with associated betas builds n*n upper triangular WY T-matrix such that:
        // H_k * H_{k+1} *...* H_{k+n-1}  =  I − V * T * V^H
        template<typename T, Layout L>
        Matrix<T, L> larft(const std::vector<Vector<T>>& vs, const std::vector<double>& betas, size_t k, size_t n) {
            Matrix<T, L> T_mat(n, n, T(0));
            for (size_t j = 0; j < n; ++j) {
                T_mat(j, j) = static_cast<T>(betas[k + j]);
                if (j == 0 || betas[k + j] == 0.0) continue;
                const Vector<T>& vj = vs[k + j];
                const size_t len_j = vj.size(); // = m − (k + j)
                // z[l] = −beta_j * SUM_{ii} conj(vs[k+l][j−l+ii]) * vj[ii]
                Vector<T> z(j, T(0));
                for (size_t l = 0; l < j; ++l) {
                    if (betas[k + l] == 0.0) continue;
                    const Vector<T>& vl = vs[k + l];
                    const size_t offset = j - l;
                    T dp = T(0);
                    for (size_t ii = 0; ii < len_j; ++ii)
                        dp += linalg::conj(vl[offset + ii]) * vj[ii];
                    z[l] = -static_cast<T>(betas[k + j]) * dp;
                };
                // T[0:j, j] = T[0:j, 0:j] * z  (upper-triangular matrix-vector product)
                for (size_t l = 0; l < j; ++l) {
                    T acc = T(0);
                    for (size_t ll = l; ll < j; ++ll) acc += T_mat(l, ll) * z[ll];
                    T_mat(l, j) = acc;
                };
            };
            return T_mat;
        };

        // Compact-WY trailing update
        template<typename T, Layout L>
        void apply_wy_left(Matrix<T, L>& W, const std::vector<Vector<T>>& vs, const Matrix<T, L>& T_mat, size_t k, size_t nb, size_t j0) {
            const size_t m = W.rows();
            const size_t n = W.cols();
            const size_t len = m - k;
            const size_t ncols_trail = (j0 < n) ? n - j0 : 0;
            if (ncols_trail == 0 || len == 0 || nb == 0) return;
            // Build V: (m-k)*nb.  V[i,j] = vs[k+j][i] for i < vs[k+j].size(), else 0
            Matrix<T, L> V(len, nb, T(0));
            for (size_t j = 0; j < nb; ++j) {
                const Vector<T>& vj = vs[k + j];
                for (size_t i = 0; i < vj.size(); ++i) V(i, j) = vj[i];
            };
            // Extract W_trail: W[k:m, j0:n] ->  temporary (m-k) * ncols_trail
            const size_t copy_thresh = std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (ncols_trail + 1));
            Matrix<T, L> W_trail(len, ncols_trail);
            parallel_for(len, copy_thresh, [&](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i)
                    for (size_t jj = 0; jj < ncols_trail; ++jj)
                        W_trail(i, jj) = W(k + i, j0 + jj);
            });
            // GEMM 1: C = V^H * W_trail  (nb * ncols_trail)
            Matrix<T, L> C(nb, ncols_trail, T(0));
            gemm(T(1), hermitian(V), expr(W_trail), T(0), C);
            // GEMM 2: TC = T_mat * C  (nb * ncols_trail)
            Matrix<T, L> TC(nb, ncols_trail, T(0));
            gemm(T(1), expr(T_mat), expr(C), T(0), TC);
            // GEMM 3: W_trail -= V * TC
            gemm(-T(1), expr(V), expr(TC), T(1), W_trail);
            // Write W_trail back into W[k:m, j0:n]
            parallel_for(len, copy_thresh, [&](size_t rs, size_t re) {
                for (size_t i = rs; i < re; ++i)
                    for (size_t jj = 0; jj < ncols_trail; ++jj)
                        W(k + i, j0 + jj) = W_trail(i, jj);
            });
        };
    };

    //pivot vector -> permutation matrix (in A*P=Q*R convertion)
    template<typename T, Layout L = Layout::RowMajor>
    Matrix<T, L> piv_to_P(const Vector<size_t>& piv) {
        const size_t n = piv.size();
        Matrix<T, L> mat = Matrix<T, L>::zeros(n, n);
        for (size_t j = 0; j < n; ++j) {
            BOUNDS_CHECK(piv[j] < n);
            mat(piv[j], j) = T(1);
        };
        return mat;
    };
 
    template<typename T, Layout L>
    Matrix<T, L> perm_matrix(const QRResult<T, L>& res) {
        if (!res.pivoted) throw std::logic_error("QR was not computed with pivoting.");
        return piv_to_P<T, L>(res.piv);
    };
};