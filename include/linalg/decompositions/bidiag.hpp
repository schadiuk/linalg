#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Bidiagonalisation package: `A = U * B * V^H`, `B` real bidiagonal.
    template<typename T, Layout LL>
    struct BidiagResult {
        Matrix<T, LL> U;    // Left Householder product.
        Vector<double> d;   // Main diagonal of `B`, length `k = min(m, n)`.
        Vector<double> e;   // Superdiagonal of `B`, length `k - 1`.
        Matrix<T, LL> V;    // Right Householder product (non-transposed).
    };

    namespace detail {
        // Multiplies column `j` of `M` by `phase[j]` for complex `T`.
        template<typename T, Layout L>
        void apply_col_phase(Matrix<T, L>& M, const std::vector<T>& phase) {
            if constexpr(!is_complex_v<T>) return;
            const size_t m = M.rows(), k = phase.size();
            if (k == 0 || m == 0) return;
            parallel_for(m, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (k + 1)),
                [&](size_t rs, size_t re) {
                    for (size_t i = rs; i < re; ++i)
                        for (size_t j = 0; j < k; ++j) M(i, j) *= phase[j];
                });
        };

        template<typename T>
        LINALG_INLINE void real_bidiag(std::vector<T>& diag, std::vector<T>& super, std::vector<T>& dl, std::vector<T>& dr) {
            const size_t k = diag.size();
            dl.assign(k, T(1));
            dr.assign(k, T(1));
            if constexpr(!is_complex_v<T>) return;

            using R = real_type_t<T>;
            // Select unit-modulus factor x such that x * z is real nonnegative.
            auto conj_phase = [](T z) -> T {
                const double m = std::abs(z);
                return (m == 0.0) ? T(1) : conj(z) / static_cast<R>(m);
            };
            for (size_t i = 0; i < k; ++i) {
                // dr[0] = 1 by convention; dl[i] must satisfy dl[i] * dr[i] * diag_original[i] real >= 0.
                const T val = dr[i] * diag[i];
                dl[i] = conj_phase(val);
                diag[i] = T(std::abs(val));
                if (i + 1 < k) {
                    const T eval = dl[i] * super[i];
                    dr[i + 1] = conj_phase(eval);
                    super[i] = T(std::abs(eval));
                };
            };
        };

        template<typename T, Layout L>
        BidiagResult<T, L> bidiag_finalise_tall(size_t m, size_t n, size_t k, const Matrix<T, L>& W,
                const std::vector<Vector<T>>& us, const std::vector<double>& ubeta,
                const std::vector<Vector<T>>& vs, const std::vector<double>& vbeta, bool accumulate_uv) {
            BidiagResult<T, L> res;
            std::vector<T> draw(k), eraw(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i < k; ++i) draw[i] = W(i, i);
            for (size_t i = 0; i + 1 < k; ++i) eraw[i] = W(i, i + 1);
 
            std::vector<T> dl, dr;
            real_bidiag(draw, eraw, dl, dr);
            res.d = Vector<double>(k);
            for (size_t i = 0; i < k; ++i) res.d[i] = std::real(static_cast<T>(draw[i]));
            res.e = Vector<double>(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i + 1 < k; ++i) res.e[i] = std::real(static_cast<T>(eraw[i]));
 
            if (!accumulate_uv) return res;
            // U accumulation: apply left reflectors in reverse order onto I_m (first k columns).
            res.U = Matrix<T, L>(m, k, T(0));
            for (size_t i = 0; i < k; ++i) res.U(i, i) = T(1);
            for (size_t ci = k; ci-- > 0; ) {
                if (ubeta[ci] == 0.0) continue;
                const size_t len = m - ci;
                const size_t ncols = k - ci;
                apply_householder_left(res.U, us[ci], ubeta[ci], ci, len, ncols);
            };
            // V accumulation:
            res.V = Matrix<T, L>(n, k, T(0));
            for (size_t i = 0; i < k; ++i) res.V(i, i) = T(1);
            for (size_t ci = 0; ci < vs.size(); ++ci) {
                if (vbeta[ci] == 0.0) continue;
                const size_t rlen = n - (ci + 1);
                apply_householder_right(res.V, vs[ci], vbeta[ci], ci + 1, rlen);
            };
 
            // Fold the realizing phases in: `U <- U * D_L^H` (column `i` scaled by `conj(dl[i])`), `V <- V * D_R` (column `i` scaled by `dr[i]`).
            if constexpr (is_complex_v<T>) {
                std::vector<T> dl_conj(k);
                for (size_t i = 0; i < k; ++i) dl_conj[i] = conj(dl[i]);
                apply_col_phase(res.U, dl_conj);
                apply_col_phase(res.V, dr);
            };

            return res;
        };

        template<typename T, Layout L>
        BidiagResult<T, L> bidiag_tall_unblocked(const Matrix<T, L>& A, bool accumulate_uv) {
            const size_t m = A.rows(), n = A.cols();
            const size_t k = n;
            Matrix<T, L> W = A;
            std::vector<Vector<T>> us(k);
            std::vector<double> ubeta(k, 0.0);
            std::vector<Vector<T>> vs(k > 0 ? k - 1 : 0);
            std::vector<double> vbeta(k > 0 ? k - 1 : 0, 0.0);
 
            for(size_t col = 0; col < k; ++col) {
                const size_t len = m - col;
                Vector<T> x(len);
                for (size_t i = 0; i < len; ++i) x[i] = W(col + i, col);
                auto [u, ubet] = householder_reflector(x);
                us[col] = u; ubeta[col] = ubet;
                apply_householder_left(W, u, ubet, col, len, n - col);
                for (size_t i = col + 1; i < m; ++i) W(i, col) = T(0);
                // Right reflector: zero W[col, col+2:n], acting on cols [col+1, n).
                if (col + 1 < n) {
                    const size_t rlen = n - (col + 1);
                    Vector<T> y(rlen);
                    for (size_t j = 0; j < rlen; ++j) y[j] = conj(W(col, col + 1 + j));
                    auto [v, vbet] = householder_reflector(y);
                    vs[col] = v; vbeta[col] = vbet;
                    apply_householder_right(W, v, vbet, col + 1, rlen);
                    for (size_t j = col + 2; j < n; ++j) W(col, j) = T(0);
                };
            };
 
            return bidiag_finalise_tall<T, L>(m, n, k, W, us, ubeta, vs, vbeta, accumulate_uv);
        };

        constexpr size_t BIDIAG_BLOCK = 64;

        namespace kernels {
            template<typename T, Layout L>
            void labrd(Matrix<T, L>& Wp, size_t M, size_t N, size_t nb, std::vector<Vector<T>>& u1s, std::vector<double>& utau,
                    std::vector<Vector<T>>& v1s, std::vector<double>& vtau, Matrix<T, L>& X, Matrix<T, L>& Y) {
                X = Matrix<T, L>(M, nb, T(0));
                Y = Matrix<T, L>(N, nb, T(0));
                u1s.clear(); utau.clear(); v1s.clear(); vtau.clear();
                u1s.reserve(nb); utau.reserve(nb); v1s.reserve(nb); vtau.reserve(nb);
                std::vector<T> d_local, e_local;
                d_local.reserve(nb); e_local.reserve(nb);
    
                for (size_t i = 0; i < nb; ++i) {
                    const size_t len_u = M - i;

                    std::vector<T> yvec(i), xvec(i);
                    for (size_t l = 0; l < i; ++l) { yvec[l] = conj(Y(i, l)); xvec[l] = Wp(l, i); };
                    parallel_for(len_u, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
                        for (size_t r = rs; r < re; ++r) {
                            T s = T(0);
                            LINALG_VECTORIZE
                            for (size_t l = 0; l < i; ++l) s += Wp(i + r, l) * yvec[l] + X(i + r, l) * xvec[l];
                            Wp(i + r, i) -= s;
                        };
                    });
    
                    // Left (Q-side) reflector annihilating Wp(i+1:M-1, i)
                    Vector<T> xcol(len_u);
                    parallel_for(len_u, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) { for (size_t r = rs; r < re; ++r) xcol[r] = Wp(i + r, i); });
                    auto [v, beta] = householder_reflector(xcol);
                    Vector<T> v1(len_u, T(0));
                    double tq = 0.0;
                    T d_i;
                    if (beta == 0.0) {
                        v1[0] = T(1);
                        d_i = xcol[0];
                    } else {
                        const T c = v[0];
                        tq = beta * std::norm(c);
                        parallel_for(len_u, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) { for (size_t r = rs; r < re; ++r) v1[r] = v[r] / c; });
                        v1[0] = T(1);
                        d_i = xcol[0] - v[0];
                    };
                    Wp(i, i) = T(1); // packed convention: forced to 1 for later steps' reads; true value recorded in d_local.

                    parallel_for(len_u > 0 ? len_u - 1 : 0, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) { for (size_t r = rs; r < re; ++r) Wp(i + 1 + r, i) = v1[1 + r]; });
                    u1s.push_back(v1); utau.push_back(tq);
                    d_local.push_back(d_i);
    
                    if (i + 1 < N) {
                        const size_t ncol_trail = N - i - 1;
                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) {
                                T s = T(0);
                                LINALG_VECTORIZE
                                for (size_t r = 0; r < len_u; ++r) s += conj(Wp(i + r, i + 1 + c2)) * v1[r];
                                Y(i + 1 + c2, i) = s;
                            };
                        });

                        std::vector<T> ybuf(i, T(0));
                        for (size_t l = 0; l < i; ++l) {
                            ybuf[l] = parallel_reduce<T>(len_u, PARALLEL_THRESHOLD_REDUCE, [&](size_t r) { return conj(Wp(i + r, l)) * v1[r]; });
                        };

                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) {
                                T s = T(0);
                                LINALG_VECTORIZE
                                for (size_t l = 0; l < i; ++l) s += Y(i + 1 + c2, l) * ybuf[l];
                                Y(i + 1 + c2, i) -= s;
                            };
                        });

                        std::vector<T> ybuf2(i, T(0));
                        for (size_t l = 0; l < i; ++l) {
                            ybuf2[l] = parallel_reduce<T>(len_u, PARALLEL_THRESHOLD_REDUCE, [&](size_t r) { return conj(X(i + r, l)) * v1[r]; });
                        };

                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) {
                                T s = T(0);
                                LINALG_VECTORIZE
                                for (size_t l = 0; l < i; ++l) s += conj(Wp(l, i + 1 + c2)) * ybuf2[l];
                                Y(i + 1 + c2, i) -= s;
                            };
                        });

                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_SIMPLE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) Y(i + 1 + c2, i) *= static_cast<T>(tq);
                        });
    
                        // Row correction: Wp(i,i+1:N-1) -= conj(Y(i+1:N-1,0:i))^T*Wp(i,0:i) + Wp(0:i-1,i+1:N-1)^T*X(i,0:i-1)
                        std::vector<T> wrow(i + 1);
                        for (size_t l = 0; l <= i; ++l) wrow[l] = Wp(i, l);
                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) {
                                T s = T(0);
                                LINALG_VECTORIZE
                                for (size_t l = 0; l <= i; ++l) s += conj(Y(i + 1 + c2, l)) * wrow[l];
                                Wp(i, i + 1 + c2) -= s;
                            };
                        });
                    
                        std::vector<T> xrow(i);
                        for (size_t l = 0; l < i; ++l) xrow[l] = X(i, l);
                        parallel_for(ncol_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) {
                                T s = T(0);
                                LINALG_VECTORIZE
                                for (size_t l = 0; l < i; ++l) s += Wp(l, i + 1 + c2) * xrow[l];
                                Wp(i, i + 1 + c2) -= s;
                            };
                        });
                        
                        // Right (P-side) reflector on row i, cols i+1..N-1 (acting on the conjugated row):
                        const size_t len_v = ncol_trail;
                        Vector<T> yrow(len_v);
                        parallel_for(len_v, PARALLEL_THRESHOLD_SIMPLE, [&](size_t cs, size_t ce) { for (size_t c2 = cs; c2 < ce; ++c2) yrow[c2] = conj(Wp(i, i + 1 + c2)); });
                        auto [vv, vbeta_] = householder_reflector(yrow);
                        Vector<T> u1v(len_v, T(0));
                        double tp = 0.0;
                        T e_i;
                        if (vbeta_ == 0.0) {
                            u1v[0] = T(1);
                            e_i = yrow[0];
                        } else {
                            const T cc = vv[0];
                            tp = vbeta_ * std::norm(cc);
                            parallel_for(len_v, PARALLEL_THRESHOLD_SIMPLE, [&](size_t cs, size_t ce) { for (size_t c2 = cs; c2 < ce; ++c2) u1v[c2] = vv[c2] / cc; });
                            u1v[0] = T(1);
                            e_i = yrow[0] - vv[0];
                        };
                        
                        Wp(i, i + 1) = T(1); // Packed convention.
                        parallel_for(len_v > 0 ? len_v - 1 : 0, PARALLEL_THRESHOLD_SIMPLE, [&](size_t cs, size_t ce) {
                            for (size_t c2 = cs; c2 < ce; ++c2) Wp(i, i + 2 + c2) = conj(u1v[1 + c2]);
                        });
                        v1s.push_back(u1v); vtau.push_back(tp);
                        e_local.push_back(conj(e_i));

                        if (i + 1 < M) {
                            const size_t mrow_trail = M - i - 1;
                            parallel_for(mrow_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
                                for (size_t r = rs; r < re; ++r) {
                                    T s = T(0);
                                    LINALG_VECTORIZE
                                    for (size_t c2 = 0; c2 < len_v; ++c2) s += Wp(i + 1 + r, i + 1 + c2) * u1v[c2];
                                    X(i + 1 + r, i) = s;
                                };
                            });
                            // xbuf[l] = conj(Y(i+1:N-1,l))^T * u1v, l = 0...i
                            std::vector<T> xbuf(i + 1, T(0));
                            for (size_t l = 0; l <= i; ++l) {
                                xbuf[l] = parallel_reduce<T>(len_v, PARALLEL_THRESHOLD_REDUCE, [&](size_t c2) { return conj(Y(i + 1 + c2, l)) * u1v[c2]; });
                            };

                            parallel_for(mrow_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
                                for (size_t r = rs; r < re; ++r) {
                                    T s = T(0);
                                    LINALG_VECTORIZE
                                    for (size_t l = 0; l <= i; ++l) s += Wp(i + 1 + r, l) * xbuf[l]; // l==i reads the packed v1 tail.
                                    X(i + 1 + r, i) -= s;
                                };
                            });

                            std::vector<T> xbuf2(i, T(0));
                            for (size_t l = 0; l < i; ++l) {
                                xbuf2[l] = parallel_reduce<T>(len_v, PARALLEL_THRESHOLD_REDUCE, [&](size_t c2) { return Wp(l, i + 1 + c2) * u1v[c2]; });
                            };

                            parallel_for(mrow_trail, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
                                for (size_t r = rs; r < re; ++r) {
                                    T s = T(0);
                                    LINALG_VECTORIZE
                                    for (size_t l = 0; l < i; ++l) s += X(i + 1 + r, l) * xbuf2[l];
                                    X(i + 1 + r, i) -= s;
                                };
                            });

                            parallel_for(mrow_trail, PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) {
                                for (size_t r = rs; r < re; ++r) X(i + 1 + r, i) *= static_cast<T>(tp);
                            });
                        };
                    };
                };

                // Unpack: restore the true diagonal/superdiagonal values over the forced-1 placeholders.
                for (size_t i = 0; i < nb; ++i) {
                    Wp(i, i) = d_local[i];
                    if (i < e_local.size()) Wp(i, i + 1) = e_local[i];
                };
            };
        };

        template<typename T, Layout L>
        BidiagResult<T, L> bidiag_tall_blocked(const Matrix<T, L>& A, bool accumulate_uv) {
            const size_t m = A.rows(), n = A.cols();
            const size_t k = n;
            Matrix<T, L> W = A;
            std::vector<Vector<T>> us(k), vs(k > 0 ? k - 1 : 0);
            std::vector<double> ubeta(k, 0.0), vbeta(k > 0 ? k - 1 : 0, 0.0);

            size_t kk = 0;
            while (kk < k) {
                const size_t nb = std::min(BIDIAG_BLOCK, k - kk);
                const size_t M = m - kk, N = n - kk;
                // Extract the remaining (M*N) submatrix into a tight local working copy.
                Matrix<T, L> Wp(M, N);
                parallel_for(M, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (N + 1)), [&](size_t is, size_t ie) {
                    for (size_t i = is; i < ie; ++i)
                        for (size_t j = 0; j < N; ++j) Wp(i, j) = W(kk + i, kk + j);
                });

                std::vector<Vector<T>> u1s, v1s;
                std::vector<double> utau, vtau;
                Matrix<T, L> X, Y;

                kernels::labrd(Wp, M, N, nb, u1s, utau, v1s, vtau, X, Y);

                for (size_t i = 0; i < u1s.size(); ++i) { us[kk + i] = u1s[i]; ubeta[kk + i] = utau[i]; };
                for (size_t i = 0; i < v1s.size(); ++i) { vs[kk + i] = v1s[i]; vbeta[kk + i] = vtau[i]; };

                const size_t Mtrail = M - nb, Ntrail = N - nb;
                if (Mtrail > 0 && Ntrail > 0) {
                    Matrix<T, L> Vtrail(Mtrail, nb), Ytrail(Ntrail, nb);
                    Matrix<T, L> Xtrail(Mtrail, nb), Utrail(nb, Ntrail, T(0));
                    parallel_for(nb, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (Mtrail + Ntrail + 1)),
                        [&](size_t is, size_t ie) {
                            for (size_t i = is; i < ie; ++i) {
                                for (size_t r = 0; r < Mtrail; ++r) Vtrail(r, i) = u1s[i][nb + r - i];
                                for (size_t r = 0; r < Mtrail; ++r) Xtrail(r, i) = X(nb + r, i);
                                for (size_t c = 0; c < Ntrail; ++c) Ytrail(c, i) = Y(nb + c, i);
                                for (size_t c = 0; c < Ntrail; ++c) Utrail(i, c) = conj(v1s[i][nb + c - i - 1]);
                            };
                        });
 
                    Matrix<T, L> Atrail(Mtrail, Ntrail);
                    parallel_for(Mtrail, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (Ntrail + 1)), [&](size_t rs, size_t re) {
                        for (size_t r = rs; r < re; ++r)
                            for (size_t c = 0; c < Ntrail; ++c) Atrail(r, c) = Wp(nb + r, nb + c);
                    });
                    // A_trail -= V*Y^H + X*U (the two deferred corrections):
                    gemm(T(-1), expr(Vtrail), hermitian(Ytrail), T(1), Atrail);
                    gemm(T(-1), expr(Xtrail), expr(Utrail), T(1), Atrail);

                    parallel_for(Mtrail, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (Ntrail + 1)), [&](size_t rs, size_t re) {
                        for (size_t r = rs; r < re; ++r)
                            for (size_t c = 0; c < Ntrail; ++c) Wp(nb + r, nb + c) = Atrail(r, c);
                    });
                };

                for (size_t i = 0; i < nb; ++i) {
                    const size_t g = kk + i;
                    W(g, g) = Wp(i, i);
                    parallel_for(m - (g + 1), PARALLEL_THRESHOLD_SIMPLE, [&](size_t rs, size_t re) { for (size_t r = rs; r < re; ++r) W(g + 1 + r, g) = T(0); });
                    if (g + 1 < n) {
                        W(g, g + 1) = Wp(i, i + 1);
                        parallel_for(n - (g + 2), PARALLEL_THRESHOLD_SIMPLE, [&](size_t cs, size_t ce) { for (size_t c = cs; c < ce; ++c) W(g, g + 2 + c) = T(0); });
                    };
                };

                parallel_for(Mtrail, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / (Ntrail + 1)), [&](size_t rs, size_t re) {
                    for (size_t r = rs; r < re; ++r)
                        for (size_t c = 0; c < Ntrail; ++c) W(kk + nb + r, kk + nb + c) = Wp(nb + r, nb + c);
                });
 
                kk += nb;
            };
 
            return bidiag_finalise_tall<T, L>(m, n, k, W, us, ubeta, vs, vbeta, accumulate_uv);
        };

        // Dispatch: blocked path for matrices wide enough to amortise panel overhead.
        template<typename T, Layout L>
        BidiagResult<T, L> bidiag_tall(const Matrix<T, L>& A, bool accumulate_uv) {
            if (A.cols() > BIDIAG_BLOCK) return bidiag_tall_blocked(A, accumulate_uv);
            return bidiag_tall_unblocked(A, accumulate_uv);
        };

    };

    /// @brief Golub-Kahan bidiagonalisation: `A = U * B * V^H`, `B` real bidiagonal.
    /// @param A Matrix to be decomposed.
    /// @param accumulate_uv Householder products accumulation flag.
    /// @return Corresponding `BidiagResult` structure.
    /// @note Wide matrices (`n > m`) are handled by bidiagonalising `A^H` (tall) and swapping the resulting `U`/`V`.
    template<typename T, Layout L>
    BidiagResult<T, L> bidiag(const Matrix<T, L>& A, bool accumulate_uv = true) {
        const size_t m = A.rows(), n = A.cols();
        if (m < n) {
            Matrix<T, L> AH = hermitian(A);
            auto sub = detail::bidiag_tall(AH, accumulate_uv);
            const size_t k = sub.d.size();
            BidiagResult<T, L> res;
            res.d = Vector<double>(k);
            for (size_t i = 0; i < k; ++i) res.d[i] = sub.d[k - 1 - i];
            res.e = Vector<double>(k > 0 ? k - 1 : 0);
            for (size_t i = 0; i + 1 < k; ++i) res.e[i] = sub.e[k - 2 - i];
            if (accumulate_uv) {
                res.U = Matrix<T, L>(sub.V.rows(), k);
                for (size_t i = 0; i < sub.V.rows(); ++i)
                    for (size_t j = 0; j < k; ++j) res.U(i, j) = sub.V(i, k - 1 - j);
                res.V = Matrix<T, L>(sub.U.rows(), k);
                for (size_t i = 0; i < sub.U.rows(); ++i)
                    for (size_t j = 0; j < k; ++j) res.V(i, j) = sub.U(i, k - 1 - j);
            };
            return res;
        };
        return detail::bidiag_tall(A, accumulate_uv);
    };

    template<typename T, Layout L, typename E>
    BidiagResult<T, L> bidiag(const MatExpr<E>& e, bool accumulate_uv = true) {
        return bidiag(Matrix<T, L>(e), accumulate_uv);
    };
};