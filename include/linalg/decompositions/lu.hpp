#pragma once

#include <linalg/blas/level3.hpp>

constexpr size_t LU_BLOCK = 64;

namespace linalg {
    // LUResult bundles the publicly-visible decomposition (P, L, U) together with the internal packed representation used by the solvers
    template<typename T, Layout LL>
    struct LUResult {
        Matrix<T, LL> P; // Permutation matrix: P * A = L * U
        Matrix<T, LL> L; // m*min(m,n) unit lower triangular factor
        Matrix<T, LL> U; // min(m,n)*n upper triangular factor
        Matrix<T, LL> packed; // In-place storage: strict lower L, upper U
        Vector<size_t> piv; // piv[j] = row swapped with row j at step j
    };

    namespace detail {
        // Blocked LU with partial pivoting
        // Algorithm per block column k:
        // 1. Unblocked panel LU on A[k:m, k:k+kb] with partial pivoting. Row swaps are applied to the full row (0:n) so previously-factored blocks are kept consistent
        // 2. Panel trsm: solve L_panel * U12 = A[k:k+kb, k+kb:n]
        // 3. Schur complement: A[k+kb:m, k+kb:n] -= L21 * U12
        template<typename T, Layout L>
        Vector<size_t> lu_factor(Matrix<T, L>& A) {
            const size_t m = A.rows();
            const size_t n = A.cols();
            const size_t mn = std::min(m, n);
            Vector<size_t> piv(mn);
            for (size_t i = 0; i < mn; ++i) piv[i] = i;
            // Parallelised submatrix copy helpers
            auto sub_copy_in = [&](Matrix<T, L>& dst, size_t r0, size_t c0) {
                const size_t nr = dst.rows(), nc = dst.cols();
                if constexpr (L == Layout::RowMajor) {
                    parallel_for(nr, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / nc),
                        [&](size_t s, size_t e) {
                            for (size_t i = s; i < e; ++i)
                                for (size_t j = 0; j < nc; ++j)
                                    dst(i, j) = A(r0 + i, c0 + j);
                        });
                } else {
                    parallel_for(nc, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / nr),
                        [&](size_t s, size_t e) {
                            for (size_t j = s; j < e; ++j)
                                for (size_t i = 0; i < nr; ++i)
                                    dst(i, j) = A(r0 + i, c0 + j);
                        });
                };
            };

            auto sub_copy_out = [&](const Matrix<T, L>& src, size_t r0, size_t c0) {
                const size_t nr = src.rows(), nc = src.cols();
                if constexpr (L == Layout::RowMajor) {
                    parallel_for(nr, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / nc),
                        [&](size_t s, size_t e) {
                            for (size_t i = s; i < e; ++i)
                                for (size_t j = 0; j < nc; ++j)
                                    A(r0 + i, c0 + j) = src(i, j);
                        });
                } else {
                    parallel_for(nc, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / nr),
                        [&](size_t s, size_t e) {
                            for (size_t j = s; j < e; ++j)
                                for (size_t i = 0; i < nr; ++i)
                                    A(r0 + i, c0 + j) = src(i, j);
                        });
                };
            };

            for (size_t k = 0; k < mn; k += LU_BLOCK) {
                const size_t kb = std::min(LU_BLOCK, mn - k);
                const size_t right_start = k + kb;
                // Unblocked panel LU
                for (size_t j = k; j < right_start; ++j) {
                    // Pivot search: find max abs(A(i,j)) for i in [j, m)
                    size_t pr = j;
                    auto   mx = std::abs(static_cast<T>(A(j, j)));
                    for (size_t i = j + 1; i < m; ++i) {
                        const auto v = std::abs(static_cast<T>(A(i, j)));
                        if (v > mx) { mx = v; pr = i; };
                    };
                    piv[j] = pr;
                    // Full-row swap: applied to columns 0:n
                    if (pr != j)
                        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE,
                            [&A, j, pr](size_t s, size_t e) {
                                for (size_t jj = s; jj < e; ++jj)
                                    std::swap(A(j, jj), A(pr, jj));
                            });
                    // Scale sub-diagonal entries
                    const T diag = static_cast<T>(A(j, j));
                    if (diag != T(0)) {
                        const T inv = T(1) / diag;
                        for (size_t i = j + 1; i < m; ++i) A(i, j) *= inv;
                    };
                    // Rank-1 update of the remaining panel columns within block
                    for (size_t i = j + 1; i < m; ++i) {
                        const T lij = static_cast<T>(A(i, j));
                        if (lij == T(0)) continue;   // skip zero multiplier rows
                        for (size_t jj = j + 1; jj < right_start; ++jj)
                            A(i, jj) -= lij * static_cast<T>(A(j, jj));
                    };
                };
 
                if (right_start >= n) continue;
                const size_t right_cols = n - right_start;
                // Panel trsm: L_panel * U12 = A[k:right_start, right_start:n]
                // The extracted panel contains both L (below) and U (on/above) in packed form
                Matrix<T, L> panel(kb, kb);
                for (size_t i = 0; i < kb; ++i)
                    for (size_t j = 0; j < kb; ++j)
                        panel(i, j) = static_cast<T>(A(k + i, k + j));
                Matrix<T, L> U12(kb, right_cols);
                sub_copy_in(U12, k, right_start);
                trsm('L', 'L', 'N', 'U', T(1), panel, U12);
                sub_copy_out(U12, k, right_start);
                if (right_start >= m) continue;
                const size_t below_rows = m - right_start;

                // Schur complement
                Matrix<T, L> L21(below_rows, kb);
                Matrix<T, L> C  (below_rows, right_cols);
                sub_copy_in(L21, right_start, k);
                sub_copy_in(C,   right_start, right_start);
                gemm(T(-1), L21, U12, T(1), C);
                sub_copy_out(C, right_start, right_start);
            };
            return piv;
        };
 
        // Reconstructs the permutation matrix
        template<typename T, Layout L>
        Matrix<T, L> piv_to_P(const Vector<size_t>& piv, size_t m) {
            Matrix<T, L> P = Matrix<T, L>::identity(m);
            for (size_t i = 0; i < piv.size(); ++i)
                if (piv[i] != i)
                    for (size_t j = 0; j < m; ++j)
                        std::swap(P(i, j), P(piv[i], j));
            return P;
        };
 
        template<typename T, Layout L>
        Matrix<T, L> unpack_L(const Matrix<T, L>& pk) {
            const size_t m = pk.rows();
            const size_t mn = std::min(m, pk.cols());
            Matrix<T, L> Lo(m, mn, T(0));
            if constexpr (L == Layout::RowMajor) {
                parallel_for(m, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / mn),
                    [&](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i)
                            for (size_t j = 0; j < mn; ++j) {
                                if (j == i) Lo(i, j) = T(1);
                                else if (j <  i) Lo(i, j) = pk(i, j);
                            };
                    });
            } else {
                parallel_for(mn, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / m),
                    [&](size_t cs, size_t ce) {
                        for (size_t j = cs; j < ce; ++j)
                            for (size_t i = 0; i < m; ++i) {
                                if (i == j) Lo(i, j) = T(1);
                                else if (i >  j) Lo(i, j) = pk(i, j);
                            };
                    });
            };
            return Lo;
        };
 
        template<typename T, Layout L>
        Matrix<T, L> unpack_U(const Matrix<T, L>& pk) {
            const size_t n = pk.cols();
            const size_t mn = std::min(pk.rows(), n);
            Matrix<T, L> Up(mn, n, T(0));
            if constexpr (L == Layout::RowMajor) {
                parallel_for(mn, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / n),
                    [&](size_t rs, size_t re) {
                        for (size_t i = rs; i < re; ++i)
                            for (size_t j = i; j < n; ++j)
                                Up(i, j) = pk(i, j);
                    });
            } else {
                parallel_for(n, std::max(size_t(1), PARALLEL_THRESHOLD_SIMPLE / mn),
                    [&](size_t cs, size_t ce) {
                        for (size_t j = cs; j < ce; ++j)
                            for (size_t i = 0; i <= j && i < mn; ++i)
                                Up(i, j) = pk(i, j);
                    });
            };
            return Up;
        };
    }; 

    // Public API for LU factorisation
    template<typename T, Layout L>
    LUResult<T, L> lu(const Matrix<T, L>& A) {
        Matrix<T, L> work = A;
        Vector<size_t> piv = detail::lu_factor(work);
        return LUResult<T, L> {
            detail::piv_to_P<T, L>(piv, A.rows()),
            detail::unpack_L<T, L>(work),
            detail::unpack_U<T, L>(work),
            std::move(work),
            std::move(piv)
        };
    };
 
    template<typename T, Layout L, typename E>
    LUResult<T, L> lu(const MatExpr<E>& e) {
        return lu(Matrix<T, L>(e));
    };

    // Single RHS solver
    template<typename T, Layout L>
    void lu_solve(const LUResult<T, L>& res, Vector<T>& b) {
        const auto& LU = res.packed;
        const auto& piv = res.piv;
        const size_t n = LU.rows();
        BOUNDS_CHECK(n == LU.cols() && n == b.size() && piv.size() == n);
        // Apply row permutation: b <- Pb
        for (size_t i = 0; i < n; ++i) if (piv[i] != i) std::swap(b[i], b[piv[i]]);
        // Forward substitution: L*x = b (unit lower triangular)
        trsv('L', 'N', 'U', LU, b);
        // Backward substitution: U*x = b (non-unit upper triangular)
        trsv('U', 'N', 'N', LU, b);
    };

    // Multiple RHS
    template<typename T, Layout L>
    void lu_solve(const LUResult<T, L>& res, Matrix<T, L>& B) {
        const auto& LU = res.packed;
        const auto& piv = res.piv;
        const size_t n = LU.rows();
        const size_t nrhs = B.cols();
        BOUNDS_CHECK(n == LU.cols() && n == B.rows() && piv.size() == n);
        // Apply row permutation to B: swap row i with row piv[i] for all columns.
        // For large nrhs the column loop is parallelised; threshold prevents
        // task-spawn overhead for small systems.
        for (size_t i = 0; i < n; ++i) {
            if (piv[i] != i) {
                const size_t pi = piv[i];
                parallel_for(nrhs, PARALLEL_THRESHOLD_SIMPLE,
                    [&B, i, pi](size_t s, size_t e) {
                        for (size_t j = s; j < e; ++j)
                            std::swap(B(i, j), B(pi, j));
                    });
            };
        };
        // Forward substitution: L*X = PB (unit lower triangular)
        trsm('L', 'L', 'N', 'U', T(1), LU, B);
        // Backward substitution: U*X = (above) (non-unit upper triangular)
        trsm('L', 'U', 'N', 'N', T(1), LU, B);
    };

    // Determinant via log-magnitude accumulation
    template<typename T, Layout L>
    T lu_det(const LUResult<T, L>& res) {
        const auto& LU = res.packed;
        const auto& piv = res.piv;
        const size_t n = LU.rows();
        BOUNDS_CHECK(n == LU.cols());
        // Sign of the permutation: (-1)^(number of non-trivial transpositions)
        int swaps = 0;
        for (size_t i = 0; i < piv.size(); ++i) if (piv[i] != i) ++swaps;
        const double perm_sign = (swaps % 2 == 0) ? 1.0 : -1.0;
        if constexpr (detail::is_complex_v<T>) {
            using R = detail::real_type_t<T>;
            double log_abs = 0.0;
            double phase = (perm_sign < 0.0) ? std::numbers::pi : 0.0;
            for (size_t i = 0; i < n; ++i) {
                const T d = static_cast<T>(LU(i, i));
                const double abs_d = std::abs(d);
                if (abs_d == 0.0) return T(0);
                log_abs += std::log(abs_d);
                phase += std::arg(d);
            };
            const double mag = std::exp(log_abs);
            return T(static_cast<R>(mag * std::cos(phase)), static_cast<R>(mag * std::sin(phase)));
        } else {
            double log_abs = 0.0;
            double diag_sign = 1.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = static_cast<double>(LU(i, i));
                if (d == 0.0) return T(0);
                log_abs += std::log(std::abs(d));
                if (d < 0.0) diag_sign = -diag_sign;
            };
            return static_cast<T>(perm_sign * diag_sign * std::exp(log_abs));
        };
    };

    // Inverse: delegates to lu_solve(LU, I)
    template<typename T, Layout L>
    Matrix<T, L> lu_inverse(const LUResult<T, L>& res) {
        BOUNDS_CHECK(res.packed.rows() == res.packed.cols());
        Matrix<T, L> inv = Matrix<T, L>::identity(res.packed.rows());
        lu_solve(res, inv);
        return inv;
    };
};