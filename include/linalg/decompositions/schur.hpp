#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    // Publicly-visible Schur decomposition package.
    template<Layout LL = Layout::RowMajor>
    struct SchurResult {
        Matrix<DefaultScalar, LL> T; // Upper-triangular Schur factor.
        Matrix<DefaultScalar, LL> Q; // Unitary matrix of Schur vectors.
        Vector<DefaultScalar> eigvals;
        Vector<double> balance_scale; // Per-index scaling factors.
        Vector<size_t> balance_perm; // Permutation for balancing.
        bool balanced{false};
    };

    namespace detail {
        struct BalanceInfo {
            Vector<double> scale; // log-base scaling exponent per index.
            Vector<size_t> perm;
            size_t ilo, ihi; // Only [ilo, ihi] is non-trivially redused.
        };

        /// @brief Parlett-Reinsch balancing.
        /// @param A Input matrix.
        /// @return `scale[i] = s_i` such that the balancing transformation is`D = diag(base^{s_0}, ..., base^{s_{n-1}})` and `perm[i]` is source row that was swapped into position `i`.
        /// @note  Applies a similarity `D^{-1}*A*D` where `D = diag(scale)` and a permutation that isolates eigenvalues in the top-left and bottom-right corners.
        template<Layout L>
        BalanceInfo balance(Matrix<DefaultScalar, L>& A) {
            const size_t n = A.rows();
            Vector<double> scale(n, 0.0);
            Vector<size_t> perm(n);
            for (size_t i = 0; i < n; ++i) perm[i] = i;
            // Phase 1: permute rows/columns to isolate decoupled eigenvalues at corners.
            size_t ilo = 0, ihi = n - 1;
            bool changed = true;
            while (changed && ihi > ilo) {
                changed = false;
                for (size_t i = ilo; i <= ihi; ++i) {
                    // Check if row i has all zeros in [ilo, ihi] columns except column i.
                    bool row_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(i, j)) != 0.0) { row_ok = false; break; };
                    };
                    bool col_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(j, i)) != 0.0) { col_ok = false; break; };
                    };

                    if (row_ok && col_ok && i != ihi) {
                        // Swap row/col i with ihi:
                        for (size_t k = 0; k < n; ++k) std::swap(A(i, k), A(ihi, k));
                        for (size_t k = 0; k < n; ++k) std::swap(A(k, i), A(k, ihi));
                        std::swap(perm[i], perm[ihi]);
                        scale[ihi] = 0.0; // No scaling needed for isolated eigenvalue.
                        --ihi;
                        changed = true;
                        break;
                    };
                };
            };

            changed = true;
            while (changed && ilo < ihi) {
                changed = false;
                for (size_t i = ilo; i <= ihi; ++i) {
                    bool row_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(i, j)) != 0.0) { row_ok = false; break; }
                    };
                    bool col_ok = true;
                    for (size_t j = ilo; j <= ihi; ++j) {
                        if (j != i && std::abs(A(j, i)) != 0.0) { col_ok = false; break; }
                    };
                    if (row_ok && col_ok && i != ilo) {
                        for (size_t k = 0; k < n; ++k) std::swap(A(i, k), A(ilo, k));
                        for (size_t k = 0; k < n; ++k) std::swap(A(k, i), A(k, ilo));
                        std::swap(perm[i], perm[ilo]);
                        scale[ilo] = 0.0;
                        ++ilo;
                        changed = true;
                        break;
                    };
                };
            };

            // Phase 2: diagonal scaling on the [ilo, ihi] submatrix.
            static constexpr double BASE = 2.0;
            static constexpr double BETA = 0.95; // Convergence threshold.
            static constexpr int MAXP = 10; // Maximum for repeated passes computing row-norm and col-norm.
            if (ilo < ihi) {
                for (int pass = 0; pass < MAXP; ++pass) {
                    bool any_scaled = false;
                    for (size_t i = ilo; i <= ihi; ++i) {
                        double rn = 0.0, cn = 0.0;
                        for (size_t j = ilo; j <= ihi; ++j) {
                            if (j != i) {
                                rn += std::abs(A(i, j));
                                cn += std::abs(A(j, i));
                            };
                        };
                        if (rn == 0.0 || cn == 0.0) continue;
                        // Compute integer power of BASE so that rn*s ~ cn/s, i.e. s = sqrt(cn/rn), rounded to nearest power of BASE.
                        double s = 1.0;
                        double f = 1.0;
                        double ratio = cn / rn;
                        // Bring ratio into [1/BASE^2, BASE^2] using integer steps.
                        while (ratio < 1.0 / (BASE * BASE)) { ratio *= BASE * BASE; f /= BASE; }
                        while (ratio > BASE * BASE) { ratio /= BASE * BASE; f *= BASE; }
                        // Adjustment: single BASE step.
                        if (ratio < 1.0 / BASE) f /= BASE;
                        else if (ratio > BASE) f *= BASE;
                        if (std::abs(f - 1.0) < 1e-14) continue;
                        if (f == 0.0) continue;
                        any_scaled = true;
                        // Apply similarity:
                        for (size_t j = 0; j < n; ++j) A(i, j) *= f;
                        for (size_t j = 0; j < n; ++j) A(j, i) /= f;
                        scale[i] += std::log2(f);
                    };
                    if (!any_scaled) break;
                };
            };

            return BalanceInfo{ std::move(scale), std::move(perm), ilo, ihi };
        };
    };
};