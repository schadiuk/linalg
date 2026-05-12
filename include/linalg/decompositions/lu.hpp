#pragma once

#include <linalg/blas/level3.hpp>

namespace linalg {
    // LUResult Bundles the publicly-visible decomposition (P, L, U) together with the internal packed representation used by the solvers
    template<typename T, Layout LL>
    struct LUResult {
        Matrix<T, LL> P; // Permutation matrix: P * A = L * U
        Matrix<T, LL> L; // m*min(m,n) unit lower triangular factor
        Matrix<T, LL> U; // min(m,n)*n upper triangular factor
        Matrix<T, LL> packed; // In-place storage: strict lower L, upper U
        Vector<size_t> piv; // piv[j] = row swapped with row j at step j
    };
};