#pragma once

#include <linalg/decompositions/qr.hpp>

namespace linalg {
    template<typename T>
    struct LstsqVecResult {
        Vector<T> x; // Solution vector (length n).
        double residual;
        int rank; // Numerical rank of RHS matrix.
    };

    template<typename T, Layout L>
    struct LstsqMatResult {
        Matrix<T, L> X; // Solution matrix (n * nrhs).
        Vector<double> residuals;
        int rank; // Numerical rank of RHS matrix.
    };
};