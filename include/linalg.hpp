#pragma once

#include <linalg/core/common.hpp>
#include <linalg/core/error.hpp>
#include <linalg/core/parallel.hpp>

#include <linalg/storage/vector.hpp>
#include <linalg/storage/matrix.hpp>
#include <linalg/storage/vector_view.hpp>
#include <linalg/storage/matrix_view.hpp>

#include <linalg/expressions/expr_base.hpp>
#include <linalg/expressions/vector_expr.hpp>
#include <linalg/expressions/matrix_expr.hpp>

#include <linalg/operations/vector_ops.hpp>
#include <linalg/operations/matrix_ops.hpp>
#include <linalg/operations/constructors.hpp>

#include <linalg/blas/level1.hpp>
#include <linalg/blas/level2.hpp>
#include <linalg/blas/level3.hpp>

#include <linalg/norms/vector_norms.hpp>
#include <linalg/norms/matrix_norms.hpp>

#include <linalg/decompositions/lu.hpp>
#include <linalg/decompositions/qr.hpp>
#include <linalg/decompositions/cholesky.hpp>
#include <linalg/decompositions/schur.hpp>
#include <linalg/decompositions/bidiag.hpp>
#include <linalg/decompositions/svd.hpp>

#include <linalg/solvers/lstsq.hpp>
#include <linalg/solvers/eig.hpp>

#include <linalg/io.hpp>