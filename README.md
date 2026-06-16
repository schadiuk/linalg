# linalg
An educational header-based C++ linear algebra library revolving around lazy expression templates and BLAS-like kernels.
---
## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Core types](#core-types)
- [Expression templates](#expression-infrastructure)
- [Operations](#operationsutility-functions)
- [BLAS](#blas)
- [Matrix decompositions](#decompositions)
- [Benchmarking](#benchmarking)
---
## Overview
`linalg` provides dense vector and matrix operations with zero memory overhead lazy evaluation, GEMM and a custom-built parallel execution infrastructure. The library targets to explore typical numerical methods workflows where control over memory layout, precision and performance meet intuitive syntax with no external dependencies.

---
## Features
- **Header-only**: including a single header file is the only setup needed.
- **Expression templates**: arithmetic expressions (eg. `alpha * A - beta * B * C`) are lazy and allocation-free until assigned.
- **BLAS**: optimised, CPU-friendly kernels for `gemv`, `gemm`, `trsv`, `trsm` and many more operations.
- **Parallelisation**: a singleton `ThreadPool` dispatches work across all hardware threads with thresholds preventing overhead on smaller problems.
- **Aligned storage**: every allocation is 64-byte aligned by means of `AlignedAllocator`, enabling auto-vectorisation.
- **Copy-free views**: `VectorView` and `MatrixView` provide zero-overhead windows, transpositions and conjugations.
- **Complex support**: all operations are designed to support `float`, `double`, and their `std::complex` counterparts.

---
## Documentation

The library features rich documentation: Doxygen-like comment blocks (natively supported by IntelliSense), detailed syntax overview placed in README, and comprehensive reference covering algorithm-heavy aspects. The text documents are available at the dedicated `reference` folder, and are organised as follows:
- [BLAS](/reference/BLAS.md)
- [Cholesky factorisation](/reference/CHOLESKY.md)
- [LU decomposition](/reference/LU.md)
- [QR decomposition](/reference/QR.md)

*Note:* the format chosen is Markdown, supported by GitHub and a number of modern IDEs and editors (including [VS Code](https://code.visualstudio.com/docs/languages/markdown) - the one used in development). For easier understanding of the algorithms, it is recommended not to rely on GitHub website's rendering of formulas (some of which may be parsed incorrectly).


---
## Prerequisites
| Requirement | Minimum |
|-------------|---------|
| C++ standard| C++20. |
| Compiler | GCC 12+, Clang 14+, MSVC 19.30+. |
| Architecture | x86-64, ARM64, or any scalar target. |

No third-party assets needed, as the standard library is the only dependency. For installation, copy the `linalg/` directory and include the umbrella header:
```cpp
#include <linalg/linalg.hpp>
```
Compile with C++20 and enable optimisations for best performance. Below are sample `g++` commands:

- Recommended set of flags (reproducible).
```
g++ -std=c++20 -march=native -O2 my_file.cpp
```
- Performance-oriented build (note that adding `-ffast-math` trades floating-point safety for speed).
```
g++ -std=c++20 -march=native -mtune=native -O3 -funroll-loops -ftree-vectorize my_file.cpp
```
---
## Quick start
```cpp
#include <linalg/linalg.hpp> // Umbrella header.
#include <iostream>

using namespace linalg; // All library classes and functions are enclosed in the namespace.

int main() {
    // Generate sample (trivial) matrix and RHS vector.
    auto A = Matrix<double>::identity(3);
    auto b = arange(1, 4); // [1.0, 2.0, 3.0]

    // Expression templates.
    Vector<double> v = A * b;
    std::cout << "v = " << v << "\n";
    Vector<double> w = 2.0 * b + v;
    std::cout << "w = " << w << "\n";

    // Decomposition.
    auto lu_res = lu(A);
    Vector<double> sol = b;

    lu_solve(lu_res, sol); // In-place solution of A * x = b.

    std::cout << "Solution: " << sol << "\n";
    std::cout << "Residual norm: " << norm(A * sol - b) << "\n";
    return 0;
};
```
---
## Core types
- `Vector<T>` - a contiguous, 64-byte aligned, heap allocated vector type. Participates in expression templates via `VecExpr<Vector<T>>`.
```cpp
// Construction.
Vector<double> v(10); // Unitialised, size 10.
Vector<double> v(10, 1.0); // Size 10 vector,filled with ones.
Vector<double> v = {3.0, 2.0, 1.0}; // Initialised via list.

// Static factories.
auto v = Vector<double>::random(100);
auto v = Vector<double>::zeros(100);
auto v = Vector<double>::ones(100);

// Accessors.
v[i]; v(i); // Unchecked
v.at(i);    // Checked indexation.
v.size();
v.data(); // Raw T* data pointer.
```

- `Matrix<T, L>` - contiguous aligned matrix. `L` denotes layout, defined by `linalg::Layout`: either `RowMajor` (default) or `ColMajor`. Participates in expression templates by means of `MatExpr<Matrix<T, L>>` and other dedicated classes. The view class - `MatrixView<T, L, Trans, Conj, Mutable>` - allows building non-owning matrix windows with compile-time transposition and conjugation flags.
`hermitian()` and `transpose()` are zero-copy: the view's operator() applies the logical transformation on every read.
```cpp
// Construction.
Matrix<double> A(3, 4); // 3 * 4, row-major, uninitialised.
Matrix<double> A(3, 4, 0.0); // Matrix, filled with zeros.
Matrix<double, Layout::ColMajor> B(3, 4); // Column-major layout.

// C-styled arrays and std::array objects are supported as well.
double arr[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
Matrix<double> A(arr);
Matrix<double> B = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};

// Static factory methods.
auto A = Matrix<double>::identity(5);
auto A = Matrix<double>::random(42, 100);
auto A = Matrix<double>::zeros(42, 100);
auto A = Matrix<double>::ones(100, 42);

// Accessors.
A(i, j); // Unchecked
A.at(i, j); // checked indexation.
A.rows();  A.cols();  A.stride();  A.data();

A.reshape(new_rows, new_cols); // Total element count must be unchanged: rows * cols = new_rows * new_cols.

auto tv = transpose(A); // MatrixView<double, RowMajor, true,  false, true>
auto hv = hermitian(A); // MatrixView<double, RowMajor, true,  true,  false>
auto mv = view(A); // MatrixView<double, RowMajor, false, false, true>
```
---
## Expression infrastructure
Expression templates are core elements of the library's logic: arithmetic operators return lightweight expression objects that are evaluated only when assigned to a `Vector` or `Matrix` object. This avoids intermediate allocations in chained operations.
The tables below summarise possible expression uses.
| Operation | Vector expression | Matrix expression | Notes |
| --- | --- | --- | --- |
| Addition | `a + b` | `A + B` | cf. optimised BLAS `axpy` / `axpby` |
| Subtraction | `a - b` | `A - B` |
| Hadamard product | `a * b` | `elementwise_multiply(A, B)` | Default `operator*` for vectors. |
| Elementwise division | `a / b` | `elementwise_divide(A, B)` | Default `operator/` for vectors. |
| Scalar multiplication | `s * a`, `a * s` | `s * A`, `A * s` | cf. optimised `scal` |
| Elementwise scalar reciprocal/division | `s / a`, `a / s`| `s / A`, `A / s` |

Other operations, supported by the expression infrastructure.
| Operation | Expression | Notes |
| --- | --- | --- | 
| Matrix-vector product | `A * v` | cf. optimised `gemv` |
| Vector-matrix product | `v * A` |
| Upper triangle extraction | `triu(A, k)` | k-offset from main diagonal, default is 0. |
| Lower triangle extraction | `tril(A, k)` | k-offset, default is 0. |

*Note:* the lazy `A * B` operator in expression templates does an element-by-element reduction on demand. For large matrices it is recommended to use the optimised [`gemm` BLAS](/reference/BLAS.md#gemm-general-matrix-matrix-product) call instead.
### Wrapping in expressions
Optionally use `expr()` to wrap a storage object so it participates expression algebra:
```cpp
Vector<double> v = expr(A) * expr(b) + expr(c); // Enforces lazy evaluation.
```

---
## Operations/utility functions
The library supports a substantial quantity of common mathematical functions, defined in the [C++ numerics library](https://en.cppreference.com/cpp/numeric). The functions are separated from `std` in `linalg` namespace, just as all other library assets.
| Function class | Present in `linalg` | Notes |
| --- | --- | --- |
| Arithmetic | `abs`, `pow`, `sqrt`, `exp`, `log` | Functions are "inherited" from the standard C++, and applied pointwise. |
| Nearest integer | `floor`, `ceil`, `round` |
| Complex-specific | `real`, `imag`, `conj` | Could be used for real/imag part extraction when assigned. |
| Trigonometric | `sin`, `asin`, `cos`, `acos`, `tan`, `atan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`|
| Reductions | Common: `sum`. Vector-specific: `dot`, `dotc` | cf. optimised BLAS `asum` for vectors. |
| Statistics | `mean`, `variance`, `stddev` |

User-defined utilties can be constructed using `UnaryMatExpr` or `UnaryVecExpr`.
### Construction-specific
Some useful utilities exist to construct structured vectors and matrices. Illustration below:
```cpp
// Uniform spacing.
auto v = linspace<double>(0.0, 1.0, 100); // 100 points in [0., 1.] interval.
auto v = linspace<double>(0.0, 1.0, 100, false); // The points in [0., 1.).

// Ranges.
auto v = arange(10); // [0, 1, 2, ... 9]
auto v = arange(2, 8); // [2, 3, 4, ... 7]
auto v = arange(0.0, 1.0, 0.1) // [0.0, 0.1, 0.2, ... 0.9]

// Diagonal operations.
auto d = diag(A); // Extract main diagonal from A as a vector.
auto D = diag(v); // Build diagonal matrix from a vector.
auto D = diag(diag(A)); // Build diagonal matrix from original A.

// Flatten to vector.
auto v = flatten(expr(A));

// Triangular extraction.
Matrix<T> U = triu(A); // Upper triangle.
Matrix<T> L = tril(A, -1); // Strict lower triangle.
```
### Norms
There are present matrix and vector norms, unified by common dispatch convention via `norm` function. List of them, indexed by `kind` argument, could be found in the table below.
| Norm kind | Vector norm | Matrix norm | Notes |
| --- | --- | --- | --- |
| `0` | Number of non-zero entries. | None | L0 pseudo-norm. |
| `1` | Absolute value sum. | Maximum absolute column sum. | L1 norm. |
| `2` | Square root of the sum of all elements' squares. | None (yet?) | L2 norm (alternatively, `fro` for vectors). |
| `fro` | Same as above. | Square root of the sum of all elements' squares. | Frobenius norm. |
| `inf` | Maximum absolute entry. | Maximum absolute row sum. | Infinity norm. |
| `-inf`* | Smallest absolute entry. | Minimum absolute row sum. | Negative infinity "norm". |

**Note:* the `kind` convention was adopted from `linalg.norm` utility present in `NumPy`, hence existence of `-inf` pseudo-norms (cf. the [documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)). 

---
## BLAS
Another defining feature of the library is the presence of optimised routines, *de facto* constituting a considerable (though by no means exhaustive) subset of the BLAS (Basic Linear Algebra Subprograms) classic specification. 

The existing routines follow standard 3-level convention:
| Level | Meaning | Present in `linalg` | Notes |
| --- | --- | --- | --- |
| BLAS-1 | Vector operations | `axpy`, `axpby`, `scal`, `copy`, `swap`, `iamax`, `iamin`, `asum`, `dot`, `dotc`, `nrm2`, `rotg`, `rot` | There exist matrix overloads for `scal`, `copy`. |
| BLAS-2 | Matrix-vector operations | `gemv`, `ger`, `gerc`, `trsv`, `trmv`, `symv`, `hemv` | `trsm` is inherently serial. |
| BLAS-3 | Matrix-matrix operations | `gemm`, `trsm`, `syrk`, `herk` |

*Note:* for in-depth coverage of BLAS cf. the [dedicated reference](reference/BLAS.md).

---
## Decompositions
The library provides common matrix factorisations, naturally integrated with the expression template and BLAS machinery.

---
### Cholesky factorisation

Blocked **PO**sitive definite **TR**iangular **F**actorisation of an HPD (Hermitian Positive Definite) matrix. For the given matrix $A$ finds lower triangular $L$ such that: $A = L L^H$ (equivalently, upper triangular: $A = U^H U$).

`potrf()` master function returns the triangular `factor` matrix alongside with its tag (`L` or `U`).
```cpp
auto res = potrf(A);

Vector<double> b;
potrs(res, b); // Solve A * x = b  (single RHS, in-place).

Matrix<double> B;
potrs(res, B); // Solve A * X = B  (multiple RHS, in-place).

double ld = cholesky_logdet(res); // Compute logarithm of the determinant.

auto Ainv = potri(res); // Matrix inverse.
```
*Note:* an in-depth look into the algorithm is provided in dedicated [reference file](/reference/CHOLESKY.md).

---
### LU decomposition
Blocked LU with partial pivoting. For original matrix $A$ finds $P$, $L$, $U$ such that: $PA = LU$. *Note:* for detailed account of the algorithm cf. the [reference](reference/LU.md).

The `lu()` function returns `P`, `L`, `U`, the packed representation, and the pivot vector:
```cpp
auto res = lu(A); // LUResult<T, L> - dedicated structure.
 
res.P // Permutation matrix.
res.L // Unit lower triangular factor.
res.U // Upper triangular factor.
res.packed // In-place storage (strict-L below diagonal, U on/above).
res.piv // Pivot indices.
 
// Connected functions:
Vector<double> b;
lu_solve(res, b); // Solve A * x = b  (single RHS, in-place).

Matrix<double> B;
lu_solve(res, B); // Solve A * X = B  (multiple RHS, in-place).

double d = lu_det(res); // Determinant.

auto Ainv = lu_inverse(res); // Inverse.
```

---
### QR factorisation
Householder QR with optional column pivoting. Uses a blocked compact-WY update for large matrices. Finds matrices satisfying: $AP = QR$ (note the column permutation - opposite convention from LU).

`qr()` master function accepts a range of `QRMode` values:
| Value | Q shape | Use case |
|---|---|---|
| `QRMode::Reduced` | m * min(m,n) | Default; economy decomposition. |
| `QRMode::Complete` | m * m | Full orthonormal basis. |
| `QRMode::R` |  | R only; fastest. |
```cpp
// Modes:
auto res = qr(A); // Reduced QR (default).
auto res = qr_complete(A); // Full Q.
auto res = qr_r(A); // R only - Q not formed.
auto res = qr_pivoted(A); // Column-pivoted.
auto res = qr_pivoted(A, tol); // With explicit rank tolerance set.
 
// Fine-grained control, returns QRResult<T, L>.
auto res = qr(A, QRMode::Reduced, /*pivoting=*/true, /*tol=*/-1.0);
 
res.Q // Orthonormal factor (m * min(m,n) for Reduced).
res.R // Upper triangular factor.
res.P // Permutation matrix (for pivoted QR: A * P = Q * R).
res.piv // Pivot indices (for pivoted QR).
res.rank  // Estimated numerical rank (pivoted only; -1 otherwise).
res.pivoted // true if pivoting was enabled.
```
*Note:* an in-depth discussion of the algorithms is provided at the [dedicated reference](/reference/QR.md).

---
## Benchmarking

The library was benchmarked using a console-based suite, with results provided in a dedicated [text file](/benchmark/benchmark.txt).

> **Hardware:** 12-thread machine with 64-byte cache line.

> **Build.** Aggressive optimisation, namely:
 `g++ -std=c++20 -march=native -mtune=native -O3 -ffast-math -funroll-loops -ftree-vectorize -Iinclude benchmark/benchmark.cpp -o main`.

---
### Global summary
```
  Peak memory BW (axpy large N)    :    54.30 GB/s
  Peak GEMM double (N>=2048)       :    54.07 GFLOP/s
  Peak GEMM float                  :    69.12 GFLOP/s
  Ridge point (double)             :    1.00 FLOP/byte
```

Every operation performed on `double` with arithmetic intensity below 1.0 FLOP/byte turns out to be memory-bandwidth bound. Per-section peak table:

| Section | Peak GFLOP/s | Peak BW, GB/s | Notes |
| --- | --- | --- | --- |
| BLAS-1 | 68.84 (`axpy`) | 445.82 (`scal`) | Limited by L1 bandwidth. |
| BLAS-2 | 30.97 | 126.79 | `gemv` best performance on L2-compatible workloads. |
| BLAS-3 | 70.81 (square), 81.30 (wide-short) `gemm` | | Wide-short cache hit. |
| Decompositions | 69.53 (`lu`) | | `gemm`-dominated workflow. |
| Norms | | 159.70 | L2 streaming for matrix `norm_inf`. |

---
### Layout comparison

A series of experiments was run to establish layout-unbiasedness of `gemm` kernels:

| Problem size | `RowMajor` (GFLOP/s) | `ColMajor` (GFLOP/s) | Difference |
| --- | --- | --- | --- |
| 128 | 28.44 | 25.44 | 3.0 |
| 256 | 51.99 | 50.90 | 1.09 |
| 512 | 63.16 | 62.54 | 0.62 | 
| 1024 | 64.77 | 64.32 | 0.45 |
| 2048 | 50.00 | 49.57 | 0.43 |

Even though `RowMajor` mode is set as the default storage, it is apparent that the two layouts are statistically indisitinguishable for $N > 256$. The blocked microkernel (`gemm_microkernel_row` / `gemm_microkernel_col`) is explicitly specialised for each layout, and both apply the same 8-wide unrolled inner loop. The layout abstraction layer adds zero measurable overhead.

---