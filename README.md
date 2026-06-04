# linalg
An educational header-based C++ linear algebra library revolving around lazy expression templates and BLAS-like kernels.
---
## Overview
`linalg` provides dense vector and matrix operations with zero memory overhead lazy evaluation, hand-tuned GEMM and a custom-built parallel execution infrastructure. The library targets to explore typical numerical methods workflows where control over memory layout, precision and performance meet intuitive syntax with no external dependencies.
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
## Prerequisites
| Requirement | Minimum |
|-------------|---------|
| C++ standard| C++20|
| Compiler | GCC 12+, Clang 14+, MSVC 19.30+ |
| Architecture | x86-64, ARM64, or any scalar target |

No third-party assets needed, as the standard library is the only dependency. For installation, copy the `linalg/` directory and include the umbrella header:
```cpp
#include <linalg.hpp>
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
| Elementise scalar reciprocal/division | `s / a`, `a / s`| `s / A`, `A / s` |

Other operations, supported by the expression infrastructure.
| Operation | Expression | Notes |
| --- | --- | --- | 
| Matrix-vector product | `A * v` | cf. optimised `gemv` |
| Vector-matrix product | `v * A` |
| Upper triangle etraction | `triu(A, k)` | k-offset from main diagonal, default is 0. |
| Lower triangle etraction | `tril(A, k)` | k-offset, default is 0. |

*Note:* the lazy `A * B` operator in expression templates does an element-by-element reduction on demand. For large matrices it is recommended to use the optimised `gemm` BLAS call instead.
### Wrapping in expressions
Optionally use `expr()` to wrap a storage object so it participates expression algebra:
```cpp
Vector<double> v = expr(A) * expr(b) + expr(c); // Enforces lazy evaluation.
```

---
## Operations/utility functions
The library supports a considerable subset of common mathematical functions, defined in the C++ numerics library. The functions are separated from `std` in `linalg` namespace, just as all other library assets.
| Function class | Present in `linalg` | Notes |
| --- | --- | --- |
| Arithmetic | `abs`, `pow`, `sqrt`, `exp`, `log` | Functions are "inherited" from the standard C++, and applied pointwise. |
| Nearest integer | `floor`, `ceil`, `round` |
| Complex-specific | `real`, `imag`, `conj` | Could be used for real/imag part extraction. |
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

**Note:* the `kind` convention was adopted from `linalg.norm` utility present in `NumPy`, hence existence of `-inf` pseudo-norms.

---
## BLAS
Another defining feature of the library is the presence of optimised routines, *de facto* constituting a considerable (though by no means exhaustive) subset of the BLAS (Basic Linear Algebra Subprograms) classic specification. 

The existing routines follow standard 3-level convention:
| Level | Meaning | Present in `linalg` | Notes |
| --- | --- | --- | --- |
| BLAS-1 | Vector operations | `axpy`, `axpby`, `scal`, `copy`, `swap`, `iamax`, `iamin`, `asum`, `dot`, `dotc`, `nrm2`, `rotg`, `rot` | There exist matrix overloads for `scal`, `copy`. |
| BLAS-2 | Matrix-vector operations | `gemv`, `ger`, `gerc`, `trsv`, `trmv`, `symv`, `hemv` | `trsm` is inherently serial. |
| BLAS-3 | Matrix-matrix operations | `gemm`, `trsm`, `syrk`, `herk` |

*Note:* for in-depth coverage of BLAS cf. the dedicated reference.


---
## Quick start
```cpp
#include <linalg.hpp> // Umbrella header.
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
