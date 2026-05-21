# linalg
An educational header-based C++ linear algebra library revolving around lazy expression templates and BLAS-like kernels.
---
## Overview
`linalg` provides dense vector and matrix operations with zero memory overhead lazy evaluation, hand-tuned GEMM and 
a custom-built parallel execution infrastructure. The library targets to explore typical numerical methods workflows
where control over memory layout, precision and performance meet intuitive syntax with no external dependencies.
