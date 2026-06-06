# LU decomposition: reference
This document covers the implementation of LU algorithm present in, and used by the `linalg` library.
> **Source file:** `lu.hpp`.

> **Dependencies:** `level3.hpp` (which pulls in `level1.hpp`, `level2.hpp`).

---
## Table of contents:
- [Preamble](#0-preamble-and-notation)
- [Mathematical foundation](#1-mathematical-foundation)
---
## 0. Preamble and notation
Matrices are $m \times n$ with entries in $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$. Elements are denoted either as $A_{ij}$ or as $A(i,j)$ using **0-based indexing throughout**, matching standard C++ array conventions.
 
The following norms and quantities appear in the analysis below.
 
| Symbol | Meaning |
|---|---|
| $\|A\|_F$ | Frobenius norm $\sqrt{\sum_{ij} \|A_{ij}\|^2}$ |
| $\|A\|_2$ | Spectral norm (largest singular value) |
| $\|A\|_\infty$ | Maximum absolute row sum |
| $\kappa(A)$ | Condition number $\|A\|_2 \|A^{-1}\|_2$ |
| $k$ | $k = \min(m, n)$ throughout |
| $\varepsilon$ | Machine epsilon ($\approx 2.2 \times 10^{-16}$ for `double`) |

**Packed vs. unpacked representation.** After factorisation, `lu_factor` stores both factors in the original matrix footprint - the *packed* matrix. The strict lower triangle (below the main diagonal) holds the multipliers of $L$; the upper triangle (including the diagonal) holds $U$. The unit diagonal of $L$ is not stored. The *unpacked* representation splits these into separate `L` and `U` matrices and is produced on demand by the `unpack_L` / `unpack_U` helpers. Solvers always operate on the packed form; the unpacked form is provided for display and verification.
 
**Pivot vector convention.** The library uses a *sequential transposition* encoding: `piv[j]` stores the row index that was physically swapped with row $j$ at step $j$ of the unblocked panel factorisation. This is distinct from a *destination permutation*, where `piv[j]` would name the final resting place of original row $j$. The sequential encoding is the LAPACK convention and is what `lu_solve` applies directly (left-to-right swap replay).

---
