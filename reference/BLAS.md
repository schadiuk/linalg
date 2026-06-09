# BLAS Reference
This document explores every BLAS-level operation in the `linalg` library: the mathematical definition of each routine, a full account of the internal algorithm and engineering decisions behind it.
> **Source files:** `level1.hpp`, `level2.hpp`, `level3.hpp`.

> **Dependencies:** `matrix.hpp`, `vector.hpp`.

---
## Table of contents
- [Parallelisation model](#parallelisation-model)
- [Conventions](#conventions)
- [BLAS-1](#level-1-vector-operations)
- [BLAS-2](#level-2-matrix-vector-operations)
- [BLAS-3](#level-3-matrix-matrix-oprations)
- [Quick reference](#quick-reference)
---
## Parallelisation model
As `linalg` actively uses threading capabilities of C++, understanding the library's parallelisation infrastructure is a prerequisite for all further reasoning about the behaviour and design of kernels.

---
### The thread pool

The library manages a `ThreadPool` singleton, created on first use via `ThreadPool::instance()`. Its constructor calls `std::thread::hardware_concurrency()` and spawns that many persistent worker threads, falling back to 4 if the query returns zero. Threads are created once at program start and live for the duration of the process: there is no per-call thread creation overhead.
 
Each worker runs an event loop that blocks on a `std::condition_variable` when idle:

```cpp
while (true) {
    std::function<void()> task;
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] {
            return stop_ || !tasks_.empty();
        });
        if (stop_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop();
    }
    task();
};
```
Work is submitted by `pool.enqueue(lambda)`, which wraps the lambda in a `std::packaged_task`, pushes it to the queue, notifies one worker via `condition_.notify_one()`, and returns a `std::future`. The caller, in turn, collects all futures with `.get()` to form a join barrier. This constitutes a standard fork-join pattern.

### Partitioned range execution via `parallel_for`

```cpp
parallel_for(total, threshold, func); // func receives (start, end) for its assigned range.
```
 
**Serial bypass.** When `total < threshold`, the lambda is called immediately on the calling thread with `func(0, total)`. There is no enqueue, no lock acquisition, no condition-variable signal: the overhead is a single branch.
 
**Thread count selection.** When above the threshold, the number of threads spawned is:
 
$$\text{num\_threads} = \min\bigl(\text{pool.thread\_count()},\;\lceil\text{total}/\text{threshold}\rceil\bigr)$$
 
This cap ensures every thread always receives at least `threshold` elements of work, preventing the task-spawn overhead from dominating for barely-above-threshold inputs.

**Static contiguous partitioning.** The range $[0, \text{total})$ is split into `num_threads` contiguous chunks. The base chunk size is $\lfloor\text{total}/T\rfloor$; the first $\text{total} \bmod T$ threads each receive one extra element. Contiguous partitioning maximises spatial locality: each thread's elements are adjacent in memory, filling a compact set of cache lines with minimal cross-thread cache-line invalidation.
 
**Synchronisation.** After all enqueues, the caller iterates `for (auto& f : futures) f.get()`, which both joins all threads and propagates any exception thrown by a worker back to the calling thread.

### Reduction patterns
```cpp
T result = parallel_reduce<T>(total, threshold, [](size_t i) -> T { return f(i); }); // Per-element interface.
 
T result = parallel_reduce_chunks<T>(total, threshold, [](size_t s, size_t e) -> T { return g(s, e); }); // Chunk interface.
```

A reduction requires threads to combine results into a single scalar. Accumulation into a shared variable requires atomic operations which inhibit vectorisation and generate cache-coherence traffic. The library uses a **private partial-sum pattern** instead:
1. Allocate `std::vector<Padded<T>> partials(num_threads)`.
2. Each thread accumulates exclusively into `partials[t].value`.
3. After the join barrier, the driver sums all partials serially.
`Padded<T>` is declared with `alignas(CACHE_LINE_SIZE)` (typically 64 bytes), ensuring consecutive entries land on separate cache lines. Without this padding, a write by thread $t$ to `partials[t]` would invalidate the cache line holding `partials[t+1]` in thread $t+1$'s cache - false sharing - causing cache-coherence traffic that eliminates most of the parallelism benefit.
 
The `parallel_reduce_chunks` variant passes a `(start, end)` range to the user function rather than a single index. This enables the chunk body to use its own vectorised or unrolled inner loop (for example `dot_chunk` or `nrm2_ssq_chunk`) rather than being forced into a single-element-per-call interface. Partial-sum logic and false-sharing prevention are identical between the two variants.

### Threshold constants
The thresholds are shared and used in different contexts across the library. They, alongside with cache constants, are defined at `linalg/include/core/parallel.hpp` file.
| Constant | Value | Governing criterion |
|---|---|---|
| `PARALLEL_THRESHOLD_SIMPLE` | 65536 | Used in memory-bound operationss. Above this size data exceeds typical L2 cache and multiple threads genuinely improve throughput. |
| `PARALLEL_THRESHOLD_COMPUTE` | 4096 | Compute-heavy kernels (`gemv` rows, `ger` columns, `trmv` rows, `symv`/`hemv` columns). Arithmetic intensity is high enough that threading pays off at a smaller element count. |
| `PARALLEL_THRESHOLD_REDUCE` | 16384 | Reduction passes (`dot`, `nrm2`, `asum`) of [BLAS-1](#level-1-vector-operations). Per-element work is light but the sequential accumulation bottleneck justifies earlier threading. |
 
### Vectorisation annotations
| Macro | GCC | Clang | MSVC |
|---|---|---|---|
| `LINALG_VECTORIZE` | `#pragma GCC ivdep` | `#pragma clang loop vectorize(enable) interleave(enable)` | `__pragma(loop(ivdep))` |
| `LINALG_UNROLL(N)` | `#pragma GCC unroll N` | `#pragma clang loop unroll_count(N)` | None |
| `LINALG_RESTRICT` | `__restrict__` | `__restrict__` | `__restrict` |
 
`LINALG_VECTORIZE` asserts to the compiler that loop iterations carry no dependencies, allowing SIMD widening without scalar fallback paths. `LINALG_RESTRICT` on pointer arguments asserts non-aliasing, removing aliasing guards from vector load/store pairs.

`LINALG_INLINE` marks functions as `[[gnu::always_inline]] inline` on GCC and Clang, preventing the compiler from outlining hot inner loops. `assume_aligned<64>(ptr)` communicates 64-byte pointer alignment - guaranteed by `AlignedAllocator<T, 64>` for all library-managed storage - enabling unmasked aligned SIMD loads without a scalar preamble.

---
## Conventions
- Character parameters follow standard BLAS naming. Both upper- and lower-case are accepted for all character parameters. 
 
| Parameter | Values | Meaning |
|---|---|---|
| `uplo` | `'U'` / `'L'` | Upper or lower triangle of A. |
| `trans` | `'N'` / `'T'` / `'C'` | No transpose / transpose / conjugate transpose. |
| `diag` | `'N'` / `'U'` | Non-unit diagonal / unit diagonal (diagonal entries treated as exactly 1, never read). |
| `side` | `'L'` / `'R'` | Triangular factor multiplies from the left or right. |
 
- Template parameters: `T` is the scalar type (`float`, `double`, `std::complex<float>`, `std::complex<double>`); `L` is `Layout::RowMajor` (default) or `Layout::ColMajor`.

---
## Level 1: vector operations
Level 1 routines perform $O(N)$ work on one or two vectors. At this level computation is almost entirely memory-bandwidth-bound. The central engineering concerns are SIMD utilisation, avoiding unnecessary memory traffic, and controlling floating-point rounding error in reductions.
- [`axpy`](#axpy-scaled-vector-addition)
- [`axpby`](#axpby-general-two-scalar-update)
- [`scal`](#scal-in-place-scaling)
- [`dot`](#dot-real-dot-product)
- [`dotc`](#dotc-conjugated-dot-product)
- [Euclidean norm: `nrm2`](#nrm2-euclidean-norm)
- [`asum`](#asum-absolute-value-sum)
- [Extremal elements: `iamax` / `iamin`](#iamax--iamin-index-of-extremal-element)
- [`copy` / `swap`](#copy-and-swap)
- [Givens rotations: `rotg` and `rot`](#givens-rotations)

---
### Infrastructure
Every routine begins by an attempt to recover a raw `const T*` from its input expression via `detail::dense_data<T>()`. This function inspects the static type at compile time:
```cpp
if constexpr (std::is_same_v<E, Vector<T>>) return e.data();
else if constexpr (std::is_same_v<E, VecRef<T>>) return e.vec.data();
else if constexpr (std::is_same_v<E, VectorView<T,false>> || std::is_same_v<E, VectorView<T,true>>)
    return (e.stride() == 1) ? e.data() : nullptr;
else if constexpr (std::is_same_v<E, VecViewRef<T,false>> || std::is_same_v<E, VecViewRef<T,true>>)
    return (e.view.stride() == 1) ? e.view.data() : nullptr;
else return nullptr;
```

A strided `VectorView` returns `nullptr` even though it holds a data pointer: its elements are not contiguous, so a SIMD load of consecutive bytes would read interleaved garbage. An arbitrary expression tree always returns `nullptr` as there is no flat backing array. When a raw pointer is obtained, it is passed through `assume_aligned<64>()` and declared `LINALG_RESTRICT`, giving the compiler full permission to emit aligned SIMD loads and stores without aliasing guards. On failure, the kernel falls back to per-element `operator()` calls.

---
### `axpy`: scaled vector addition
```cpp
axpy(alpha, expr(x), y);
axpy(alpha, expr(x), y_view);  // Mutable VectorView output is supported as well.
```
 
**Operation:** $y_i \leftarrow y_i + \alpha x_i$ for all $i$.

The pragma combined with RESTRICT declarations gives the compiler permission to emit FMA instructions at the full SIMD width. For `VectorView<T, true>` output, the code forks on `y.stride()`: unit-stride views use the same vectorised path; strided views fall back to `y(i) += ...` through `operator()`. Parallelised with `parallel_for` at `PARALLEL_THRESHOLD_SIMPLE`.

---
### `axpby`: general two-scalar update
```cpp
axpby(alpha, expr(x), beta, y);
```

**Operation:** $y_i \leftarrow \alpha x_i + \beta y_i$.
 
**Motivation.** A separate `scal(beta, y)` followed by `axpy(alpha, x, y)` reads and writes `y` twice, consuming a total of $3N \cdot \text{sizeof}(T)$ bytes of bandwidth. `axpby` fuses both scalings into one pass:
 
```cpp
LINALG_VECTORIZE
for (size_t i = s; i < e; ++i) y[i] = alpha * x[i] + beta * y[i];
```
 
On hardware with FMA (Fused Multiply-Add) support this is two multiply-adds per element in a single loop body - the same memory bandwidth as `axpy` but more arithmetic per byte fetched. Pointer extraction and parallelisation logic are identical to those of `axpy`.

---
### `scal`: in-place scaling
```cpp
scal(alpha, x); // Vector<T>.
scal(alpha, x_view); // VectorView<T, true>.
scal(alpha, A); // Matrix<T, L> - flat array treated as 1-D.
scal(alpha, A_view); // Mutable MatrixView.
```
 
**Operation:** $x_i \leftarrow \alpha x_i$, where $x_i$ is `std::vector` entry, which enables overloads for matrix types as well.

The chunk kernel carries two pragmas:
```cpp
LINALG_UNROLL(4)
LINALG_VECTORIZE
for (size_t i = s; i < e; ++i) x[i] *= alpha;
```
 
`LINALG_UNROLL(4)` requests 4-wide software unrolling, exposing four independent multiply chains to the out-of-order scheduler. On a core that can retire two FP multiplies per cycle, a single chain saturates only one execution port; four independent chains can saturate both. `LINALG_VECTORIZE` then widens each chain to the full SIMD width. For `Matrix<T, L>`, the entire flat backing array of $M \times N$ elements is treated as a single 1-D span, enabling a single `parallel_for` over $M \times N$ contiguous elements.

---
### `dot`: real dot product
```cpp
double d = dot(expr(x), expr(y));
```

**Operation:** $\sum_i x_i y_i$.
 
**Algorithm:** 4-unrolled kernel. The chunk kernel maintains four independent accumulators:
```cpp
T s0{}, s1{}, s2{}, s3{};
LINALG_VECTORIZE
for (size_t i = s; i < n4; i += 4) {
        s0 += x[i] * y[i];
        s1 += x[i+1] * y[i+1];
        s2 += x[i+2] * y[i+2];
        s3 += x[i+3] * y[i+3];
    };
T acc = (s0 + s1) + (s2 + s3); // Pairwise sum reduces rounding error.
```

Two distinct improvements over a naive single-accumulator loop are achieved. First, a naive single-chain `acc += x[i]*y[i]` has a loop-carried dependency through `acc` at every step; with FMA latency of 4 cycles, each iteration stalls 4 cycles waiting for the previous result, giving effective throughput of one FMA per 4 cycles regardless of peak hardware capability. Four independent accumulators break this dependency, enabling the out-of-order scheduler to overlap all four chains. Second, the final pairwise combine $(s_0 + s_1) + (s_2 + s_3)$ is a two-level tree, reducing floating-point rounding error from $O(N\varepsilon)$ for sequential summation to $O(\log_2(4) \cdot \varepsilon) = O(2\varepsilon)$ for the combine step. `parallel_reduce_chunks` is used rather than `parallel_reduce` so the unrolled vectorised kernel runs inside the lambda with no per-element function-call overhead.
 
---
### `dotc`: conjugated dot product
```cpp
auto dc = dotc(expr(x), expr(y));
```
 
**Operation:** $\sum_i \overline{x_i} \, y_i$, equivalent to the BLAS `zdotc` [convention](https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot_ga2d59b29ec40fb1bedeb3f10205155ee6.html). For real types the operation is identical to `dot`.
 
The chunk kernel mirrors `dot_chunk` exactly but substitutes $\overline{x_i}$ in each multiply. On hardware, $\overline{z} \cdot w$ is a complex multiply with the imaginary part of $z$ negated, which has the same instruction count as an unconjugated multiply. There is no runtime cost relative to `dot`.

---
### `nrm2`: Euclidean norm
```cpp
double n = nrm2(expr(x)); // Computes vector norm.
```

**Operation:** $\|x\|_2 = \sqrt{\sum_i |x_i|^2}$.
 
**Why not just square and sum?** Two failure modes exist: (a) *overflow*: for $x_i = 10^{200}$ in double, squaring gives $10^{400} = +\infty$ before the sum; (b) *underflow*: for $x_i = 10^{-200}$, squaring gives $10^{-400} = 0$, losing contributions from non-zero elements.
 
**Scaled two-pass algorithm.** Pass 1 (sequential max scan) finds:
 
$$s = \max_i |x_i|$$
 
Pass 2 (parallel scaled squared-sum, via `parallel_reduce_chunks`) computes:
 
$$\text{ssq} = \sum_i \left(\frac{|x_i|}{s}\right)^2$$

The result is $s\sqrt{\text{ssq}}$, which reconstructs $\|x\|_2$ exactly. Dividing by $s$ before squaring guarantees every squared term lies in $[0, 1]$ interval, eliminating both overflow and underflow. The pass-2 chunk kernel uses the same 4-accumulator structure as `dot_chunk` for the same ILP (Instruction-Level Parallelism) and accuracy reasons. All squared sums are computed in `double` regardless of `T`, so `nrm2` of a `float` vector accumulates with double-precision accuracy.

The max scan is sequential because extracting a maximum in parallel would require a second synchronisation step, and the scan is entirely bandwidth-bound - additional threads provide negligible benefit.

---
### `asum`: absolute value sum
```cpp
double s = asum(expr(x));
```

**Operation:**  $\sum_i \bigl(|Re(x_i)| + |Im(x_i)|\bigr)$. That is $\sum_i |x_i|$ for real types.

The 4-accumulator chunk kernel handles both real and complex cases via `if constexpr (is_complex_v<T>)`, which is resolved entirely at compile time so no branch appears in the compiled output. All accumulation is in `double` for the same precision reasons as `nrm2`. Dispatched via `parallel_reduce_chunks`.

---
### `iamax` / `iamin`: index of extremal element
```cpp
size_t idx = iamax(expr(x)); // Index of largest |x_i|.
size_t idx = iamin(expr(x)); // Index of smallest |x_i|.
```

Sequential linear scan with no parallelism. Two reasons justify this: (a) the operation is bandwidth-limited with minimal arithmetic work, so parallel speedup is marginal; (b) BLAS specifies that `iamax` returns the *first* index of the maximum in the event of ties. A parallel max-index scan cannot guarantee this without a second pass.

---
### `copy` and `swap`
```cpp
copy(expr(x), y); copy(expr(A), B);
swap(x, y);
```

`copy` delegates to the assignment operator of the destination type, which already contains aliasing detection and parallel fill. `swap` uses `parallel_for` with `std::swap(x[i], y[i])` in the body; the three-register swap requires no extra allocation.

---
### Givens rotations
- `rotg` (real)
```cpp
double a = 3.0, b = 4.0, c, s;
rotg(a, b, c, s);
// Post-condition: a == 5., b == 0., c == 0.6, s == 0.8.
```

**Operation.** Find $c = \cos\theta$, $s = \sin\theta$ such that:
 
$$\begin{bmatrix} c & s \\ -s & c \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} r \\ 0 \end{bmatrix}$$
 
**Algorithm.** Compute $r = hypot(a, b)$, which is overflow-safe by specification. Then $c = a/r$, $s = b/r$. The sign of $r$ is assigned to match whichever of $|a|$, $|b|$ is larger:
 
```cpp
if (std::abs(a) >= std::abs(b)) r = std::copysign(r, a);
else r = std::copysign(r, b);
```

This minimises cancellation in the subsequent divisions. If $|a|$ dominates and $r = sign(a) \cdot hypot$, then $c = a/r \approx 1$ (no cancellation) and $s = b/r$ is small.

- `rotg` (complex)
For complex types, the rotation is unitary with $c$ real and $s$ complex, satisfying $c^2 + |s|^2 = 1$. The algorithm is as follows:
 
1. If $|a| = 0$: set $c = 0$, $s = 1$, $a = b$, $b = 0$.
2. Extract $\phi = a / |a|$ (the unit-magnitude phase of $a$).
3. Scale to avoid overflow: let $\sigma = |a| + |b|$, then compute $r = \sigma\sqrt{|a/\sigma|^2 + |b/\sigma|^2}$.
4. Set $c = |a|/r$, $\;s = \phi \cdot \overline{b}/r$, $\;a = \phi \cdot r$, $\;b = 0$.

The phase extraction ensures the output $a = \phi \cdot r$ is aligned so that the rotation maps $a$ to the positive real magnitude $r$.

- `rot`: apply the rotation
```cpp
rot(x, y, c, s);
// Real: x_i = c·x_i + s·y_i; y_i = −s·x_i + c·y_i
```

The inner loop is vectorised:
```cpp
LINALG_VECTORIZE
for (size_t i = rs; i < re; ++i) {
    T xi = xp[i], yi = yp[i];
    xp[i] = c * xi + s * yi;
    yp[i] = -s * xi + c * yi;
};
```

Two reads and two writes per element, two FMA pairs per iteration. The complex overload replaces $-s$ with $-\overline{s}$ in the $y$ update. $\overline{s}$ is computed once outside the loop (`cs = conj(s)` earlier in the implementation) and captured by the lambda, keeping the hot loop free of unnecessary complex arithmetic.

---
## Level 2: matrix-vector operations

Level 2 routines perform $O(MN)$ arithmetic with $O(MN)$ memory traffic - arithmetic intensity close to 1 op/byte for double. They are bandwidth-bound for large matrices. The central engineering concerns are cache-line-friendly access order, selecting the correct parallelism axis for the storage layout, and avoiding extra passes over large working sets.
- [`gemv`](#gemv-general-matrix-vector-multiply)
- [`trsv`](#trsv-single-rhs-triangular-solve)
- [`trmv`](#trmv-triangular-matrix-vector-product)
- [`ger` / `gerc`](#ger-and-gerc-rank-1-updates)
- [`symv` / `hemv`](#symv-and-hemv-symmetrichermitian-matrix-vector-products)

---
### Infrastructure

Level 2 introduces a unified helper `detail::resolve_vec<T>(expr, tmp)`, which either recovers a raw pointer plus stride via `raw_vec_ptr`, or materialises the expression into `tmp` and returns `{tmp.data(), 1}`. This replaces *ad hoc* materialisation scattered across callers, ensuring all Level 2 routines handle strided views and arbitrary expressions consistently.

---
### `gemv`: general matrix-vector multiply
```cpp
// y = alpha * A * x + beta * y, A is M * N matrix.
gemv(alpha, expr(A), expr(x), beta, y);
gemv<double, Layout::ColMajor>(alpha, expr(A), expr(x), beta, y);
```
 
**Operation:** $y \leftarrow \alpha A x + \beta y$.

**Dispatch:** `gemv_impl` operates in two stages. First, `resolve_vec<T>` obtains `(x_ptr, incx)`; a non-unit stride is handled explicitly in the kernel, avoiding materialisation for the common case of column-views. Second, `raw_mat_info<T>` attempts to extract `(data*, lda, Layout)` from the matrix expression; on failure the expression is materialised into a temporary `Matrix<T, L>`. Kernel dispatch follows from `a_info->layout`.

- Row-major kernel
```cpp
parallel_for(M, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
    for (size_t i = rs; i < re; ++i) {
            const T* LINALG_RESTRICT a_row = A + i * lda;
            T acc = T(0);
            if (incx == 1) { // Unit-stride: vectorisable dot product.
            LINALG_VECTORIZE
            for (size_t j = 0; j < N; ++j) acc += a_row[j] * x[j];
        } else {
            for (size_t j = 0; j < N; ++j) acc += a_row[j] * x[j * incx];
        };
        y[i] = (beta == T(0)) ? alpha * acc : alpha * acc + beta * y[i];
    };
});
```
 
For row-major matrix $A$, `a_row[j]` for $j = 0\ldots N{-}1$ is a contiguous scan that enables optimal cache utilisation. `x[j]` is also contiguous when `incx == 1`. Parallelism is over rows: each thread owns a disjoint range of output elements $y[r_s, r_e)$ with no write conflicts and no reduction required. The `beta == T(0)` special case overwrites $y_i$ without reading it, avoiding a potential NaN propagation.

- Column-major kernel

For `ColMajor` storage, $A[i,j]$ resides at address `data + j*lda + i`: iterating over $j$ for fixed $i$ accesses memory with stride `lda`, producing one cache miss per element for large `lda`. The kernel transposes the loop order to a SAXPY-like sweep over columns:
 
$$y \mathrel{+}= x_0 \cdot A[:,0] \;+\; x_1 \cdot A[:,1] \;+\; \ldots$$
 
Each inner axpy reads a contiguous column of $A$. For the parallel path, since the SAXPY loop cannot be parallelised over $j$ without write conflicts on $y$, each thread accumulates into a private local vector of size $M$:
 
After the join barrier, the local vectors are summed into $y$ via a vectorised reduction loop. The private accumulators are `std::vector<T, AlignedAllocator<T>>`, each separately 64-byte aligned to avoid false sharing between threads.

---
### `trsv`: single RHS triangular solve
```cpp
trsv('U', 'N', 'N', expr(A), x); // Upper, no-trans, non-unit diagonal.
trsv('L', 'N', 'U', expr(A), x); // Lower, no-trans, unit diagonal.
trsv('U', 'T', 'N', expr(A), x); // Upper transpose.
trsv('L', 'T', 'N', expr(A), x); // Lower transpose.
trsv('U', 'C', 'N', expr(A), x); // Upper conjugate-transpose.
```
 
**Operation.** Solve $op(A)\,x = b$ in-place, where $b$ is the input value of $x$ and $op \in \{I, {}^\top, {}^H\}$. `trsv` is inherently serial: each component depends on all previously computed components.

- Non-transposed cases

    * **Upper** - backward substitution. For each $i$ from $N{-}1$ down to $0$:
    $$x_i = \frac{x_i - \sum_{j > i} A_{ij} x_j}{A_{ii}}$$
     * **Lower** - forward substitution. For each $i$ from $0$ to $N{-}1$:
    $$x_i = \frac{x_i - \sum_{j < i} A_{ij} x_j}{A_{ii}}$$
    Both inner loops are annotated with `LINALG_VECTORIZE`. For RowMajor storage, the access $A(i,j)$ for $j > i$ (upper) or $j < i$ (lower) is contiguous in the inner loop, so the auto-vectoriser can emit an efficient dot product.

- Transposed cases - gather loops

    * **Upper transposed**. Solving $A^\top x = b$ for upper $A$ expands the $j$-th equation as:
 
    $$\sum_{i \le j} A_{ij} x_i = b_j \implies x_j = \frac{b_j - \sum_{i < j} A_{ij} x_i}{A_{jj}}$$
 
    All contributions to $x_j$ from already-resolved $x_{0 \ldots j-1}$ must be **gathered** before the diagonal division.

    * **Lower transposed — backward gather.** For each $i$ from $N{-}1$ down to $0$:
 
    $$x_i = \frac{x_i - \sum_{k > i} A_{ki} x_k}{A_{ii}}$$

In both transposed cases the `do_conj` flag is checked only when forming the pre-computed local `aji` / `aki`, adding at most one branch per outer iteration with zero overhead in the inner loop.

---
### `trmv`: triangular matrix-vector-product
```cpp
trmv('U', 'N', 'N', expr(A), x);  // x = A * x.
trmv('L', 'T', 'U', expr(A), x);  // x = A^T * x, unit diagonal.
trmv('U', 'C', 'N', expr(A), x);  // x = A^H * x.
```
 
**Operation:** $x \leftarrow op(A)\,x$ in-place.
 
Computing $x^\text{new}_i = \sum_j A_{ij} x^\text{old}_j$ while overwriting $x$ corrupts the source. The implementation resolves this by writing into a zeroed temporary `tmp`, then copying `tmp` back to `x`. The updated source completely rewrites `trmv` into eight specialised kernels (placed in `detail::trmv`) parameterised at compile time by `Layout L`, `bool Unit`, and `bool DoConj`, eliminating all runtime branching inside the hot loops. Layout and diagonal flags are encoded as `std::integral_constant` values so template arguments are resolved at instantiation time:
```cpp
if (!do_trans) {
    if (unit) dispatch_notrans(std::true_type{});
    else dispatch_notrans(std::false_type{});
} else {
    if (unit && do_conj) dispatch_trans(std::true_type{}, std::true_type{});
    // etc.
};
```

- Non-transposed kernels (`trmv_notrans_upper`, `trmv_notrans_lower`)
 
Both no-transpose kernels are **parallelised over rows** via `parallel_for`. Each row computes a dot product of the relevant triangle row with the original `x`. For instance,
```cpp
template<typename T, Layout L, bool Unit>
LINALG_INLINE void trmv_notrans_upper(const T* LINALG_RESTRICT Ap, size_t lda, const T* LINALG_RESTRICT xp, T* LINALG_RESTRICT tp, size_t N) {
    parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [=](size_t rs, size_t re) {
        for (size_t i = rs; i < re; ++i) {
            T sum;
            if constexpr (L == Layout::RowMajor) {
                const T* LINALG_RESTRICT row = Ap + i * lda;
                sum = Unit ? xp[i] : row[i] * xp[i];
                LINALG_VECTORIZE
                for (size_t j = i + 1; j < N; ++j) sum += row[j] * xp[j];
            } else {
                // ColMajor: A[i,j] = Ap[j*lda + i].
                sum = Unit ? xp[i] : Ap[i * lda + i] * xp[i];
                // j > i: each column j is non-contiguous in i but x[j] is scalar.
                for (size_t j = i + 1; j < N; ++j) sum += Ap[j * lda + i] * xp[j];
            };
            tp[i] = sum;
        };
    });
};
```

For RowMajor storage the off-diagonal run `row[j]` for $j > i$ is contiguous, enabling the `LINALG_VECTORIZE` pragma.
- Transposed kernels (`trmv_trans_upper`, `trmv_trans_lower`)
 
The transposed operation computes $tp_k = \sum_j A^\top_{kj} x_j = \sum_j A_{jk} x_j$. This is a scatter pattern over columns of $A^\top$ (equivalently, rows of $A$). Since the scatter writes to multiple locations of `tp` (which would conflict if parallelised naively), both transposed kernels use **thread-local accumulator pattern** (as in `gemv_kernel_col`):
```cpp
if (num_threads <= 1) {
    // Serial scatter: for each column j of A (= row j of A^T), update tp[0...j] with A[k,j]*x[j] for k <= j.
    for (size_t j = 0; j < N; ++j) {
        const T xj = xp[j];
        tp[j] += (Unit ? T(1) : Aval(j, j)) * xj;
        for (size_t k = j + 1; k < N; ++k) tp[k] += Aval(j, k) * xj;
    };
    return;
};
using AlignedVec = std::vector<T, AlignedAllocator<T>>;
std::vector<AlignedVec> locals(num_threads, AlignedVec(N, T(0)));
std::vector<std::future<void>> futures;
futures.reserve(num_threads);
for (size_t t = 0; t < num_threads; ++t) {
    futures.push_back(pool.enqueue([=, &locals, &Aval]() {
        T* LINALG_RESTRICT loc = detail::assume_aligned<64>(locals[t].data());
        for (size_t j = t; j < N; j += num_threads) {
            const T xj = xp[j];
            loc[j] += (Unit ? T(1) : Aval(j, j)) * xj;
            for (size_t k = j + 1; k < N; ++k) loc[k] += Aval(j, k) * xj;
        };
    }));
}; // Followed by: tp[i] += sum over locals.
```

---
### `ger` and `gerc`: rank-1 updates
```cpp
ger (alpha, expr(x), expr(y), A); // A += alpha * x * y^T
gerc(alpha, expr(x), expr(y), A); // A += alpha * x * conj(y)^T
```

**Operations:** $A_{ij} \leftarrow A_{ij} + \alpha x_i y_j$ and $A_{ij} \leftarrow A_{ij} + \alpha x_i \overline{y_j}$ respectively.

In the updated source, `ger` and `gerc` share a single `ger_kernel<T, L, Conj>` template. The `Conj` boolean is a **compile-time parameter**, eliminating the runtime branch or pre-conjugation pass that appeared in the previous version. The inner loop of the RowMajor path reads:
 
```cpp
if constexpr (Conj) {
    LINALG_VECTORIZE
    for (size_t j = 0; j < N; ++j) a_row[j] += xi * conj(ya[j]);
} else {
    LINALG_VECTORIZE
    for (size_t j = 0; j < N; ++j) a_row[j] += xi * ya[j];
};
```
 
The `if constexpr` is resolved at instantiation, so each specialisation contains exactly one branch-free vectorised loop. For real types, `Conj = true` instantiates the same loop body as `Conj = false` since `conj` is the identity, producing bit-identical compiled code.
 
Before invoking the kernel, both `x` and `y` are resolved via `resolve_vec<T>`. If either has a non-unit stride, it is materialised into a unit-stride buffer so the inner SAXPY loop can use `LINALG_VECTORIZE` without gather penalties.
 
- RowMajor kernel is parallelised over rows. For each row $i$, `xi = alpha * x[i]` is hoisted out of the $j$-loop, reducing the inner loop from two multiplies per element to one.
 
- ColMajor kernel. Parallelised over columns. For each column $j$, `yj = alpha * (Conj ? conj(y[j]) : y[j])` is hoisted, then a vectorised axpy updates the contiguous column of $A$.

---
### `symv` and `hemv`: symmetric/Hermitian matrix-vector products
```cpp
symv(uplo, alpha, expr(A), expr(x), beta, y); // y = alpha * A * x + beta * y, A is symmetric.
hemv(uplo, alpha, expr(A), expr(x), beta, y); // y = alpha * A * x + beta * y, A - Hermitian.
```
 
**Operations:** $y \leftarrow \alpha A x + \beta y$ where $A$ is symmetric ($A = A^\top$) or Hermitian ($A = A^H$) and only the triangle specified by `uplo` is referenced.

Both routines are implemented via a single shared `symv_hemv_kernel<T, L, DoConj>` function. The `DoConj` template parameter selects between symmetric and Hermitian behaviour at compile time, completely eliminating the runtime conditional in the hot loop. The implementation also extends pointer extraction to the matrix operand: `raw_mat_info<T>` is attempted on `A`, and the layout obtained is forwarded to the kernel (which handles both `Layout::RowMajor` and `Layout::ColMajor` via the `Aload` lambda). This eliminates the unconditional RowMajor materialisation.
 
**Two-vector sweep.** The key algorithm is processing of the stored triangle in a single pass that simultaneously updates *both* the direct and the symmetric scatter contributions. Each stored element $A_{ij}$ is touched exactly once, halving the number of cache-line loads compared to a two-pass implementation:
```cpp
// Upper storage branch:
for (size_t i = 0; i < N; ++i) {
    // Diagonal (real part for Hermitian).
    const T aii = DoConj ? static_cast<T>(std::real(Aload(i, i))) : Aload(i, i);
    T sum_i = aii * xp[i];
    for (size_t j = i + 1; j < N; ++j) {
        const T aij = Aload(i, j);
        sum_i += aij * xp[j];
        // Symmetric contribution: A[j,i] = conj(A[i,j]) for Hermitian.
        if constexpr (DoConj) yp[j] += alpha * conj(aij) * xp[i];
        else yp[j] += alpha * aij * xp[i];
    };
    yp[i] += alpha * sum_i;
};
```

Parallelising the serial two-vector sweep directly would cause write conflicts on `yp[j]` (the scatter target). The parallel path separates the two contributions into independent arrays:
 
- `row_acc[t][i]`: direct dot-product contribution to $y_i$ accumulated by thread $t$ over its assigned range of $i$ values.
- `scatter[t][j]`: symmetric contributions to $y_j$ accumulated by thread $t$ over the same range.
Threads are assigned contiguous blocks of rows using static partitioning. After the join barrier, a final `parallel_for` reduces both arrays into `yp[i]`:
```cpp
parallel_for(N, PARALLEL_THRESHOLD_COMPUTE, [&](size_t rs, size_t re) {
    for (size_t i = rs; i < re; ++i) {
        T acc = row_acc[0][i];
        LINALG_VECTORIZE
        for (size_t t = 1; t < num_threads; ++t) acc += row_acc[t][i];
        T sct = scatter[0][i];
        LINALG_VECTORIZE
        for (size_t t = 1; t < num_threads; ++t) sct += scatter[t][i];
        yp[i] += alpha * (acc + sct);
    };
});
```

This is analogous to the private-accumulator reduction in `gemv_kernel_col`, with the addition of a separate `scatter` array to hold the off-diagonal symmetric contributions.

---
## Level 3: matrix-matrix oprations

Level 3 routines perform $O(MNK)$ arithmetic with only $O(MN + NK + MK)$ data traffic. Arithmetic intensity scales as $O(\min(M,N,K))$, meaning that the routines become compute-bound for large dimensions. Cache blocking and parallelism strategy are the primary performance drivers.
- [`gemm`](#gemm-general-matrix-matrix-product)
- [`trsm`](#trsm-multiple-rhs-triangular-solve)
- [`syrk` / `herk`](#syrk-and-herk-rank-k-updates)

---
### `gemm`: general matrix-matrix product
```cpp
// C = alpha * A * B + beta * C, A is M * K, B is K * N, C is M * N.
gemm(alpha, expr(A), expr(B), beta, C);
```
 
**Operation:** $C \leftarrow \alpha AB + \beta C$.

Before any multiply-add work the entire flat $C$ array is rescaled in a single vectorised `parallel_for` pass over $M \times N$ contiguous elements. For $\beta = 0$, `std::fill` is used to unconditionally zero $C$, avoiding a potential NaN propagation. Then `raw_mat_info<T>` is attempted on both $A$ and $B$. On failure, `detail::materialise<T, L>` creates a fresh `Matrix<T, L>` via a parallel layout-aware copy.

**Problem dispatch:**
```cpp
if (M * N * K < PARALLEL_THRESHOLD_COMPUTE * 10)
    gemm_microkernel_{row|col}(0, M, 0, N, 0, K); // Direct.
else
    gemm_blocked<T, L>(...); // Tiled.
```
 
Below approximately 41000 total FMAs, the matrices fit in L1/L2 cache and blocking provides no cache-reuse benefit, while tile bookkeeping adds extra cost.

**Microkernels**

The auto-vectorised version (present as commented-out in the source code) was benchmarked and found slower for all scalar types. The explicit 8-wide unroll avoids compiler conservatism at parametric tile boundaries:
 
- **Row-major microkernel** (outer-product form, $i \to k \to j$):
```cpp
for (size_t i = i0; i < i1; ++i) {
    const T* LINALG_RESTRICT a_row = A + i * lda;
    T* LINALG_RESTRICT c_row = C + i * ldc;
    for (size_t k = k0; k < k1; ++k) {
        const T a_ik = alpha * a_row[k]; // Broadcast scalar.
        const T* b_row = B + k * ldb;
        size_t j = j0;
        for (; j + 8 <= j1; j += 8) {
            c_row[j] += a_ik * b_row[j];
            c_row[j+1] += a_ik * b_row[j+1];
            // etc. through j+7.
        };
        for (; j < j1; ++j) c_row[j] += a_ik * b_row[j];
    };
};
```

For each $(i, k)$ pair, `a_ik` is a broadcast scalar and `b_row` is a contiguous row - optimal access for RowMajor $B$. The 8 independent `c_row[j..j+7] +=` statements form 8 independent FMA chains that the out-of-order scheduler overlaps. On AVX2 (256-bit, 4 doubles), the compiler groups these into two 4-wide SIMD FMAs, saturating both FP execution ports.
 
- **Column-major microkernel** (dual structure, $j \to k \to i$): hoists `b_kj = alpha * b_col[k]` per $(j, k)$ pair, then performs an 8-wide unroll over $i$ into a contiguous column of $C$.

**Blocked algorithm**

The blocked path uses tile size $b_s = L1\_\text{BLOCK} \times 2 = 64$. A $64 \times 64$ double tile is $64 \times 64 \times 8 = 32\,\text{KB}$, fitting in typical L1d cache. The $(i,j)$ tile grid is linearised into $\lceil M/b_s \rceil \times \lceil N/b_s \rceil$ tiles and distributed round-robin across threads. Each thread runs the full $k$-loop serially for each of its $(i,j)$ tiles, keeping the $C$-tile resident in register/L1 during accumulation with no inter-thread reduction on $C$.

---
### `trsm`: multiple RHS triangular solve
```cpp
trsm('L', 'U', 'N', 'N', alpha, expr(A), B); // op(A) * X = alpha * B.
trsm('R', 'L', 'T', 'N', alpha, expr(A), B); // X * op(A) = alpha * B.
```

**Operation.** Overwrite $B$ with $X$ satisfying $op(A)\,X = \alpha B$ (left) or $X\,op(A) = \alpha B$ (right).

$B$ is prescaled in-place via a layout-aware parallel loop before any solve. For row-major $B$: parallel over rows with a vectorised column scan; for column-major $B$: parallel over columns with a vectorised row scan, ensuring the inner loop always accesses memory contiguously.
 
**Left solve.** Each column $j$ of matrix $B$ satisfies the independent system $op(A)\,x_j = b_j$. Columns are distributed round-robin across threads. For row-major $B$, column $j$ is strided; each thread gathers it into a contiguous buffer, calls `trsm_col_solve`, and scatters the solution back. For column-major $B$, column $j$ is contiguous and the solve operates in-place. `trsm_col_solve` implements the same backward/forward substitution with gather loops as `trsv`, covering all four `(uplo, trans)` combinations.
 
**Right solve:** $X\,op(A) = B$ is equivalent to $op(A)^\top X^\top = B^\top$. This is solved row-by-row: each row of $B$ is an independent right-hand side for the transposed system, with `trans` flipped (`'N'` to `'T'`). For the conjugate-transpose case (`trans = 'C'`), each row is conjugated before and after the solve via a vectorised in-place pass.

---
### `syrk` and `herk`: rank-k updates
```cpp
syrk(uplo, trans, alpha, expr(A), beta, C);  // C = alpha * op(A) * op(A)^T + beta * C
herk(uplo, trans, alpha, expr(A), beta, C);  // C = alpha * op(A) * op(A)^H + beta * C
```
 
**Operations.** $C \leftarrow \alpha \cdot op(A)\,op(A)^\top + \beta C$ and $C \leftarrow \alpha \cdot op(A)\,op(A)^H + \beta C$, where $C$ is $N \times N$ (Hermitian-)symmetric and only the triangle specified by `uplo` is written.
 
`herk` is a thin wrapper over `syrk_impl` with `conjugate = true` and the real `alpha`/`beta` widened to `T`. Both routines share `syrk_core`.
 
**Why prefer `syrk` over `gemm(A, transpose(A))`-like calls?** `gemm` writes all $N^2$ elements of $C$ and materialises the transpose. `syrk` writes only $N(N+1)/2 \approx N^2/2$ elements, touches each stored element once, and leaves the unreferenced triangle completely unchanged - an important invariant for downstream routines such as Cholesky factorisation.

Only the referenced triangle is prescaled. For $\beta = 0$, `std::fill` is used on each relevant row segment (RowMajor) or the scalar path visits individual elements (ColMajor). The unreferenced triangle is not touched.
 
Rather than computing $A[i,k]$ on demand from the physical layout during the inner product, `syrk_core` first packs $A$ into a row-major buffer:
 
$$\text{Ab}[i \cdot K + k] = \text{logical\_}A(i, k)$$
 
This normalises all four `(Layout, notrans)` combinations into a single memory access pattern for the inner product. The packing is parallelised and vectorised:
```cpp
for (size_t i = rs; i < re; ++i) {
    T* LINALG_RESTRICT dst = Ab + i * K;
    if constexpr (L == Layout::RowMajor) {
        if (notrans) {
            const T* src = ap + i * lda;
            LINALG_VECTORIZE for (size_t k = 0; k < K; ++k) dst[k] = src[k];
        } else {
            LINALG_VECTORIZE for (size_t k = 0; k < K; ++k) dst[k] = ap[k * lda + i];
        };
    } else {
        if (notrans) {
            LINALG_VECTORIZE for (size_t k = 0; k < K; ++k) dst[k] = ap[k * lda + i];
        } else {
            const T* src = ap + i * lda;
            LINALG_VECTORIZE for (size_t k = 0; k < K; ++k) dst[k] = src[k];
        };
    };
};
```

After packing, the outer-product computation uses a blocked tiled loop over $i$-tiles of size $b_s = 64$, parallelised at tile level. Within each tile pair $(t_i, t_j)$, an 8-wide unrolled dot product computes the $(i,j)$ entry of $op(A)\,op(A)^{T/H}$ (cf. the algorithm [here](#dot-real-dot-product)):
```cpp
T s0{}, s1{}, s2{}, s3{}, s4{}, s5{}, s6{}, s7{};
const size_t K8 = (K / 8) * 8;
if (conjugate) {
    LINALG_VECTORIZE
    for (size_t k = 0; k < K8; k += 8) {
        s0 += ai[k ] * conj(aj[k]);
        s1 += ai[k+1] * conj(aj[k+1]);
        // etc. through s7.
    };
    for (size_t k = K8; k < K; ++k) s0 += ai[k] * conj(aj[k]);
} else {
    LINALG_VECTORIZE
    for (size_t k = 0; k < K8; k += 8) {
        s0 += ai[k] * aj[k];
        s1 += ai[k+1] * aj[k+1];
        // etc. through s7.
    };
    for (size_t k = K8; k < K; ++k) s0 += ai[k] * aj[k];
};
const T s = ((s0+s1)+(s2+s3)) + ((s4+s5)+(s6+s7));
```

The 8-accumulator structure gives 8 independent FMA chains. The pairwise tree reduction $((s_0{+}s_1)+(s_2{+}s_3))+((s_4{+}s_5)+(s_6{+}s_7))$ minimises floating-point rounding error by keeping the binary tree balanced. Parallelism is over $i$-tiles: each tile is assigned to a thread, which iterates over all $j$-tiles in the relevant triangle half, accumulating the full dot product for each $(i,j)$ entry before writing. This ensures no two threads write to the same output element, requiring no reduction on $C$.
 
For `herk`, the `conjugate` branch activates `conj(aj[k])` in the inner product, computing $a_i[k] \cdot \overline{a_j[k]}$ - the $(i,j)$ entry of $op(A)\,op(A)^H$. For real types `conj` is the identity and the compiler emits the same code as the `conjugate = false` path.

---
## Quick reference
- ### `trans` semantics

| `trans` | `trsv` / `trsm` | `trmv` | `syrk` | `herk` |
|---|---|---|---|---|
| `'N'` | $A$ | $A$ | $op(A) = A$ | $op(A) = A$ |
| `'T'` | $A^\top$ (gather) | $A^\top$ (index fix) | $op(A) = A^\top$ | $op(A) = A^\top$ |
| `'C'` | $A^H$ (gather) | $A^H$ (index fix) |  | $op(A) = A^H$ |
 
[`gemv`](#gemv-general-matrix-vector-multiply) has no `trans` parameter. Passing `transpose(A)` or `hermitian(A)` triggers materialisation of the transposed view; for repeated use it is strongly recommended to materialise once explicitly before the loop.
 
- ### `uplo` semantics
| `uplo` | Triangle read/written | Symmetric access | Hermitian access |
|---|---|---|---|
| `'U'` | $i \le j$ | $A[j,i]$ | $\overline{A[j,i]}$ |
| `'L'` | $i \ge j$ | $A[j,i]$ | $\overline{A[j,i]}$ |
 
- ### Output requirements
| Routine | Output | Notes |
|---|---|---|
| `axpy`, `axpby` | `Vector<T>` or unit-stride `VectorView<T,true>`. | Accumulates or overwrites. |
| `scal` | `Vector<T>`, `Matrix<T,L>`, or mutable views. | In-place. |
| `gemv` | `Vector<T>` or unit-stride `VectorView<T,true>`. | Pre-sized to $M$. |
| `trsv` | `Vector<T>` | Overwritten in-place: $b$ on entry, $x$ on exit. |
| `trmv` | `Vector<T>` or unit-stride `VectorView<T,true>`. | In-place; uses internal temporary. |
| `ger`, `gerc` | `Matrix<T,L>` or mutable `MatrixView`.| Accumulates. |
| `symv`, `hemv` | `Vector<T>`. | Accumulates. |
| `gemm` | `Matrix<T,L>`. | Pre-allocated $M \times N$. |
| `trsm` | `Matrix<T,L>` or mutable `MatrixView`. | Overwritten in-place: $B$ on entry, $X$ on exit. |
| `syrk`, `herk` | `Matrix<T,L>` or mutable `MatrixView`. | Square $N \times N$ required. |

---