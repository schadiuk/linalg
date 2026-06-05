# BLAS Reference
This document explores every BLAS-level operation in the `linalg` library: the mathematical definition of each routine, a full account of the internal algorithm and engineering decisions behind it, and usage examples.
---
## Table of contents
- [Parallelisation model](#parallelisation-model)
- [Conventions](#conventions)
- [BLAS-1](#level-1-vector-operations)
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
 
$$\text{num\_threads} = \min\!\bigl(\text{pool.thread\_count()},\;\lceil\text{total}/\text{threshold}\rceil\bigr)$$
 
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
| `PARALLEL_THRESHOLD_REDUCE` | 16384 | Reduction passes (`dot`, `nrm2`, `asum`). Per-element work is light but the sequential accumulation bottleneck justifies earlier threading. |
 
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
| `side` | `'L'` / `'R'` | Triangular factor multiplies from the keft or right. |
 
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

**Operation:**  $\sum_i \bigl(|\operatorname{Re}(x_i)| + |\operatorname{Im}(x_i)|\bigr)$. That is $\sum_i |x_i|$ for real types.

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
 
**Algorithm.** Compute $r = \operatorname{hypot}(a, b)$, which is overflow-safe by specification. Then $c = a/r$, $s = b/r$. The sign of $r$ is assigned to match whichever of $|a|$, $|b|$ is larger:
 
```cpp
if (std::abs(a) >= std::abs(b)) r = std::copysign(r, a);
else r = std::copysign(r, b);
```

This minimises cancellation in the subsequent divisions. If $|a|$ dominates and $r = \operatorname{sign}(a) \cdot \operatorname{hypot}$, then $c = a/r \approx 1$ (no cancellation) and $s = b/r$ is small.

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