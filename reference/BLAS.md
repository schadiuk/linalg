# BLAS Reference
 
This document explores every BLAS-level operation in the `linalg` library: the mathematical definition of each routine, a full account of the internal algorithm and engineering decisions behind it, and usage examples.
---
## Table of contents
- [Parallelisation model](#parallelisation-model)
---
## Parallelisation model

As `linalg` actively uses threading capabilities of C++, understanding the library's parallelisation infrastructure is a prerequisite for all further reasoning about the behaviour and design of kernels.

---
### The thread pool

The library manages a single `ThreadPool` singleton, created on first use via `ThreadPool::instance()`. Its constructor calls `std::thread::hardware_concurrency()` and spawns that many persistent worker threads, falling back to 4 if the query returns zero. Threads are created once at program start and live for the duration of the process: there is no per-call thread creation overhead.
 
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

The thresholds are shared and used in different contexts across the library. They are, alongside with cache constants, are defined at `linalg/include/core/parallel.hpp` file.
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