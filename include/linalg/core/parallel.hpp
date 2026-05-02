#pragma once

#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace linalg {
    // Parallelisation thresholds
    constexpr size_t PARALLEL_THRESHOLD_SIMPLE  = 65536;
    constexpr size_t PARALLEL_THRESHOLD_COMPUTE = 4096;
    constexpr size_t PARALLEL_THRESHOLD_REDUCE  = 16384;
    // Block sizes for cache-friendly operations
    constexpr size_t L1_BLOCK = 32;
    constexpr size_t L2_BLOCK = 128;

	#ifdef __cpp_lib_hardware_interference_size
		inline constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
	#else
		inline constexpr size_t CACHE_LINE_SIZE = 64;
	#endif

    // Padded struct to avoid false sharing
    template<typename T>
    struct alignas(CACHE_LINE_SIZE) Padded {
        T value;
        Padded() : value(T{}) {};
        Padded(const T& v) : value(v) {};
        operator T& () { return value; };
        operator const T& () const { return value; };
    };

    // Singleton thread pool for parallel execution
    class ThreadPool {
    public:
        static ThreadPool& instance() {
            static ThreadPool pool;
            return pool;
        };

        template<typename F>
        auto enqueue(F&& f) -> std::future<decltype(f())> {
            using return_type = decltype(f());
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::forward<F>(f)
            );
            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
                tasks_.emplace([task]() { (*task)(); });
            }
            condition_.notify_one();
            return res;
        };

        // Get the number of worker threads
        size_t thread_count() const { return workers_.size(); };

        // Destructor: stop all threads and join them
        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                stop_ = true;
            }
            condition_.notify_all();
            for (std::thread& worker : workers_) {
                if (worker.joinable()) worker.join();
            };
        };

    private:
        // Constructor: create worker threads
        ThreadPool() : stop_(false) {
            size_t num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
            for (size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back([this] {
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
                });
            };
        };

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_;
    };

    // Parallel execution of a lambda
    template<typename F>
    void parallel_for(size_t total, size_t threshold, F&& func) {
        if (total < threshold) {
            func(0, total);
            return;
        };
        auto& pool = ThreadPool::instance();
        size_t num_threads = std::min(pool.thread_count(),
            (total + threshold - 1) / threshold);
        std::vector<std::future<void>> futures;
        size_t chunk = total / num_threads;
        size_t remainder = total % num_threads;
        size_t offset = 0;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t count = chunk + (t < remainder ? 1 : 0);
            size_t start = offset;
            size_t end = start + count;
            futures.push_back(pool.enqueue([&func, start, end]() {
                func(start, end);
            }));
            offset += count;
        };
        for (auto& f : futures) { f.get(); };
    };

    // Specialised for reductions
    template<typename T, typename F>
    T parallel_reduce(size_t total, size_t threshold, F&& func) {
        if (total == 0) return T(0);
        if (total < threshold) {
            T s = T(0);
            for (size_t i = 0; i < total; ++i) s += func(i);
            return s;
        };
        auto& pool = ThreadPool::instance();
        size_t num_threads = std::min(pool.thread_count(),
            (total + threshold - 1) / threshold);
        std::vector<Padded<T>> partials(num_threads); // false sharing prevention
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        size_t chunk = total / num_threads;
        size_t remainder = total % num_threads;
        size_t offset = 0;
        for (size_t t = 0; t < num_threads; ++t) {
            const size_t count = chunk + (t < remainder ? 1 : 0);
            const size_t start = offset;
            futures.push_back(pool.enqueue(
                [&func, &partials, start, count, t]() {
                    T ps = T(0);
                    for (size_t i = start; i < start + count; ++i)
                        ps += func(i);
                    partials[t].value = ps;
                }));
            offset += count;
        };
        for (auto& f : futures) f.get();
        T result = T(0);
        for (const auto& p : partials) result += p.value;
        return result;
    };
};