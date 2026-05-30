#pragma once

// Portable performance infrastructure with no platform intrinsic headers included.
// All SIMD is delegated to the auto-vectoriser; the macros and helpers below give it the information it needs to generate wide, aligned loads/stores.

#include <cstddef>
#include <cstdint>
#include <new>
#include <algorithm>
#include <type_traits>

namespace linalg {
    // Compiler detection.
    #if defined(_MSC_VER)
    #define LINALG_MSVC 1
    #elif defined(__clang__)
    #define LINALG_CLANG 1
    #elif defined(__GNUC__)
    #define LINALG_GCC 1
    #endif

    // Force-inlining.
    #if defined(LINALG_MSVC)
    #define LINALG_INLINE __forceinline
    #elif defined(LINALG_CLANG) || defined(LINALG_GCC)
    #define LINALG_INLINE [[gnu::always_inline]] inline
    #else
    #define LINALG_INLINE inline
    #endif

    // No-alias qualifier.
    #if defined(LINALG_MSVC)
    #define LINALG_RESTRICT __restrict
    #elif defined(LINALG_CLANG) || defined(LINALG_GCC)
    #define LINALG_RESTRICT __restrict__
    #else
    #define LINALG_RESTRICT
    #endif

    // Vectorisation macros.
    #if defined(LINALG_CLANG)
    #define LINALG_VECTORIZE _Pragma("clang loop vectorize(enable) interleave(enable)")
    #define LINALG_UNROLL(N) _Pragma("clang loop unroll_count(" #N ")")
    #elif defined(LINALG_GCC)
    #define LINALG_PRAGMA(x) _Pragma(#x)
    #define LINALG_VECTORIZE LINALG_PRAGMA(GCC ivdep)
    #define LINALG_UNROLL(N) LINALG_PRAGMA(GCC unroll N)
    #elif defined(LINALG_MSVC)
    #define LINALG_VECTORIZE __pragma(loop(ivdep))
    #define LINALG_UNROLL(N)
    #else
    #define LINALG_VECTORIZE
    #define LINALG_UNROLL(N)
    #endif

    // Prefetch.
    #if defined(LINALG_CLANG) || defined(LINALG_GCC)
    #define LINALG_PREFETCH(addr, rw, loc) __builtin_prefetch((addr), (rw), (loc))
    #else
    #define LINALG_PREFETCH(addr, rw, loc) ((void)0)
    #endif

    namespace detail {
        /// @brief Informs the optimiter that pointer starts on an N-byte boundary.
        template<std::size_t N, typename T>
        LINALG_INLINE T* assume_aligned(T* p) noexcept {
            #if defined(__cpp_lib_assume_aligned)          // C++20 §20.10.6
                return std::assume_aligned<N>(p);
            #elif defined(LINALG_CLANG) || defined(LINALG_GCC)
                return static_cast<T*>(__builtin_assume_aligned(p, N));
            #elif defined(LINALG_MSVC)
                __assume(reinterpret_cast<std::uintptr_t>(p) % N == 0);
                return p;
            #else
                return p;
            #endif
        };
        
        /// @brief Informs the optimiter that pointer starts on an N-byte boundary.
        template<std::size_t N, typename T>
        LINALG_INLINE const T* assume_aligned(const T* p) noexcept {
            #if defined(__cpp_lib_assume_aligned)
                return std::assume_aligned<N>(p);
            #elif defined(LINALG_CLANG) || defined(LINALG_GCC)
                return static_cast<const T*>(__builtin_assume_aligned(p, N));
            #elif defined(LINALG_MSVC)
                __assume(reinterpret_cast<std::uintptr_t>(p) % N == 0);
                return p;
            #else
                return p;
            #endif
        };

        // Compile-time constants that mirror the widest ISA extension visible to the translation unit.
        // Used to derive unroll factors without including any intrinsic headers.
        #if defined(__AVX512F__)
        inline constexpr std::size_t simd_bytes = 64;
        #elif defined(__AVX2__) || defined(__AVX__)
        inline constexpr std::size_t simd_bytes = 32;
        #elif defined(__SSE2__) || defined(__ARM_NEON)
        inline constexpr std::size_t simd_bytes = 16;
        #else
        inline constexpr std::size_t simd_bytes = 8;   // scalar / unknown arch
        #endif
        
        /// @brief Elements per SIMD register for scalar type T.
        /// @tparam T Scalar type.
        template<typename T>
        inline constexpr std::size_t simd_lanes = simd_bytes / sizeof(T);
        
        /// @brief Recommended manual unroll depth.
        /// @tparam T Scalar type.
        template<typename T>
        inline constexpr std::size_t unroll_factor = std::max<std::size_t>(4u, 2u * simd_lanes<T>);
    };

    /// @brief Replacement for std::allocator<T>. Every allocation starts on an Align-byte boundary. Default 64 B is one cache line, equivalent to widest AVX-512 store.
    /// @tparam T Scalar type.
    /// @tparam Align 
    template<typename T, std::size_t Align = 64>
    struct AlignedAllocator {
        static_assert((Align & (Align - 1)) == 0, "Align must be a power-of-two");
        static_assert(Align >= alignof(T), "Align must be >= alignof(T)");
    
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using propagate_on_container_move_assignment = std::true_type;
        using is_always_equal = std::true_type;
    
        template<typename U>
        struct rebind { using other = AlignedAllocator<U, Align>; };
    
        constexpr AlignedAllocator() noexcept = default;

        template<typename U>
        constexpr AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {};
    
        [[nodiscard]] T* allocate(std::size_t n) {
            if (n == 0) return nullptr;
            return static_cast<T*>(
                ::operator new(n * sizeof(T), std::align_val_t{Align}));
        };
    
        void deallocate(T* p, std::size_t n) noexcept {
            ::operator delete(p, n * sizeof(T), std::align_val_t{Align});
        };
    };
    
    template<typename T, typename U, std::size_t A>
    bool operator==(const AlignedAllocator<T,A>&, const AlignedAllocator<U,A>&) noexcept { return true;  };
    template<typename T, typename U, std::size_t A>
    bool operator!=(const AlignedAllocator<T,A>&, const AlignedAllocator<U,A>&) noexcept { return false; };
};