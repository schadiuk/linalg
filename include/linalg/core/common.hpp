#pragma once

#include <complex>
#include <cmath>
#include <type_traits>
#include <concepts>
#include <random>

namespace linalg {
    // Default scalar type for matrices, using complex double for generality
    using DefaultScalar = std::complex<double>;

    // Enumeration for matrix storage layout
    enum class Layout { RowMajor, ColMajor };

    // Concept for scalar types: arithmetic types or complex float/double
    template<typename T>
    concept Scalar = std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

    // Concept requiring two types to have compatible dimensions (same matrix rows and columns number)
    template<typename T, typename U>
    concept CompatibleDimensions = requires(T t, U u) {
	    { t.rows() == u.rows() } -> std::same_as<bool>;
	    { t.cols() == u.cols() } -> std::same_as<bool>;
    };

    // Conjugate helper function: returns complex conjugate for complex types, identity for real types
    template<typename T>
    T conj(const T& z) {
	    if constexpr (std::is_same_v<T, std::complex<float>> ||
		    std::is_same_v<T, std::complex<double>>) {
		    return std::conj(z);
	    }
		else return z;
    };

    // Generates a random scalar of type T, with real part in [-1,1], imaginary in [-1,1]
    template<typename T>
    T randomScalar() {
	    thread_local static std::random_device rd;
	    thread_local static std::mt19937 gen(rd());
	    thread_local static std::uniform_real_distribution<double> dis(-1., 1.);
	    if constexpr (std::is_same_v<T, std::complex<double>>) {
		    return T(dis(gen), dis(gen));
	    }
	    else if constexpr (std::is_same_v<T, std::complex<float>>) {
		    return T(static_cast<float>(dis(gen)), static_cast<float>(dis(gen)));
	    }
	    else {
		    return static_cast<T>(dis(gen));
	    };
    };

    // Forward declarations
    template<typename T, bool Mutable> class VectorView;
	template<typename T, Layout L, bool Trans, bool Conj, bool Mutable> class MatrixView;
	template<typename U> struct MatExpr;
	template<typename U> struct VecExpr;
};