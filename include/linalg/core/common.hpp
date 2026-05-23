#pragma once

#include <complex>
#include <cmath>
#include <type_traits>
#include <concepts>
#include <random>

namespace linalg {
    /// @brief Default scalar type for matrices, using complex double for generality.
    using DefaultScalar = std::complex<double>;

    /// @brief Enumeration for matrix storage layout.
    enum class Layout { RowMajor, ColMajor };

    /// @brief Concept for scalar types: arithmetic types or complex float/double.
    template<typename T>
    concept Scalar = std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

    /// @brief Conjugate helper function.
	/// @param z Complex number.
	/// @return Complex conjugate of z.
    template<typename T>
    T conj(const T& z) {
	    if constexpr (std::is_same_v<T, std::complex<float>> ||
		    std::is_same_v<T, std::complex<double>>) {
		    return std::conj(z);
	    }
		else return z;
    };

    /// @brief Generator of random scalars of type T.
	/// @return A complex number with real and imaginary parts distributed uniformly on [-1, 1].
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

    template<typename T, bool Mutable> class VectorView;
	template<typename T, Layout L, bool Trans, bool Conj, bool Mutable> class MatrixView;
	template<typename U> struct MatExpr;
	template<typename U> struct VecExpr;
};