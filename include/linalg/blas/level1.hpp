#pragma once

#include <linalg/storage/matrix.hpp>
#include <linalg/storage/vector.hpp>

namespace linalg {
	namespace detail {
		// Type trait for complex scalars
		template<typename T> struct is_complex_impl : std::false_type {};
        template<typename T> struct is_complex_impl<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex_impl<std::remove_cvref_t<T>>::value;
	};
	// y += alpha * x
	template<typename Alpha, typename EX, typename T>
	void axpy(Alpha alpha, const VecExpr<EX>& x, Vector<T>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const size_t n = y.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i)
				y[i] += a * static_cast<T>(xx(i));
		});
	};
	
	template<typename Alpha, typename EX, typename T>
	void axpy(Alpha alpha, const VecExpr<EX>& x, VectorView<T, true>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const size_t n = y.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i)
				y(i) += a * static_cast<T>(xx(i));
		});
	};

	// y = alpha * x + beta * y
	template<typename Alpha, typename EX, typename Beta, typename T>
	void axpby(Alpha alpha, const VecExpr<EX>& x, Beta beta, Vector<T>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const T b = static_cast<T>(beta);
		const size_t n = y.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a, b](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i)
				y[i] = a * static_cast<T>(xx(i)) + b * y[i];
		});
	};
	
	template<typename Alpha, typename EX, typename Beta, typename T>
	void axpby(Alpha alpha, const VecExpr<EX>& x, Beta beta, VectorView<T, true>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const T b = static_cast<T>(beta);
		const size_t n = y.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a, b](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i)
				y(i) = a * static_cast<T>(xx(i)) + b * y(i);
		});
	};

	// In-place scaling
	template<typename Alpha, typename T>
	void scal(Alpha alpha, Vector<T>& x) {
		const T a  = static_cast<T>(alpha);
		T* data = x.data();
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [data, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) data[i] *= a;
		});
	};
	
	template<typename Alpha, typename T>
	void scal(Alpha alpha, VectorView<T, true>& x) {
		const T a = static_cast<T>(alpha);
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) x(i) *= a;
		});
	};
	
	template<typename Alpha, typename T, Layout L>
	void scal(Alpha alpha, Matrix<T, L>& A) {
		const T a = static_cast<T>(alpha);
		T* data = A.data();
		const size_t total = A.rows() * A.cols();
		parallel_for(total, PARALLEL_THRESHOLD_SIMPLE, [data, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) data[i] *= a;
		});
	};

	template<typename Alpha, typename T, Layout L, bool Trans, bool Conj>
	void scal(Alpha alpha, MatrixView<T, L, Trans, Conj, true>& A) {
		const T a = static_cast<T>(alpha);
		const size_t m = A.rows();
		const size_t n = A.cols();
		parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&A, a, n](size_t rs, size_t re) {
			for (size_t i = rs; i < re; ++i) 
				for (size_t j = 0; j < n; ++j) A(i, j) *= a;
		});
	}

	// Delegation to assignment operators (supports mutable views)
	template<typename EX, typename T>
	void copy(const VecExpr<EX>& x, Vector<T>& y) {
		BOUNDS_CHECK(x.self().size() == y.size());
		y = x;
	};
	
	template<typename EX, typename T>
	void copy(const VecExpr<EX>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.self().size() == y.size());
		y = x;
	};
	
	template<typename EM, typename T, Layout L>
	void copy(const MatExpr<EM>& A, Matrix<T, L>& B) {
		BOUNDS_CHECK(A.self().rows() == B.rows() && A.self().cols() == B.cols());
		B = A;
	};
	
	template<typename EM, typename T, Layout L, bool Trans, bool Conj>
	void copy(const MatExpr<EM>& A, MatrixView<T, L, Trans, Conj, true>& B) {
		BOUNDS_CHECK(A.self().rows() == B.rows() && A.self().cols() == B.cols());
		B = A;
	};

	// Elementwise swapping with overloads for all combinations of Vector/VectorView
	template<typename T>
	void swap(Vector<T>& x, Vector<T>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) std::swap(x[i], y[i]);
		});
	};
	
	template<typename T>
	void swap(VectorView<T, true>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) { T tmp = x(i); x(i) = y(i); y(i) = tmp; };
		});
	};
	
	template<typename T>
	void swap(Vector<T>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) { T tmp = x[i]; x[i] = y(i); y(i) = tmp; };
		});
	};
	
	template<typename T>
	void swap(VectorView<T, true>& x, Vector<T>& y) { swap(y, x); };

	// Index of maximum absolute value
	template<typename EX>
	size_t iamax(const VecExpr<EX>& x) {
		const auto& xx = x.self();
		const size_t n = xx.size();
		if (n == 0) return 0;
		size_t max_idx = 0;
		auto max_val = std::abs(xx(0));

		for (size_t i = 1; i < n; ++i) {
			auto val = std::abs(xx(i));
			if (val > max_val) {
				max_val = val;
				max_idx = i;
			};
		};
		return max_idx;
	};
	
	// Index of minimum absolute value
	template<typename EX>
	size_t iamin(const VecExpr<EX>& x) {
		const auto& xx = x.self();
		const size_t n = xx.size();
		if (n == 0) return 0;
		size_t min_idx = 0;
		auto min_val = std::abs(xx(0));

		for (size_t i = 1; i < n; ++i) {
			auto val = std::abs(xx(i));
			if (val < min_val) {
				min_val = val;
				min_idx = i;
			};
		};
		return min_idx;
	};

	// Sum of absolute values. Follows BLAS convention for complex types: abs(re(x)) + abs(im(x))
	template<typename EX>
	double asum(const VecExpr<EX>& x) {
		const auto& xx = x.self();
		const size_t n = xx.size();
		if (n == 0) return 0.0;
		using T = std::remove_cvref_t<decltype(xx(0))>;
		return parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE, [&xx](size_t i) -> double {
			if constexpr (std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>) {
				return std::abs(std::real(xx(i))) + std::abs(std::imag(xx(i)));
			} else {
				return std::abs(static_cast<double>(xx(i)));
			};
		});
	};

	// Dot product (naive)
	template<typename EX, typename EY>
	auto dot(const VecExpr<EX>& x, const VecExpr<EY>& y) {
		const auto& xx = x.self();
		const auto& yy = y.self();
		BOUNDS_CHECK(xx.size() == yy.size());
		using T = std::remove_cvref_t<decltype(xx(0) * yy(0))>;
		return parallel_reduce<T>(xx.size(), PARALLEL_THRESHOLD_REDUCE,
			[&xx, &yy](size_t i) { return xx(i) * yy(i); });
	};

	// Complex dot (taking conjugate of the first argument)
	template<typename EX, typename EY>
	auto dotc(const VecExpr<EX>& x, const VecExpr<EY>& y) {
		const auto& xx = x.self();
		const auto& yy = y.self();
		BOUNDS_CHECK(xx.size() == yy.size());
		// BLAS zdotc: conj(x)^H * y
		using T = std::remove_cvref_t<decltype(conj(xx(0)) * yy(0))>;
		return parallel_reduce<T>(xx.size(), PARALLEL_THRESHOLD_REDUCE,
			[&xx, &yy](size_t i) { return conj(xx(i)) * yy(i); });
	};

	// Euclidean norm
	template<typename EX>
    double nrm2(const VecExpr<EX>& x) {
        const auto& xx = x.self();
        const size_t n = xx.size();
        if (n == 0) return 0.0;
        using T = std::remove_cvref_t<decltype(xx(0))>;
		// Overflow-safe algorithm
        // Pass 1: find max absolute value
        double scale = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double a;
            if constexpr (detail::is_complex_v<T>)
                a = std::abs(xx(i));
            else
                a = std::abs(static_cast<double>(xx(i)));
            if (a > scale) scale = a;
        }
        if (scale == 0.0) return 0.0;
        // Pass 2: parallel sum of (x[i]/scale)^2
        double ssq = parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE,
            [&xx, scale](size_t i) -> double {
                double a;
                if constexpr (detail::is_complex_v<T>)
                    a = std::abs(xx(i)) / scale;
                else
                    a = std::abs(static_cast<double>(xx(i))) / scale;
                return a * a;
            });
        return scale * std::sqrt(ssq);
    };

	// Givens rotation parameters
	// Finds (c, s) such that  [[c, s], [-s, c] ]^T * [a, b] = [r, 0]
	// On exit: a <- r, b <- 0, c <- cos(theta), s <- sin(theta)
	template<typename T>
    void rotg(T& a, T& b, T& c, T& s) requires std::is_floating_point_v<T> {
        T r = std::hypot(a, b);
        if (r == T(0)) {
            c = T(1); s = T(0); a = T(0); b = T(0);
            return;
        };
        // Sign of r matches larger-magnitude component (minimises cancellation)
        if (std::abs(a) >= std::abs(b)) r = std::copysign(r, a);
        else r = std::copysign(r, b);
        c = a / r;
        s = b / r;
        a = r;
        b = T(0);
    };
 
	// Complex overload computes a unitary rotation where c is real and
    // s is complex such that: c^2 + abs(s)^2 = 1 by convention
    template<typename T>
    void rotg(std::complex<T>& a, std::complex<T>& b, T& c, std::complex<T>& s) {
        const T abs_a = std::abs(a);
        if (abs_a == T(0)) {
            c = T(0);
            s = std::complex<T>(1);
            a = b;
            b = std::complex<T>(0);
            return;
        };
        // Scale by abs_a to avoid intermediate overflow
        const T scale = abs_a + std::abs(b);
        const T an = std::abs(a / scale), bn = std::abs(b / scale);
        const T r_scaled = scale * std::sqrt(an * an + bn * bn);
        const std::complex<T> phase = a / abs_a;        // unit complex: a / abs(a)
        c = abs_a / r_scaled;
        s = phase * std::conj(b) / r_scaled;
        a = phase * r_scaled;
        b = std::complex<T>(0);
    };

	 template<typename T>
    void rot(Vector<T>& x, Vector<T>& y, T c, T s) requires std::is_floating_point_v<T> {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                T xi = x[i], yi = y[i];
                x[i] = c * xi + s * yi;
                y[i] = -s * xi + c * yi;
            };
        });
    };

    template<typename T>
    void rot(VectorView<T, true>& x, VectorView<T, true>& y, T c, T s) requires std::is_floating_point_v<T> {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                T xi = x(i), yi = y(i);
                x(i) = c * xi + s * yi;
                y(i) = -s * xi + c * yi;
            };
        });
    };

    template<typename T>
    void rot(Vector<T>& x, VectorView<T, true>& y, T c, T s) requires std::is_floating_point_v<T> {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                T xi = x[i], yi = y(i);
                x[i] = c * xi + s * yi;
                y(i) = -s * xi + c * yi;
            };
        });
    };
 
    template<typename T>
    void rot(VectorView<T, true>& x, Vector<T>& y, T c, T s) requires std::is_floating_point_v<T> { rot(y, x, c, -s); };
 
    // Complex rot uses complex s and real c
    // Applies: x[i] <- c * x[i] + s * y[i]
    //          y[i] <- -conj(s) * x[i] + c * y[i]
    template<typename T>
    void rot(Vector<std::complex<T>>& x, Vector<std::complex<T>>& y, T c, std::complex<T> s) {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s, conj(s)](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                std::complex<T> xi = x[i], yi = y[i];
                x[i] =  c * xi + s * yi;
                y[i] = -conj(s) * xi + c * yi;
            };
        });
    };
 
    template<typename T>
    void rot(VectorView<std::complex<T>, true>& x, VectorView<std::complex<T>, true>& y, T c, std::complex<T> s) {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s, conj(s)](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                std::complex<T> xi = x(i), yi = y(i);
                x(i) = c * xi + s * yi;
                y(i) = -conj(s) * xi + c * yi;
            };
        });
    };
};