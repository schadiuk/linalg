#pragma once

#include <linalg/storage/matrix.hpp>
#include <linalg/storage/vector.hpp>

namespace linalg {
	namespace detail {
		// Type trait for complex scalars
		template<typename T> struct is_complex_impl : std::false_type {};
        template<typename T> struct is_complex_impl<std::complex<T>> : std::true_type {};
        template<typename T> inline constexpr bool is_complex_v = is_complex_impl<std::remove_cvref_t<T>>::value;

		template<typename T>
		LINALG_INLINE double abs_as_double(const T& v) noexcept {
			if constexpr (is_complex_v<T>) return std::abs(v);
    		else return std::abs(static_cast<double>(v));
		};

		// Dense data accessor
		template<typename T, typename E>
		LINALG_INLINE const T* dense_data(const E& e) noexcept {
			if constexpr (std::is_same_v<E, Vector<T>>) return e.data();
			else if constexpr (std::is_same_v<E, VecRef<T>>) return e.vec.data();
			else if constexpr (std::is_same_v<E, VectorView<T,false>> || std::is_same_v<E, VectorView<T,true>>)
				return (e.stride() == 1) ? e.data() : nullptr;
			else if constexpr (std::is_same_v<E, VecViewRef<T,false>> || std::is_same_v<E, VecViewRef<T,true>>)
				return (e.view.stride() == 1) ? e.view.data() : nullptr;
			else return nullptr;
		};

		template<typename T>
		LINALG_INLINE void axpy_chunk(T alpha, const T* LINALG_RESTRICT x,	T* LINALG_RESTRICT y, size_t s, size_t e) noexcept {
			LINALG_VECTORIZE
			for (size_t i = s; i < e; ++i) y[i] += alpha * x[i];
		};
		
		template<typename T>
		LINALG_INLINE void axpby_chunk(T alpha, const T* LINALG_RESTRICT x,	T beta,  T* LINALG_RESTRICT y, size_t s, size_t e) noexcept {
			LINALG_VECTORIZE
			for (size_t i = s; i < e; ++i) y[i] = alpha * x[i] + beta * y[i];
		};
		
		// x[i] *= alpha  (element-wise scale, no accumulation needed)
		template<typename T>
		LINALG_INLINE void scal_chunk(T alpha, T* LINALG_RESTRICT x, size_t s, size_t e) noexcept {
			LINALG_UNROLL(4)
			LINALG_VECTORIZE
			for (size_t i = s; i < e; ++i) x[i] *= alpha;
		};

		template<typename T>
		LINALG_INLINE T dot_chunk(const T* LINALG_RESTRICT x, const T* LINALG_RESTRICT y, size_t s, size_t e) noexcept {
			T s0{}, s1{}, s2{}, s3{};
			const size_t n4 = s + ((e - s) / 4) * 4;
			LINALG_VECTORIZE
			for (size_t i = s; i < n4; i += 4) {
				s0 += x[i] * y[i];
				s1 += x[i+1] * y[i+1];
				s2 += x[i+2] * y[i+2];
				s3 += x[i+3] * y[i+3];
			};
			T acc = (s0 + s1) + (s2 + s3); // pairwise sum reduces rounding error?
			for (size_t i = n4; i < e; ++i) acc += x[i] * y[i];
			return acc;
		};
		
		template<typename T>
		LINALG_INLINE T dotc_chunk(const T* LINALG_RESTRICT x, const T* LINALG_RESTRICT y,	size_t s, size_t e) noexcept {
			T s0{}, s1{}, s2{}, s3{};
			const size_t n4 = s + ((e - s) / 4) * 4;
			LINALG_VECTORIZE
			for (size_t i = s; i < n4; i += 4) {
				s0 += conj(x[i]) * y[i];
				s1 += conj(x[i+1]) * y[i+1];
				s2 += conj(x[i+2]) * y[i+2];
				s3 += conj(x[i+3]) * y[i+3];
			};
			T acc = (s0 + s1) + (s2 + s3);
			for (size_t i = n4; i < e; ++i) acc += conj(x[i]) * y[i];
			return acc;
		};

		template<typename T>
		LINALG_INLINE double nrm2_ssq_chunk(const T* LINALG_RESTRICT x,	size_t s, size_t e, double scale) noexcept {
			double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
			const size_t n4 = s + ((e - s) / 4) * 4;
			LINALG_VECTORIZE
			for (size_t i = s; i < n4; i += 4) {
				const double a0 = abs_as_double(x[i])   / scale;
				const double a1 = abs_as_double(x[i+1]) / scale;
				const double a2 = abs_as_double(x[i+2]) / scale;
				const double a3 = abs_as_double(x[i+3]) / scale;
				s0 += a0 * a0; s1 += a1 * a1;
				s2 += a2 * a2; s3 += a3 * a3;
			};
			double ssq = (s0+s1) + (s2+s3);
			for (size_t i = n4; i < e; ++i) {
				const double a = abs_as_double(x[i]) / scale;
				ssq += a * a;
			};
			return ssq;
		};

		template<typename T>
		LINALG_INLINE double asum_chunk(const T* LINALG_RESTRICT x,	size_t s, size_t e) noexcept {
			double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
			const size_t n4 = s + ((e - s) / 4) * 4;
			LINALG_VECTORIZE
			for (size_t i = s; i < n4; i += 4) {
				if constexpr (is_complex_v<T>) {
					s0 += std::abs(std::real(x[i]))   + std::abs(std::imag(x[i]));
					s1 += std::abs(std::real(x[i+1])) + std::abs(std::imag(x[i+1]));
					s2 += std::abs(std::real(x[i+2])) + std::abs(std::imag(x[i+2]));
					s3 += std::abs(std::real(x[i+3])) + std::abs(std::imag(x[i+3]));
				} else {
					s0 += std::abs(static_cast<double>(x[i]));
					s1 += std::abs(static_cast<double>(x[i+1]));
					s2 += std::abs(static_cast<double>(x[i+2]));
					s3 += std::abs(static_cast<double>(x[i+3]));
				};
			};
			double acc = (s0+s1) + (s2+s3);
			for (size_t i = n4; i < e; ++i) {
				if constexpr (is_complex_v<T>)
					acc += std::abs(std::real(x[i])) + std::abs(std::imag(x[i]));
				else
					acc += std::abs(static_cast<double>(x[i]));
			};
			return acc;
		};
	};

	// y += alpha * x
	template<typename Alpha, typename EX, typename T>
	LINALG_INLINE void axpy(Alpha alpha, const VecExpr<EX>& x, Vector<T>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const size_t n = y.size();
		if (const T* xr = detail::dense_data<T>(xx)) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			T* LINALG_RESTRICT ya = detail::assume_aligned<64>(y.data());
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xa, ya, a](size_t s, size_t e) {
				detail::axpy_chunk(a, xa, ya, s, e);
			});
		} else {
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a](size_t s, size_t e) {
			LINALG_VECTORIZE
			for (size_t i = s; i < e; ++i)
				y[i] += a * static_cast<T>(xx(i));
			});
		};
	};
	
	template<typename Alpha, typename EX, typename T>
	LINALG_INLINE void axpy(Alpha alpha, const VecExpr<EX>& x, VectorView<T, true>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const size_t n = y.size();
		if (y.stride() == 1) {
				if (const T* xr = detail::dense_data<T>(xx)) {
					// Unit-stride view output: no alignment guarantee on y.data(), but RESTRICT still removes aliasing fences.
					const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
					T* LINALG_RESTRICT ya = y.data();
					parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xa, ya, a](size_t s, size_t e) {
						detail::axpy_chunk(a, xa, ya, s, e);
					});
					return;
				};
				parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a](size_t s, size_t e) {
					LINALG_VECTORIZE
					for (size_t i = s; i < e; ++i)	y(i) += a * static_cast<T>(xx(i));
				});
				return;
		};
		// Strided output: indexing via operator()(i) * stride; no vectorisation.
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) y(i) += a * static_cast<T>(xx(i));
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
		if (const T* xr = detail::dense_data<T>(xx)) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			T* LINALG_RESTRICT ya = detail::assume_aligned<64>(y.data());
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xa, ya, a, b](size_t s, size_t e) {
				detail::axpby_chunk(a, xa, b, ya, s, e);
			});
		} else {
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a, b](size_t s, size_t e) {
				LINALG_VECTORIZE
				for (size_t i = s; i < e; ++i)	y[i] = a * static_cast<T>(xx(i)) + b * y[i];
			});
		};
	};
	
	template<typename Alpha, typename EX, typename Beta, typename T>
	void axpby(Alpha alpha, const VecExpr<EX>& x, Beta beta, VectorView<T, true>& y) {
		const auto& xx = x.self();
		BOUNDS_CHECK(xx.size() == y.size());
		const T a = static_cast<T>(alpha);
		const T b = static_cast<T>(beta);
		const size_t n = y.size();
		if (y.stride() == 1) {
			if (const T* xr = detail::dense_data<T>(xx)) {
				const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
				T* LINALG_RESTRICT ya = y.data();
				parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xa, ya, a, b](size_t s, size_t e) {
					detail::axpby_chunk(a, xa, b, ya, s, e);
				});
				return;
			};
		};
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&xx, &y, a, b](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i)
				y(i) = a * static_cast<T>(xx(i)) + b * y(i);
		});
	};

	// In-place scaling
	template<typename Alpha, typename T>
	LINALG_INLINE void scal(Alpha alpha, Vector<T>& x) {
		const T a = static_cast<T>(alpha);
		T* LINALG_RESTRICT xp = detail::assume_aligned<64>(x.data());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xp, a](size_t s, size_t e) {
			detail::scal_chunk(a, xp, s, e);
		});
	};
	
	template<typename Alpha, typename T>
	LINALG_INLINE void scal(Alpha alpha, VectorView<T, true>& x) {
		const T a = static_cast<T>(alpha);
		const size_t n = x.size();
		if (x.stride() == 1) {
			T* LINALG_RESTRICT xp = x.data();
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xp, a](size_t s, size_t e) {
				detail::scal_chunk(a, xp, s, e);
			});
		} else {
			parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, a](size_t s, size_t e) {
				for (size_t i = s; i < e; ++i) x(i) *= a;
			});
		};
	};
	
	template<typename Alpha, typename T, Layout L>
	LINALG_INLINE void scal(Alpha alpha, Matrix<T, L>& A) {
		const T a = static_cast<T>(alpha);
		T* LINALG_RESTRICT dp = detail::assume_aligned<64>(A.data());
		const size_t total = A.rows() * A.cols();
		parallel_for(total, PARALLEL_THRESHOLD_SIMPLE, [dp, a](size_t s, size_t e) {
			detail::scal_chunk(a, dp, s, e);
		});
	};
	
	template<typename Alpha, typename T, Layout L, bool Trans, bool Conj>
	LINALG_INLINE void scal(Alpha alpha, MatrixView<T, L, Trans, Conj, true>& A) {
		const T a = static_cast<T>(alpha);
		const size_t m = A.rows(), nc = A.cols();
		parallel_for(m, PARALLEL_THRESHOLD_SIMPLE, [&A, a, nc](size_t rs, size_t re) {
			for (size_t i = rs; i < re; ++i) {
				LINALG_VECTORIZE
				for (size_t j = 0; j < nc; ++j) A(i, j) *= a;
			};
		});
	};

	// Delegation to assignment operators (supports mutable views)
	template<typename EX, typename T>
	LINALG_INLINE void copy(const VecExpr<EX>& x, Vector<T>& y) {
		BOUNDS_CHECK(x.self().size() == y.size());
		y = x;
	};
	
	template<typename EX, typename T>
	LINALG_INLINE void copy(const VecExpr<EX>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.self().size() == y.size());
		y = x;
	};
	
	template<typename EM, typename T, Layout L>
	LINALG_INLINE void copy(const MatExpr<EM>& A, Matrix<T, L>& B) {
		BOUNDS_CHECK(A.self().rows() == B.rows() && A.self().cols() == B.cols());
		B = A;
	};
	
	template<typename EM, typename T, Layout L, bool Trans, bool Conj>
	LINALG_INLINE void copy(const MatExpr<EM>& A, MatrixView<T, L, Trans, Conj, true>& B) {
		BOUNDS_CHECK(A.self().rows() == B.rows() && A.self().cols() == B.cols());
		B = A;
	};

	// Elementwise swapping with overloads for all combinations of Vector/VectorView
	template<typename T>
	LINALG_INLINE void swap(Vector<T>& x, Vector<T>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) std::swap(x[i], y[i]);
		});
	};
	
	template<typename T>
	LINALG_INLINE void swap(VectorView<T, true>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) { T tmp = x(i); x(i) = y(i); y(i) = tmp; };
		});
	};
	
	template<typename T>
	LINALG_INLINE void swap(Vector<T>& x, VectorView<T, true>& y) {
		BOUNDS_CHECK(x.size() == y.size());
		const size_t n = x.size();
		parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y](size_t s, size_t e) {
			for (size_t i = s; i < e; ++i) { T tmp = x[i]; x[i] = y(i); y(i) = tmp; };
		});
	};
	
	template<typename T>
	LINALG_INLINE void swap(VectorView<T, true>& x, Vector<T>& y) { swap(y, x); };

	// Index of maximum absolute value
	template<typename EX>
	LINALG_INLINE size_t iamax(const VecExpr<EX>& x) {
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
	LINALG_INLINE size_t iamin(const VecExpr<EX>& x) {
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
	LINALG_INLINE double asum(const VecExpr<EX>& x) {
		const auto& xx = x.self();
		const size_t n = xx.size();
		if (n == 0) return 0.0;
		using T = std::remove_cvref_t<decltype(xx(0))>;
		if (const T* xr = detail::dense_data<T>(xx)) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			return parallel_reduce_chunks<double>(n, PARALLEL_THRESHOLD_REDUCE,
				[xa](size_t s, size_t e) {
					return detail::asum_chunk(xa, s, e);
				});
		};
		return parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE,
			[&xx](size_t i) -> double {
				if constexpr (detail::is_complex_v<T>) return std::abs(std::real(xx(i))) + std::abs(std::imag(xx(i)));
				else return std::abs(static_cast<double>(xx(i)));
			});
	};

	// Dot product (naive)
	template<typename EX, typename EY>
	LINALG_INLINE auto dot(const VecExpr<EX>& x, const VecExpr<EY>& y) {
		const auto& xx = x.self();
		const auto& yy = y.self();
		BOUNDS_CHECK(xx.size() == yy.size());
		const size_t n = xx.size();
		using T = std::remove_cvref_t<decltype(xx(0) * yy(0))>;
		const T* xr = detail::dense_data<T>(xx);
		const T* yr = detail::dense_data<T>(yy);
		if (xr && yr) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			const T* LINALG_RESTRICT ya = detail::assume_aligned<64>(yr);
			return parallel_reduce_chunks<T>(n, PARALLEL_THRESHOLD_REDUCE,
				[xa, ya](size_t s, size_t e) {
					return detail::dot_chunk(xa, ya, s, e);
				});
		};
		return parallel_reduce<T>(xx.size(), PARALLEL_THRESHOLD_REDUCE,
			[&xx, &yy](size_t i) { return xx(i) * yy(i); });
	};

	// Complex dot (taking conjugate of the first argument)
	template<typename EX, typename EY>
	LINALG_INLINE auto dotc(const VecExpr<EX>& x, const VecExpr<EY>& y) {
		const auto& xx = x.self();
		const auto& yy = y.self();
		BOUNDS_CHECK(xx.size() == yy.size());
		const size_t n = xx.size();
		// BLAS zdotc: conj(x) * y
		using T = std::remove_cvref_t<decltype(conj(xx(0)) * yy(0))>;
		const T* xr = detail::dense_data<T>(xx);
		const T* yr = detail::dense_data<T>(yy);
		if (xr && yr) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			const T* LINALG_RESTRICT ya = detail::assume_aligned<64>(yr);
			return parallel_reduce_chunks<T>(n, PARALLEL_THRESHOLD_REDUCE,
				[xa, ya](size_t s, size_t e) {
					return detail::dotc_chunk(xa, ya, s, e);
				});
		};
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
		const T* xr = detail::dense_data<T>(xx);
    	// Pass 1: scale = max(abs(x[i])).
		double scale = 0.0;
		if (xr) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			LINALG_VECTORIZE
			for (size_t i = 0; i < n; ++i) {
				const double a = detail::abs_as_double(xa[i]);
				if (a > scale) scale = a;
			};
		} else {
			for (size_t i = 0; i < n; ++i) {
				const double a = detail::abs_as_double(xx(i));
				if (a > scale) scale = a;
			};
		}
		if (scale == 0.0) return 0.0;
		// Pass 2: parallel  SUM(|x[i]|/scale)^2.
		double ssq;
		if (xr) {
			const T* LINALG_RESTRICT xa = detail::assume_aligned<64>(xr);
			ssq = parallel_reduce_chunks<double>(n, PARALLEL_THRESHOLD_REDUCE,
				[xa, scale](size_t s, size_t e) {
					return detail::nrm2_ssq_chunk(xa, s, e, scale);
				});
		} else {
			ssq = parallel_reduce<double>(n, PARALLEL_THRESHOLD_REDUCE,
				[&xx, scale](size_t i) -> double {
					const double a = detail::abs_as_double(xx(i)) / scale;
					return a * a;
				});
		};
		return scale * std::sqrt(ssq);
    };

	// Givens rotation parameters
	// Finds (c, s) such that  [[c, s], [-s, c] ]^T * [a, b] = [r, 0]
	// On exit: a <- r, b <- 0, c <- cos(theta), s <- sin(theta)
	template<typename T>
    LINALG_INLINE void rotg(T& a, T& b, T& c, T& s) requires std::is_floating_point_v<T> {
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
    LINALG_INLINE void rotg(std::complex<T>& a, std::complex<T>& b, T& c, std::complex<T>& s) {
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
    LINALG_INLINE void rot(Vector<T>& x, Vector<T>& y, T c, T s) requires std::is_floating_point_v<T> {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
		T* LINALG_RESTRICT xp = detail::assume_aligned<64>(x.data());
    	T* LINALG_RESTRICT yp = detail::assume_aligned<64>(y.data());
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [xp, yp, c, s](size_t rs, size_t re) {
			LINALG_VECTORIZE
            for (size_t i = rs; i < re; ++i) {
                T xi = xp[i], yi = yp[i];
                xp[i] = c * xi + s * yi;
                yp[i] = -s * xi + c * yi;
            };
        });
    };

    template<typename T>
    LINALG_INLINE void rot(VectorView<T, true>& x, VectorView<T, true>& y, T c, T s) requires std::is_floating_point_v<T> {
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
    LINALG_INLINE void rot(Vector<T>& x, VectorView<T, true>& y, T c, T s) requires std::is_floating_point_v<T> {
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
    LINALG_INLINE void rot(VectorView<T, true>& x, Vector<T>& y, T c, T s) requires std::is_floating_point_v<T> { rot(y, x, c, -s); };
 
    // Complex rot uses complex s and real c
    // Applies: x[i] <- c * x[i] + s * y[i]
    //          y[i] <- -conj(s) * x[i] + c * y[i]
    template<typename T>
    LINALG_INLINE void rot(Vector<std::complex<T>>& x, Vector<std::complex<T>>& y, T c, std::complex<T> s) {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s, cs = conj(s)](size_t rs, size_t re) {
			LINALG_VECTORIZE
            for (size_t i = rs; i < re; ++i) {
                std::complex<T> xi = x[i], yi = y[i];
                x[i] =  c * xi + s * yi;
                y[i] = -cs * xi + c * yi;
            };
        });
    };
 
    template<typename T>
    LINALG_INLINE void rot(VectorView<std::complex<T>, true>& x, VectorView<std::complex<T>, true>& y, T c, std::complex<T> s) {
        BOUNDS_CHECK(x.size() == y.size());
        const size_t n = x.size();
        parallel_for(n, PARALLEL_THRESHOLD_SIMPLE, [&x, &y, c, s, cs = conj(s)](size_t rs, size_t re) {
            for (size_t i = rs; i < re; ++i) {
                std::complex<T> xi = x(i), yi = y(i);
                x(i) = c * xi + s * yi;
                y(i) = -cs * xi + c * yi;
            };
        });
    };
};