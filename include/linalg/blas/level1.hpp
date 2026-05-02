#pragma once

#include <linalg/storage/matrix.hpp>
#include <linalg/storage/vector.hpp>

namespace linalg {
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
		parallel_for(m, 1, [&A, a, n](size_t rs, size_t re) {
			for (size_t i = rs; i < re; ++i) {
				for (size_t j = 0; j < n; ++j) A(i, j) *= a;
			}
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
		double max_val = std::abs(xx(0));

		for (size_t i = 1; i < n; ++i) {
			double val = std::abs(xx(i));
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
		double min_val = std::abs(xx(0));

		for (size_t i = 1; i < n; ++i) {
			double val = std::abs(xx(i));
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
};