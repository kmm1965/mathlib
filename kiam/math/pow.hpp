#pragma once

#include "math_def.h"

#ifndef __CUDACC__

#include <boost/math/special_functions/pow.hpp>

#endif	// __CUDACC__

_KIAM_MATH_BEGIN

#ifdef __CUDACC__

template<int N, typename T>
__HOST __DEVICE
typename std::enable_if<(N == 0), T>::type
math_pow(T x){ return 1; }

template<int N, typename T>
__HOST __DEVICE
typename std::enable_if<(N > 0) && (N % 2 == 1), T>::type
math_pow(T x);

template<int N, typename T>
__HOST __DEVICE
typename std::enable_if<(N > 0) && (N % 2 == 0), T>::type
math_pow(T x){ T p = math_pow<N / 2>(x); return p * p; }

template<int N, typename T>
__HOST __DEVICE
typename std::enable_if<(N > 0) && (N % 2 == 1), T>::type
math_pow(T x){ return math_pow<N - 1>(x) * x; }

template<int N, typename T>
__HOST __DEVICE
typename std::enable_if<(N < 0), T>::type
math_pow(T x){ return 1 / math_pow<-N>(x); }

#else	// __CUDACC__

template<int N, typename T>
T math_pow(const T &x){
	return boost::math::pow<N>(x);
}

#endif	// __CUDACC__

template<typename T>
__DEVICE __HOST
T pow_(T x, T y){
	return exp(log(x) * y);
}

template<typename T>
__DEVICE __HOST
T pown(T x, int n)
{
	if (n < 0)
		return 1 / pown(x, -n);
	else if (n == 0)
		return 1;
	else if(n == 1)
		return x;
	else if (n % 2 == 0){
		T xx = pown(x, n / 2);
		return xx * xx;
	} else // if(n % 2 == 1)
		return pown(x, n - 1) * x;
}

_KIAM_MATH_END
