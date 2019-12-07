#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<int N>
size_t gcd(int a, int b)
{
	assert(a != 0 && b != 0);
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	if (a < b)
		math_swap(a, b);
	while (b != 0){
		math_swap(a, b);
		b %= a;
	}
	return a;
}

template<int N>
std::pair<int, int> euclid(int a0, int b0)
{
	int a = a0 < 0 ? -a0 : a0;
	int b = b0 < 0 ? -b0 : b0;
	assert(gcd<N>(a, b) == 1);
	if (gcd<N>(a, b) != 1)
		return std::make_pair(0, 0);
	bool swap = a < b;
	if (swap)
		math_swap(a, b);
	int e00 = 1, e01 = 0, e10 = 0, e11 = 1;
	int r;
	while ((r = a % b) != 0){
		int q = a / b;
		math_swap(e00, e01);
		math_swap(e10, e11);
		e01 -= e00 * q;
		e11 -= e10 * q;
		a = b;
		b = r;
	}
	a = e01;
	b = e11;
	if (swap)
		math_swap(a, b);
	if (a0 < 0) a = -a;
	if (b0 < 0) b = -b;
	return std::make_pair(a, b);
}

template<int N>
size_t inverse(int a, int n)
{
	assert(n > 0);
	a %= n;
	assert(a != 0);
	if (a < 0)
		a += n;
	assert(gcd<N>(a, n) == 1);
	int b = euclid<N>(a, n).first;
	if (b < 0)
		b += n;
	return b;
}

_KIAM_MATH_END
