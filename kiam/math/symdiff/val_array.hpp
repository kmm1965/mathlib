#pragma once

#include <array>
#include <numeric>
#include <functional>

#include "square_matrix.hpp"

_SYMDIFF_BEGIN

template<typename T, size_t N>
struct val_array : std::array<T, N>
{
	typedef std::array<T, N> super;
	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef T &reference;
	typedef const T &const_reference;

    constexpr val_array(){
		std::fill(super::begin(), super::end(), value_type());
	}

    constexpr val_array(const val_array &other){
		std::copy(other.cbegin(), other.cend(), super::begin());
	}

    constexpr val_array& operator=(const val_array &other){
		std::copy(other.cbegin(), other.cend(), super::begin());
		return *this;
	}

    constexpr reference operator[](size_t i){
		return super::operator[](i);
	}

    constexpr const_reference operator[](size_t i) const {
		return super::operator[](i);
	}

    constexpr val_array& operator+=(const val_array &other)
	{
		std::transform(super::cbegin(), super::cend(), other.cbegin(), super::begin(), std::plus<value_type>());
		return *this;
	}

    constexpr val_array& operator-=(const val_array &other)
	{
		std::transform(super::cbegin(), super::cend(), other.cbegin(), super::begin(), std::minus<value_type>());
		return *this;
	}

    constexpr val_array& operator*=(const value_type &x)
	{
        std::transform(super::cbegin(), super::cend(), super::begin(), [&x](const value_type& v) { return v * x; });
		return *this;
	}

    constexpr val_array& operator/=(const value_type &x)
	{
		std::transform(super::cbegin(), super::cend(), super::begin(), [&x](const value_type& v) { return v / x; });
		return *this;
	}

    constexpr value_type sqr() const
	{
		return std::accumulate(super::cbegin(), super::cend(), value_type(), [](value_type val, value_type value)
		{
			return val + value * value;
		});
	}

    constexpr value_type length() const {
		return std::sqrt(sqr());
	}
};

template<typename T, size_t N>
constexpr val_array<T, N> operator-(const val_array<T, N> &x)
{
	val_array<T, N> result;
	std::transform(x.cbegin(), x.cend(), result.begin(), [](T val){ return -val; });
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator+(const val_array<T, N> &x, const val_array<T, N> &y)
{
	val_array<T, N> result;
	std::transform(x.cbegin(), x.cend(), y.cbegin(), result.begin(), std::plus<T>());
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator-(const val_array<T, N> &x, const val_array<T, N> &y)
{
	val_array<T, N> result;
	std::transform(x.cbegin(), x.cend(), y.cbegin(), result.begin(), std::minus<T>());
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator*(const val_array<T, N> &x, const T &y)
{
	val_array<T, N> result;
	std::transform(x.cbegin(), x.cend(), result.begin(), [&y](const T& v) { return v * y; });
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator*(const T &x, const val_array<T, N> &y)
{
	val_array<T, N> result;
    std::transform(y.cbegin(), y.cend(), result.begin(), [&x](const T& yy) { return x * yy; });
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator/(const val_array<T, N> &x, const T &y)
{
    using namespace std::placeholders;
	val_array<T, N> result;
	std::transform(x.cbegin(), x.cend(), result.begin(), std::bind(std::divides<T>(), _1, y));
	return result;
}

template<typename T, size_t N>
constexpr val_array<T, N> operator*(const square_matrix<T, N> &A, const val_array<T, N> &x)
{
	val_array<T, N> result;
	for(size_t i = 0; i < N; ++i)
		result[i] = std::inner_product(&A(i, 0), &A(i, 0) + N, x.cbegin(), T());
	return result;
}

template<typename T, size_t N>
constexpr T max_abs(const val_array<T, N> &x)
{
	return std::accumulate(x.cbegin(), x.cend(), T(), [](const T &val, const T &v)
	{
		const T v1 = std::abs(v);
		return val > v1 ? val : v1;
	});
}

template<typename T, size_t N>
std::ostream& operator<<(std::ostream &o, const val_array<T, N> &x)
{
	o << "{ ";
	for(size_t i = 0; i < N; ++i)
		o << x[i] << ' ';
	return o << '}';
}

_SYMDIFF_END
