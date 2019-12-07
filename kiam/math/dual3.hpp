#pragma once

#include "type_traits.hpp"
#include "kiam_math_func.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct dual3
{
	typedef dual3 type;
	typedef T value_type;

	__DEVICE __HOST
	CONSTEXPR dual3(const value_type &value = 0,
		const value_type &derivative_x = 0,
		const value_type &derivative_y = 0,
		const value_type &derivative_z = 0)
	: m_value(value), m_derivative_x(derivative_x), m_derivative_y(derivative_y), m_derivative_z(derivative_z){}

	__DEVICE __HOST
    CONSTEXPR dual3(const dual3 &rhs) : m_value(rhs.m_value),
		m_derivative_x(rhs.m_derivative_y), m_derivative_y(rhs.m_derivative_y), m_derivative_z(rhs.m_derivative_z){}

	__DEVICE __HOST
    CONSTEXPR value_type value() const { return m_value; }

	__DEVICE __HOST
    CONSTEXPR value_type derivative_x() const { return m_derivative_x; }

	__DEVICE __HOST
    CONSTEXPR value_type derivative_y() const { return m_derivative_y; }

	__DEVICE __HOST
    CONSTEXPR value_type derivative_z() const { return m_derivative_z; }

	__DEVICE __HOST
    CONSTEXPR dual3& operator=(value_type value)
	{
		m_value = value;
		m_derivative_x = m_derivative_y = m_derivative_z = 0;
		return *this;
	}

	__DEVICE __HOST
    CONSTEXPR dual3& operator=(const dual3 &rhs)
	{
		m_value = rhs.m_value;
		m_derivative_x = rhs.m_derivative_x;
		m_derivative_y = rhs.m_derivative_y;
		m_derivative_z = rhs.m_derivative_z;
		return *this;
	}

	__DEVICE __HOST
    CONSTEXPR void operator+=(const value_type &rhs){ m_value += rhs; }

	__DEVICE __HOST
    CONSTEXPR void operator-=(const value_type &rhs){ m_value -= rhs; }

	__DEVICE __HOST
    CONSTEXPR void operator*=(const value_type &rhs)
	{
		m_value *= rhs;
		m_derivative_x *= rhs;
		m_derivative_y *= rhs;
		m_derivative_z *= rhs;
	}

	__DEVICE __HOST
    CONSTEXPR void operator/=(const value_type &rhs)
	{
		m_value /= rhs;
		m_derivative_x /= rhs;
		m_derivative_y /= rhs;
		m_derivative_z /= rhs;
	}

	__DEVICE __HOST
    CONSTEXPR void operator+=(const dual3 &rhs)
	{
		m_value += rhs.m_value;
		m_derivative_x += rhs.m_derivative_x;
		m_derivative_y += rhs.m_derivative_y;
		m_derivative_z += rhs.m_derivative_z;
	}

	__DEVICE __HOST
    CONSTEXPR void operator-=(const dual3 &rhs)
	{
		m_value -= rhs.m_value;
		m_derivative_x -= rhs.m_derivative_x;
		m_derivative_y -= rhs.m_derivative_y;
		m_derivative_z -= rhs.m_derivative_z;
	}

	__DEVICE __HOST
    CONSTEXPR void operator*=(const dual3 &rhs)
	{
		m_derivative_x = m_derivative_x * rhs.m_value + m_value * rhs.m_derivative_x;
		m_derivative_y = m_derivative_y * rhs.m_value + m_value * rhs.m_derivative_y;
		m_derivative_z = m_derivative_z * rhs.m_value + m_value * rhs.m_derivative_z;
		m_value *= rhs.m_value;
	}

	__DEVICE __HOST
    CONSTEXPR void operator/=(const dual3 &rhs)
	{
		value_type pow_rhs_value = func::sqr(rhs.m_value);
		m_derivative_x = (m_derivative_x * rhs.m_value - m_value * rhs.m_derivative_x) / pow_rhs_value;
		m_derivative_y = (m_derivative_y * rhs.m_value - m_value * rhs.m_derivative_y) / pow_rhs_value;
		m_derivative_z = (m_derivative_z * rhs.m_value - m_value * rhs.m_derivative_z) / pow_rhs_value;
		m_value /= rhs.m_value;
	}

private:
	value_type m_value, m_derivative_x, m_derivative_y, m_derivative_z;
};

template <class T> struct is_dual3 : std::false_type {};
template <class T> struct is_dual3<const T> : is_dual3<T> {};
template <class T> struct is_dual3<volatile const T> : is_dual3<T> {};
template <class T> struct is_dual3<volatile T> : is_dual3<T> {};
template <class T> struct is_dual3<dual3<T> > : std::true_type {};

template<typename T>
struct get_scalar_type<dual3<T> >
{
    typedef T type;
};

template<typename T>
struct supports_multiplies<dual3<T>, dual3<T> > : std::true_type {};

template<typename T>
struct multiplies_result_type<dual3<T>, dual3<T> >
{
    typedef dual3<T> type;
};

template<typename T>
struct supports_divides<dual3<T>, dual3<T> > : std::true_type {};

template<typename T>
struct divides_result_type<dual3<T>, dual3<T> >
{
    typedef dual3<T> type;
};

template<typename T>
struct supports_divides<T, dual3<T> > : std::true_type {};

template<typename T>
struct divides_result_type<T, dual3<T> >
{
    typedef dual3<T> type;
};

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator-(const dual3<T> &a){
	return dual3<T>(-a.value(), -a.derivative_x(), -a.derivative_y(), -a.derivative_z());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator+(const dual3<T> &a, const dual3<T> &b){
	return dual3<T>(a.value() + b.value(),
		a.derivative_x() + b.derivative_x(),
		a.derivative_y() + b.derivative_y(),
		a.derivative_z() + b.derivative_z());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator-(const dual3<T> &a, const dual3<T> &b){
	return dual3<T>(a.value() - b.value(),
		a.derivative_x() - b.derivative_x(),
		a.derivative_y() - b.derivative_y(),
		a.derivative_z() - b.derivative_z());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator*(const dual3<T> &a, const dual3<T> &b){
	return dual3<T>(a.value() * b.value(),
		a.derivative_x() * b.value() + a.value() * b.derivative_x(),
		a.derivative_y() * b.value() + a.value() * b.derivative_y(),
		a.derivative_z() * b.value() + a.value() * b.derivative_z());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator/(const dual3<T> &a, const dual3<T> &b)
{
	T pow2_b = func::sqr(b.value());
	return dual3<T>(a.value() / b.value(),
		(a.derivative_x() * b.value() - a.value() * b.derivative_x()) / pow2_b,
		(a.derivative_y() * b.value() - a.value() * b.derivative_y()) / pow2_b,
		(a.derivative_z() * b.value() - a.value() * b.derivative_z()) / pow2_b);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator*(const dual3<T> &a, const T &b){
	return dual3<T>(a.value() * b, a.derivative_x() * b, a.derivative_y() * b, a.derivative_z() * b);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator*(const T &a, const dual3<T> &b){
	return dual3<T>(a * b.value(), a * b.derivative_x(), a * b.derivative_y(), a * b.derivative_z());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator/(const dual3<T> &a, const T &b){
	return dual3<T>(a.value() / b, a.derivative_x() / b, a.derivative_y() / b, a.derivative_z() / b);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> operator/(const T &a, const dual3<T> &b)
{
	T c = -a / func::sqr(b.value());
	return dual3<T>(a / b.value(), c * b.derivative_x(), c * b.derivative_y(), c * b.derivative_z());
}

namespace func {

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> sin(dual3<T> a)
{
	T cos_a = func::cos(a.value());
	return dual3<T>(func::sin(a.value()),
		a.derivative_x() * cos_a,
		a.derivative_y() * cos_a,
		a.derivative_z() * cos_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> cos(dual3<T> a)
{
	T sin_a = -func::sin(a.value());
	return dual3<T>(func::cos(a.value()),
		a.derivative_x() * sin_a,
		a.derivative_y() * sin_a,
		a.derivative_z() * sin_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> tan(dual3<T> a)
{
	T pow2_cos_a = func::sqr(func::cos(a.value()));
	return dual3<T>(func::tan(a.value()),
		a.derivative_x() / pow2_cos_a,
		a.derivative_y() / pow2_cos_a,
		a.derivative_z() / pow2_cos_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> asin(dual3<T> a)
{
	T sqrt_1_pow2_a = func::sqrt(1 - func::sqr(a.value()));
	return dual3<T>(func::asin(a.value()),
		a.derivative_x() / sqrt_1_pow2_a,
		a.derivative_y() / sqrt_1_pow2_a,
		a.derivative_z() / sqrt_1_pow2_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> acos(dual3<T> a)
{
	T sqrt_1_pow2_a = -func::sqrt(1 - func::sqr(a.value()));
	return dual3<T>(func::acos(a.value()),
		a.derivative_x() / sqrt_1_pow2_a,
		a.derivative_y() / sqrt_1_pow2_a,
		a.derivative_z() / sqrt_1_pow2_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> atan(dual3<T> a)
{
	T _1_pow2_a = (1 - func::sqr(a.value()));
	return dual3<T>(func::atan(a.value()),
		a.derivative_x() / _1_pow2_a,
		a.derivative_y() / _1_pow2_a,
		a.derivative_z() / _1_pow2_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> sinh(dual3<T> a)
{
	T cosh_a = func::cosh(a.value());
	return dual3<T>(func::sinh(a.value()),
		a.derivative_x() * cosh_a,
		a.derivative_y() * cosh_a,
		a.derivative_z() * cosh_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> cosh(dual3<T> a)
{
	T sinh_a = func::sinh(a.value());
	return dual3<T>(func::cosh(a.value()),
		a.derivative_x() * sinh_a,
		a.derivative_y() * sinh_a,
		a.derivative_z() * sinh_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> tanh(dual3<T> a)
{
	T _4_pow2_cosh_a = 4 / func::sqr(func::cosh(a.value()));
	return dual3<T>(func::tanh(a.value()),
		a.derivative_x() * _4_pow2_cosh_a,
		a.derivative_y() * _4_pow2_cosh_a,
		a.derivative_z() * _4_pow2_cosh_a);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> exp(dual3<T> a)
{
	T value = func::exp(a.value());
	return dual3<T>(value,
		a.derivative_x() * value,
		a.derivative_y() * value,
		a.derivative_z() * value);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> log(dual3<T> a){
	return dual3<T>(func::log(a.value()),
		a.derivative_x() / a.value(),
		a.derivative_y() / a.value(),
		a.derivative_z() / a.value());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> log10(dual3<T> a)
{
	T a_ln10 = a.value() * M_LN10;
	return dual3<T>(func::log10(a.value()),
		a.derivative_x() / a_ln10,
		a.derivative_y() / a_ln10,
		a.derivative_z() / a_ln10);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> sqrt(dual3<T> a)
{
	T value = func::sqrt(a.value());
	T value_2 = value * 2;
	return dual3<T>(value,
		a.derivative_x() / value_2,
		a.derivative_y() / value_2,
		a.derivative_z() / value_2);
}

#ifdef __CUDACC__

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> sinpi(dual3<T> a)
{
	T cospi_value = -func::sinpi(a.value());
	return dual3<T>(func::sinpi(a.value()),
		a.derivative_x() * cospi_value,
		a.derivative_y() * cospi_value,
		a.derivative_z() * cospi_value);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> cospi(dual3<T> a)
{
	T sinpi_value = -func::sinpi(a.value());
	return dual3<T>(func::cospi(a.value()),
		a.derivative_x() * sinpi_value,
		a.derivative_y() * sinpi_value,
		a.derivative_z() * sinpi_value);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> exp2(dual3<T> a)
{
	T value = func::exp2(a.value());
	T value2 = value * CUDART_LN2;
	return dual3<T>(value,
		a.derivative_x() * value2,
		a.derivative_y() * value2,
		a.derivative_z() * value2);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> log1p(dual3<T> a)
{
	T value2 = 1 / (1 + a.value());
	return dual3<T>(func::log1p(a.value()),
		a.derivative_x() * value2,
		a.derivative_y() * value2,
		a.derivative_z() * value2);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual3<T> log2(dual3<T> a)
{
	T value2 = 1 / (a.value() * (T) CUDART_LN2);
	return dual3<T>(func::log2(a.value()),
		a.derivative_x() * value2,
		a.derivative_y() * value2,
		a.derivative_z() * value2);
}

#endif	// __CUDACC__

}	// namespace func

_KIAM_MATH_END
