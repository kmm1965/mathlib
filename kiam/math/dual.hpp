#pragma once

#include "type_traits.hpp"
#include "kiam_math_func.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct dual
{
	typedef dual type;
	typedef T value_type;

	__DEVICE __HOST
	CONSTEXPR dual(const value_type &value = 0, const value_type &derivative = 0) : m_value(value), m_derivative(derivative){}

	__DEVICE __HOST
    CONSTEXPR dual(const dual &rhs) : m_value(rhs.m_value), m_derivative(rhs.m_derivative){}

	__DEVICE __HOST
    CONSTEXPR value_type value() const { return m_value; }

	__DEVICE __HOST
    CONSTEXPR value_type derivative() const { return m_derivative; }

	__DEVICE __HOST
    CONSTEXPR dual& operator=(value_type value)
	{
		m_value = value;
		m_derivative = 0;
		return *this;
	}

	__DEVICE __HOST
    CONSTEXPR dual& operator=(const dual &rhs)
	{
		m_value = rhs.m_value;
		m_derivative = rhs.m_derivative;
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
		m_derivative *= rhs;
	}

	__DEVICE __HOST
    CONSTEXPR void operator/=(const value_type &rhs)
	{
		m_value /= rhs;
		m_derivative /= rhs;
	}

	__DEVICE __HOST
    CONSTEXPR void operator+=(const dual &rhs)
	{
		m_value += rhs.m_value;
		m_derivative += rhs.m_derivative;
	}

	__DEVICE __HOST
    CONSTEXPR void operator-=(const dual &rhs)
	{
		m_value -= rhs.m_value;
		m_derivative -= rhs.m_derivative;
	}

	__DEVICE __HOST
    CONSTEXPR void operator*=(const dual &rhs)
	{
		m_derivative = m_derivative * rhs.m_value + m_value * rhs.m_derivative;
		m_value *= rhs.m_value;
	}

	__DEVICE __HOST
    CONSTEXPR void operator/=(const dual &rhs)
	{
		m_derivative = (m_derivative * rhs.m_value - m_value * rhs.m_derivative) / func::sqr(rhs.m_value);
		m_value /= rhs.m_value;
	}

private:
	value_type m_value, m_derivative;
};

template <class T> struct is_dual : std::false_type {};
template <class T> struct is_dual<const T> : is_dual<T> {};
template <class T> struct is_dual<volatile const T> : is_dual<T> {};
template <class T> struct is_dual<volatile T> : is_dual<T> {};
template <class T> struct is_dual<dual<T> > : std::true_type {};

template<typename T>
struct get_scalar_type<dual<T> >
{
    typedef T type;
};

template<typename T>
struct supports_multiplies<dual<T>, dual<T> > : std::true_type {};

template<typename T>
struct multiplies_result_type<dual<T>, dual<T> >
{
    typedef dual<T> type;
};

template<typename T>
struct supports_divides<dual<T>, dual<T> > : std::true_type {};

template<typename T>
struct divides_result_type<dual<T>, dual<T> >
{
    typedef dual<T> type;
};

template<typename T>
struct supports_divides<T, dual<T> > : std::true_type {};

template<typename T>
struct divides_result_type<T, dual<T> >
{
    typedef dual<T> type;
};

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator-(const dual<T> &a){
	return dual<T>(-a.value(), -a.derivative());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator+(const dual<T> &a, const dual<T> &b){
	return dual<T>(a.value() + b.value(), a.derivative() + b.derivative());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator-(const dual<T> &a, const dual<T> &b){
	return dual<T>(a.value() - b.value(), a.derivative() - b.derivative());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator*(const dual<T> &a, const dual<T> &b){
	return dual<T>(a.value() * b.value(), a.derivative() * b.value() + a.value() * b.derivative());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator/(const dual<T> &a, const dual<T> &b){
	return dual<T>(a.value() / b.value(), (a.derivative() * b.value() - a.value() * b.derivative()) / func::sqr(b.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator*(const dual<T> &a, const T &b){
	return dual<T>(a.value() * b, a.derivative() * b);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator*(const T &a, const dual<T> &b){
	return dual<T>(a * b.value(), a * b.derivative());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator/(const dual<T> &a, const T &b){
	return dual<T>(a.value() / b, a.derivative() / b);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> operator/(const T &a, const dual<T> &b){
	return dual<T>(a / b.value(), -a * b.derivative() / func::sqr(b.value()));
}

namespace func {

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> sin(dual<T> a){
	return dual<T>(func::sin(a.value()), a.derivative() * func::cos(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> cos(dual<T> a){
	return dual<T>(func::cos(a.value()), -a.derivative() * func::sin(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> tan(dual<T> a){
	return dual<T>(func::tan(a.value()), a.derivative() / func::sqr(func::cos(a.value())));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> asin(dual<T> a){
	return dual<T>(func::asin(a.value()), a.derivative() / func::sqrt(1 - func::sqr(a.value())));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> acos(dual<T> a){
	return dual<T>(func::acos(a.value()), -a.derivative() / func::sqrt(1 - func::sqr(a.value())));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> atan(dual<T> a){
	return dual<T>(func::atan(a.value()), a.derivative() / (1 - func::sqr(a.value())));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> sinh(dual<T> a){
	return dual<T>(func::sinh(a.value()), a.derivative() * func::cosh(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> cosh(dual<T> a){
	return dual<T>(func::cosh(a.value()), a.derivative() * func::sinh(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> tanh(dual<T> a){
	return dual<T>(func::tanh(a.value()), a.derivative() * 4 / func::sqr(func::cosh(a.value())));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> exp(dual<T> a)
{
	T value = func::exp(a.value());
	return dual<T>(value, a.derivative() * value);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> log(dual<T> a){
	return dual<T>(func::log(a.value()), a.derivative() / a.value());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> log10(dual<T> a){
	return dual<T>(func::log10(a.value()), a.derivative() / a.value() / M_LN10);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> sqrt(dual<T> a)
{
	T value = func::sqrt(a.value());
	return dual<T>(value, a.derivative() / value / 2);
}

#ifdef __CUDACC__

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> sinpi(dual<T> a){
	return dual<T>(func::sinpi(a.value()), a.derivative() * func::cospi(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> cospi(dual<T> a){
	return dual<T>(func::cospi(a.value()), -a.derivative() * func::sinpi(a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> exp2(dual<T> a)
{
	T value = func::exp2(a.value());
	return dual<T>(value, a.derivative() * value * (T) CUDART_LN2);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> log1p(dual<T> a){
	return dual<T>(func::log1p(a.value()), a.derivative() / (1 + a.value()));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR dual<T> log2(dual<T> a){
	return dual<T>(func::log2(a.value()), a.derivative() / (a.value() * (T) CUDART_LN2));
}

#endif	// __CUDACC__

}	// namespace func

_KIAM_MATH_END
