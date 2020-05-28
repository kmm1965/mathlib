#pragma once

#include <cassert>

#include "type_traits.hpp"
#include "kiam_math_func.hpp"

_KIAM_MATH_BEGIN

template<class T>
struct vector2_value
{
    typedef vector2_value type;
    typedef T value_type;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    __DEVICE __HOST
    CONSTEXPR vector2_value() : m_value_x(value_type()), m_value_y(value_type()){}

    __DEVICE __HOST
    CONSTEXPR vector2_value(value_type x, value_type y) : m_value_x(x), m_value_y(y){}

    __DEVICE __HOST
    CONSTEXPR vector2_value(const vector2_value& rhs) : m_value_x(rhs.value_x()), m_value_y(rhs.value_y()){}

    __DEVICE __HOST
    vector2_value& operator=(const vector2_value &rhs)
    {
        m_value_x = rhs.value_x();
        m_value_y = rhs.value_y();
        return *this;
    }

    __DEVICE __HOST
    void set(const value_type &x, const value_type &y)
    {
        m_value_x = x;
        m_value_y = y;
    }

    __DEVICE __HOST
    CONSTEXPR const_reference value_x() const { return m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_reference value_y() const { return m_value_y; }

    __DEVICE __HOST
    reference value_x(){ return m_value_x; }

    __DEVICE __HOST
    reference value_y(){ return m_value_y; }

    __DEVICE __HOST
    void value_x(value_type const& value){ m_value_x = value; }

    __DEVICE __HOST
    void value_y(value_type const& value){ m_value_y = value; }

    __DEVICE __HOST
    CONSTEXPR unsigned size() const {
        return 2;
    }

    __DEVICE __HOST
    pointer data() { return &m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_pointer data() const { return &m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_reference operator[](unsigned i) const
    {
#ifndef NDEBUG
        assert(i < 2);
#endif
        return i == 0 ? m_value_x : m_value_y;
    }

    __DEVICE __HOST
    reference operator[](unsigned i)
    {
        assert(i < 2);
        return i == 0 ? m_value_x : m_value_y;
    }

    __DEVICE __HOST
    iterator begin() {
        return data();
    }

    __DEVICE __HOST
    iterator end() {
        return begin() + size();
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator begin() const {
        return data();
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator end() const {
        return begin() + size();
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator cbegin() const {
        return data();
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator cend() const {
        return cbegin() + size();
    }

    __DEVICE __HOST
    vector2_value& operator+=(const vector2_value& rhs)
    {
        m_value_x += rhs.value_x();
        m_value_y += rhs.value_y();
        return *this;
    }

    __DEVICE __HOST
    vector2_value& operator-=(const vector2_value& rhs)
    {
        m_value_x -= rhs.value_x();
        m_value_y -= rhs.value_y();
        return *this;
    }

    __DEVICE __HOST
    vector2_value& operator*=(value_type rhs)
    {
        m_value_x *= rhs;
        m_value_y *= rhs;
        return *this;
    }

    __DEVICE __HOST
    vector2_value& operator/=(value_type rhs)
    {
        m_value_x /= rhs;
        m_value_y /= rhs;
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR value_type length() const {
        return func::sqrt(m_value_x * m_value_x + m_value_y * m_value_y);
    }

    __DEVICE __HOST
    CONSTEXPR value_type sqr() const {
        return m_value_x * m_value_x + m_value_y * m_value_y;
    }

    __DEVICE __HOST
    CONSTEXPR vector2_value operator-() const {
        return vector2_value(-m_value_x, -m_value_y);
    }

    __DEVICE __HOST
    CONSTEXPR bool operator==(const vector2_value& rhs) const {
        return m_value_x == rhs.value_x() && m_value_y == rhs.value_y();
    }

    __DEVICE __HOST
    CONSTEXPR bool operator!=(const vector2_value& rhs) const {
        return m_value_x != rhs.value_x() || m_value_y != rhs.value_y();
    }

    __DEVICE __HOST
    vector2_value norm(){
        return *this *= 1 / length();
    }

private:
    value_type m_value_x, m_value_y;
};

template <class T> struct is_vector2_value : std::false_type {};
template <class T> struct is_vector2_value<const T> : is_vector2_value<T> {};
template <class T> struct is_vector2_value<volatile const T> : is_vector2_value<T> {};
template <class T> struct is_vector2_value<volatile T> : is_vector2_value<T> {};
template <class T> struct is_vector2_value<vector2_value<T> > : std::true_type {};

template<typename T>
struct get_scalar_type<vector2_value<T> >
{
    typedef T type;
};

template<typename T>
struct supports_scalar_product<vector2_value<T> > : std::true_type {};

template<typename T>
struct supports_component_product<vector2_value<T> > : std::true_type {};

template<class T>
__DEVICE __HOST
vector2_value<T> operator+(const vector2_value<T>& x, const vector2_value<T>& y){
    return vector2_value<T>(x.value_x() + y.value_x(), x.value_y() + y.value_y());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator-(const vector2_value<T>& x, const vector2_value<T>& y){
    return vector2_value<T>(x.value_x() - y.value_x(), x.value_y() - y.value_y());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator*(const vector2_value<T>& x, T y){
    return vector2_value<T>(x.value_x() * y, x.value_y() * y);
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator*(T x, const vector2_value<T>& y){
    return vector2_value<T>(x * y.value_x(), x * y.value_y());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator/(const vector2_value<T>& x, T y){
    return vector2_value<T>(x.value_x() / y, x.value_y() / y);
}

// Покомпонентное деление
template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator/(const vector2_value<T> &x, const vector2_value<T> &y){
    return vector2_value<T>(x.value_x() / y.value_x(), x.value_y() / y.value_y());
}

// Скалярное произведение
template<class T>
__DEVICE __HOST
CONSTEXPR T operator&(const vector2_value<T>& x, const vector2_value<T>& y){
    return x.value_x() * y.value_x() + x.value_y() * y.value_y();
}

// Покомпонентное произведение
template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> operator^(const vector2_value<T>& x, const vector2_value<T>& y){
    return vector2_value<T>(
        x.value_x() * y.value_x(),
        x.value_y() * y.value_y());
}

template<class T>
__DEVICE __HOST
CONSTEXPR T operator*(const vector2_value<T>& x, const vector2_value<T>& y){
    return x.value_x() * y.value_y() - x.value_y() * y.value_x();
}

// Площадь треугольника, построенного на векторах x и y
template<class T>
__DEVICE __HOST
CONSTEXPR T area_2D(const vector2_value<T>& x, const vector2_value<T>& y){
    return func::abs(x.value_x() * y.value_y() - x.value_y() * y.value_x()) / 2;
}

namespace func {

template<class T>
__DEVICE __HOST
CONSTEXPR T sqr(const vector2_value<T>& x){
    return x & x;
}

template<class T>
__DEVICE __HOST
CONSTEXPR T abs(const vector2_value<T>& x){
    return x.length();
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector2_value<T> vabs(const vector2_value<T>& x){
    return vector2_value<T>(math_abs(x.value_x()), math_abs(x.value_y()));
}

template<class T>
__DEVICE __HOST
CONSTEXPR T sum_abs(const vector2_value<T>& x){
    return math_abs(x.value_x()) + math_abs(x.value_y());
}

}   // namespace func

_KIAM_MATH_END
