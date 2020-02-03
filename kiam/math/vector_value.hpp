#pragma once

#include <cassert>

#include "type_traits.hpp"
#include "kiam_math_func.hpp"

_KIAM_MATH_BEGIN

template<class T>
struct vector_value
{
    typedef vector_value type;
    typedef T value_type;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    __DEVICE __HOST
    CONSTEXPR vector_value() : m_value_x(value_type()), m_value_y(value_type()), m_value_z(value_type()){}

    __DEVICE __HOST
    CONSTEXPR vector_value(const value_type &x, const value_type &y, const value_type &z) : m_value_x(x), m_value_y(y), m_value_z(z){}

    __DEVICE __HOST
    CONSTEXPR vector_value(const vector_value &rhs) : m_value_x(rhs.value_x()), m_value_y(rhs.value_y()), m_value_z(rhs.value_z()){}

    __DEVICE __HOST
    vector_value& operator=(const vector_value &rhs)
    {
        m_value_x = rhs.value_x();
        m_value_y = rhs.value_y();
        m_value_z = rhs.value_z();
        return *this;
    }

    __DEVICE __HOST
    void set(const value_type &x, const value_type &y, const value_type &z)
    {
        m_value_x = x;
        m_value_y = y;
        m_value_z = z;
    }

    __DEVICE __HOST
    CONSTEXPR const_reference value_x() const { return m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_reference value_y() const { return m_value_y; }

    __DEVICE __HOST
    CONSTEXPR const_reference value_z() const { return m_value_z; }

    __DEVICE __HOST
    reference value_x(){ return m_value_x; }

    __DEVICE __HOST
    reference value_y(){ return m_value_y; }

    __DEVICE __HOST
    reference value_z(){ return m_value_z; }

    __DEVICE __HOST
    void value_x(const value_type &value){ m_value_x = value; }

    __DEVICE __HOST
    void value_y(const value_type &value){ m_value_y = value; }

    __DEVICE __HOST
    void value_z(const value_type &value){ m_value_z = value; }

    __DEVICE __HOST
    CONSTEXPR unsigned size() const {
        return 3;
    }

    __DEVICE __HOST
    pointer data() { return &m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_pointer data() const { return &m_value_x; }

    __DEVICE __HOST
    CONSTEXPR const_reference operator[](unsigned i) const
    {
#ifndef NDEBUG
        assert(i < 3);
#endif
        return i == 0 ? m_value_x : i == 1 ? m_value_y : m_value_z;
    }

    __DEVICE __HOST
    reference operator[](unsigned i)
    {
        assert(i < 3);
        return i == 0 ? m_value_x : i == 1 ? m_value_y : m_value_z;
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
    CONSTEXPR value_type value_xy() const { return value_type(m_value_x * m_value_y); }

    __DEVICE __HOST
    CONSTEXPR value_type value_xz() const { return value_type(m_value_x * m_value_z); }

    __DEVICE __HOST
    CONSTEXPR value_type value_yz() const { return value_type(m_value_y * m_value_z); }

    __DEVICE __HOST
    vector_value& operator+=(const vector_value &rhs)
    {
        m_value_x += rhs.value_x();
        m_value_y += rhs.value_y();
        m_value_z += rhs.value_z();
        return *this;
    }

    __DEVICE __HOST
    vector_value& operator-=(const vector_value &rhs)
    {
        m_value_x -= rhs.value_x();
        m_value_y -= rhs.value_y();
        m_value_z -= rhs.value_z();
        return *this;
    }

    __DEVICE __HOST
    vector_value& operator*=(const value_type &rhs)
    {
        m_value_x *= rhs;
        m_value_y *= rhs;
        m_value_z *= rhs;
        return *this;
    }

    __DEVICE __HOST
    vector_value& operator/=(const value_type &rhs)
    {
        m_value_x /= rhs;
        m_value_y /= rhs;
        m_value_z /= rhs;
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR value_type length() const {
        return func::sqrt(sqr());
    }

    __DEVICE __HOST
    CONSTEXPR value_type sqr() const {
        return m_value_x * m_value_x + m_value_y * m_value_y + m_value_z * m_value_z;
    }

    __DEVICE __HOST
    CONSTEXPR vector_value operator-() const {
        return vector_value(-m_value_x, -m_value_y, -m_value_z);
    }

    __DEVICE __HOST
    CONSTEXPR bool operator==(const vector_value &rhs) const {
        return m_value_x == rhs.value_x() && m_value_y == rhs.value_y() && m_value_z == rhs.value_z();
    }

    __DEVICE __HOST
    CONSTEXPR bool operator!=(const vector_value &rhs) const {
        return m_value_x != rhs.value_x() || m_value_y != rhs.value_y() || m_value_z != rhs.value_z();
    }

    __DEVICE __HOST
    vector_value norm(){
        return *this *= 1 / length();
    }

private:
    value_type m_value_x, m_value_y, m_value_z;
};

template <class T> struct is_vector_value : std::false_type {};
template <class T> struct is_vector_value<const T> : is_vector_value<T> {};
template <class T> struct is_vector_value<volatile const T> : is_vector_value<T> {};
template <class T> struct is_vector_value<volatile T> : is_vector_value<T> {};
template <class T> struct is_vector_value<vector_value<T> > : std::true_type {};

template<typename T>
struct get_scalar_type<vector_value<T> >
{
    typedef T type;
};

template<typename T>
struct supports_multiplies<vector_value<T>, vector_value<T> > : std::true_type {};

template<typename T>
struct multiplies_result_type<vector_value<T>, vector_value<T> >
{
    typedef vector_value<T> type;
};

template<typename T>
struct supports_scalar_product<vector_value<T> > : std::true_type {};

template<typename T>
struct supports_component_product<vector_value<T> > : std::true_type {};

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator+(const vector_value<T> &x, const vector_value<T> &y){
    return vector_value<T>(x.value_x() + y.value_x(), x.value_y() + y.value_y(), x.value_z() + y.value_z());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator-(const vector_value<T> &x, const vector_value<T> &y){
    return vector_value<T>(x.value_x() - y.value_x(), x.value_y() - y.value_y(), x.value_z() - y.value_z());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator*(const vector_value<T> &x, T y){
    return vector_value<T>(x.value_x() * y, x.value_y() * y, x.value_z() * y);
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator*(T x, const vector_value<T> &y){
    return vector_value<T>(x * y.value_x(), x * y.value_y(), x * y.value_z());
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator/(const vector_value<T> &x, T y){
    return vector_value<T>(x.value_x() / y, x.value_y() / y, x.value_z() / y);
}

// Покомпонентное деление
template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator/(const vector_value<T> &x, const vector_value<T> &y){
    return vector_value<T>(x.value_x() / y.value_x(), x.value_y() / y.value_y(), x.value_z() / y.value_z());
}

// Векторное произведение
template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator*(const vector_value<T> &x, const vector_value<T> &y)
{
    return vector_value<T>(
        x.value_y() * y.value_z() - x.value_z() * y.value_y(),
        x.value_z() * y.value_x() - x.value_x() * y.value_z(),
        x.value_x() * y.value_y() - x.value_y() * y.value_x());
}

// Скалярное произведение
template<class T>
__DEVICE __HOST
CONSTEXPR T operator&(const vector_value<T> &x, const vector_value<T> &y){
    return x.value_x() * y.value_x() + x.value_y() * y.value_y() + x.value_z() * y.value_z();
}

// Покомпонентное произведение
template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> operator^(const vector_value<T> &x, const vector_value<T> &y){
    return vector_value<T>(
        x.value_x() * y.value_x(),
        x.value_y() * y.value_y(),
        x.value_z() * y.value_z());
}

namespace func {

template<class T>
__DEVICE __HOST
CONSTEXPR T sqr(const vector_value<T> &x){
    return x & x;
}

template<class T>
__DEVICE __HOST
CONSTEXPR T abs(const vector_value<T> &x){
    return x.length();
}

template<class T>
__DEVICE __HOST
CONSTEXPR vector_value<T> vabs(const vector_value<T> &x){
    return vector_value<T>(func::abs(x.value_x()), func::abs(x.value_y()), func::abs(x.value_z()));
}

template<class T>
__DEVICE __HOST
CONSTEXPR T sum_abs(const vector_value<T> &x){
    return func::abs(x.value_x()) + func::abs(x.value_y()) + func::abs(x.value_z());
}

}   // namespace func

_KIAM_MATH_END
