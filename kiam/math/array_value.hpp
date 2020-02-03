#pragma once

#include <cassert>

#include "math_utils.hpp"
#include "type_traits.hpp"
#include "math_mpl.hpp"
#include "kiam_math_func.hpp"

_KIAM_MATH_BEGIN

template<class T, size_t N>
struct array_value
{
    typedef array_value type;
    typedef T value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef pointer iterator;
    typedef const_pointer const_iterator;
    static const size_t array_size = N;

    __DEVICE __HOST
    CONSTEXPR array_value(){
        math_fill_n(m_values, array_size, value_type());
    }

    __DEVICE __HOST
    CONSTEXPR array_value(const array_value &rhs){
        math_copy(rhs.m_values, rhs.m_values + array_size, m_values);
    }

    __DEVICE __HOST
    CONSTEXPR size_t size() const {
        return array_size;
    }

    __DEVICE __HOST
    CONSTEXPR pointer data(){ return m_values; }

    __DEVICE __HOST
    CONSTEXPR const_pointer data() const { return m_values; }

    __DEVICE __HOST
    CONSTEXPR const_reference operator[](size_t i) const
    {
        assert(i < array_size);
        return m_values[i];
    }

    __DEVICE __HOST
    CONSTEXPR reference operator[](size_t i)
    {
        assert(i < array_size);
        return m_values[i];
    }

    __DEVICE __HOST
    CONSTEXPR array_value& operator=(const value_type &value)
    {
        math_fill_n(m_values, array_size, value);
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR iterator begin() {
        return data();
    }

    __DEVICE __HOST
    CONSTEXPR iterator end() {
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
    CONSTEXPR array_value& operator+=(const array_value &rhs)
    {
        math_transform_n(m_values, array_size, rhs.m_values, m_values, plus<value_type>());
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR array_value& operator-=(const array_value &rhs)
    {
        math_transform_n(m_values, array_size, rhs.m_values, m_values, minus<value_type>());
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR array_value& operator*=(const value_type &rhs)
    {
        math_transform_n(m_values, array_size, m_values, [&rhs](const value_type& x) { return x * rhs; });
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR array_value& operator/=(const value_type &rhs)
    {
        math_transform_n(m_values, array_size, m_values, [&rhs](const value_type& x) { return x / rhs; });
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR value_type length() const {
        return func::sqrt(math_inner_product_n(m_values, array_size, m_values, value_type()));
    }

    __DEVICE __HOST
    CONSTEXPR value_type sqr() const {
        return *this & *this;
    }

    __DEVICE __HOST
    CONSTEXPR value_type abs() const {
        return length();
    }

    __DEVICE __HOST
    CONSTEXPR bool operator==(const array_value &rhs) const {
        return math_inner_product_n(m_values, array_size, rhs.m_values, true, logical_and<bool>(), equal_to<value_type>());
    }

    __DEVICE __HOST
    CONSTEXPR bool operator!=(const array_value &rhs) const {
        return math_inner_product_n(m_values, array_size, rhs.m_values, false, logical_or<bool>(), not_equal_to<value_type>());
    }

private:
    value_type m_values[array_size];
};

template <class T> struct is_array_value : std::false_type {};
template <class T> struct is_array_value<const T> : is_array_value<T> {};
template <class T> struct is_array_value<volatile const T> : is_array_value<T> {};
template <class T> struct is_array_value<volatile T> : is_array_value<T> {};
template<class T, size_t N> struct is_array_value<array_value<T, N> > : std::true_type {};

template<typename T, size_t N>
struct get_scalar_type<array_value<T, N> >
{
    typedef T type;
};

template<typename T, size_t N>
struct supports_multiplies<array_value<T, N>, T> : std::true_type {};

template<typename T, size_t N>
struct multiplies_result_type<array_value<T, N>, T>
{
    typedef array_value<T, N> type;
};

template<typename T, size_t N>
struct supports_divides<array_value<T, N>, T> : std::true_type {};

template<typename T, size_t N>
struct divides_result_type<array_value<T, N>, T>
{
    typedef array_value<T, N> type;
};

template<typename T, size_t N>
struct supports_scalar_product<array_value<T, N> > : std::true_type {};

template<typename T, size_t N>
struct supports_component_product<array_value<T, N> > : std::true_type {};

template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator+(const array_value<T, N> &x, const array_value<T, N> &y)
{
    array_value<T, N> result;
    math_transform_n(x.data(), N, y.data(), result.data(), plus<T>());
    return result;
}

template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator-(const array_value<T, N> &x, const array_value<T, N> &y)
{
    array_value<T, N> result;
    math_transform_n(x.data(), N, y.data(), result.data(), minus<T>());
    return result;
}

template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator*(const array_value<T, N> &a, const T &y)
{
    array_value<T, N> result;
    math_transform_n(a.data(), N, result.data(), [&y](const T &x) { return x * y; });
    return result;
}

template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator*(const T &x, const array_value<T, N> &a)
{
    array_value<T, N> result;
    math_transform_n(a.data(), N, result.data(), [&x](const T &y) { return x * y; });
    return result;
}

template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator/(const array_value<T, N> &a, const T &y)
{
    array_value<T, N> result;
    math_transform_n(a.data(), N, result.data(), [&y](const T &x) { return x / y; });
    return result;
}

// Скалярное произведение
template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR T operator&(const array_value<T, N> &x, const array_value<T, N> &y){
    return math_inner_product(x.data(), x.data() + N, y.data(), T());
}

// Покомпонентное произведение
template<class T, size_t N>
__DEVICE __HOST
CONSTEXPR array_value<T, N> operator^(const array_value<T, N> &x, const array_value<T, N> &y)
{
    array_value<T, N> result;
    math_transform_n(x.data(), N, y.data(), result.data(), multiplies<T>());
    return result;
}

_KIAM_MATH_END
