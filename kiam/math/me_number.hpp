#pragma once

_KIAM_MATH_BEGIN

template<typename T>
struct me_number
{
    typedef me_number type;
    typedef T data_type;

    __DEVICE __HOST
    CONSTEXPR me_number(data_type mantissa = 0, int exponent = 0)
    {
        m_mantissa = func::frexp(mantissa, &m_exponent);
        if(m_mantissa != 0)
            m_exponent += exponent;
    }

    __DEVICE __HOST
    CONSTEXPR me_number(const me_number &m) : m_mantissa(m.mantissa()), m_exponent(m.exponent()){}

    __DEVICE __HOST
    CONSTEXPR me_number& operator=(const me_number &m){
        m_mantissa = m.mantissa();
        m_exponent = m.exponent();
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR me_number& operator=(data_type v){
        m_mantissa = func::frexp(v, &m_exponent);
        return *this;
    }

    __DEVICE __HOST
    CONSTEXPR data_type mantissa() const {
        return m_mantissa;
    }

    __DEVICE __HOST
    CONSTEXPR int exponent() const {
        return m_exponent;
    }

    __DEVICE __HOST
    CONSTEXPR data_type value() const {
        return func::ldexp(m_mantissa, m_exponent);
    }

    __DEVICE __HOST
    CONSTEXPR operator data_type() const {
        return value();
    }

    __DEVICE __HOST
    CONSTEXPR void operator*=(const me_number &m)
    {
        if(m_mantissa != 0){
            int exponent;
            if((m_mantissa = func::frexp(m_mantissa * m.mantissa(), &exponent)) == 0)
                m_exponent = 0;
            else m_exponent += m.exponent() + exponent;
        }
    }

    __DEVICE __HOST
    CONSTEXPR void operator/=(const me_number &m)
    {
        if(m_mantissa != 0){
            if(m.mantissa() == 0){
                m_mantissa = numeric_limits<data_type>::infinity();
                m_exponent = 0;
            } else {
                int exponent;
                m_mantissa = func::frexp(m_mantissa / m.mantissa(), &exponent);
                m_exponent -= m.exponent() - exponent;
            }
        }
    }

    __DEVICE __HOST
    CONSTEXPR void operator*=(data_type v)
    {
        if(m_mantissa != 0){
            if(v == 0){
                m_mantissa = 0;
                m_exponent = 0;
            } else {
                int exponent;
                m_mantissa = func::frexp(m_mantissa * v, &exponent);
                m_exponent += exponent;
            }
        }
    }

    __DEVICE __HOST
    CONSTEXPR void operator/=(data_type v)
    {
        if(m_mantissa != 0){
            if(v == 0){
                m_mantissa = numeric_limits<data_type>::infinity();
                m_exponent = 0;
            } else {
                int exponent;
                m_mantissa = func::frexp(m_mantissa / v, &exponent);
                m_exponent += exponent;
            }
        }
    }

    __DEVICE __HOST
    CONSTEXPR void operator+=(const me_number &m)
    {
        *this = *this + m;
    }

    __DEVICE __HOST
    CONSTEXPR void operator-=(const me_number &m){
        operator+=(-m);
    }

    __DEVICE __HOST
    CONSTEXPR void operator+=(data_type v){
        operator+=(me_number(v));
    }

    __DEVICE __HOST
    CONSTEXPR void operator-=(data_type v){
        operator+=(me_number(-v));
    }

private:
    data_type m_mantissa;
    int m_exponent;
};

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator-(const me_number<T> &m){
    return me_number<T>(-m.mantissa(), m.exponent());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator*(const me_number<T> &m1, const me_number<T> &m2)
{
    int exponent;
    T mantissa = func::frexp(m1.mantissa() * m2.mantissa(), &exponent);
    return mantissa == 0 ? me_number<T>() : me_number<T>(mantissa, m1.exponent() + m2.exponent() + exponent);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator*(const me_number<T> &m, T v){
    return m * me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator*(T v, const me_number<T> &m){
    return me_number<T>(v) * m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator/(const me_number<T> &m1, const me_number<T> &m2)
{
    int exponent;
    T mantissa = func::frexp(m1.mantissa() / m2.mantissa(), &exponent);
    return mantissa == 0 ? me_number<T>() : me_number<T>(mantissa, m1.exponent() - m2.exponent() + exponent);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator/(const me_number<T> &m, T v){
    return m / me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator/(T v, const me_number<T> &m){
    return me_number<T>(v) / m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator+(const me_number<T> &m1, const me_number<T> &m2)
{
    T mantissa1 = m1.mantissa(), mantissa2 = m2.mantissa();
    int exponent1 = m1.exponent(), exponent2 = m2.exponent();
    if(exponent1 == exponent2){
        int exponent;
        T mantissa = func::frexp(mantissa1 + mantissa2, &exponent);
        return me_number<T>(mantissa, exponent1 + exponent);
    }
    if(exponent1 < exponent2){
        math_swap(mantissa1, mantissa2);
        math_swap(exponent1, exponent2);
    }
    // Now exponent1 > exponent2
    mantissa2 = func::ldexp(mantissa2, exponent2 - exponent1);
    int exponent;
    T mantissa = func::frexp(mantissa1 + mantissa2, &exponent);
    return me_number<T>(mantissa, exponent1 + exponent);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator+(const me_number<T> &m, T v){
    return m + me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator+(T v, const me_number<T> &m){
    return me_number<T>(v) + m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator-(const me_number<T> &m1, const me_number<T> &m2){
    return m1 + me_number<T>(-m2.mantissa(), m2.exponent());
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator-(const me_number<T> &m, T v){
    return m - me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR me_number<T> operator-(T v, const me_number<T> &m){
    return me_number<T>(v) - m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<(const me_number<T> &m1, const me_number<T> &m2){
    return
        m1.mantissa() == 0 ? m2.mantissa() > 0 :
        m2.mantissa() == 0 ? m1.mantissa() < 0 :
        m1.mantissa() < 0 && m2.mantissa() > 0 ? true :
        m1.mantissa() > 0 && m2.mantissa() < 0 ? false :
        m1.exponent() > m2.exponent() ? m1.mantissa() < 0 :
        m1.exponent() < m2.exponent() ? m1.mantissa() > 0 :
        m1.mantissa() < m2.mantissa();
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<(const me_number<T> &m, T v){
    return m < me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<(T v, const me_number<T> &m){
    return me_number<T>(v) < m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>(const me_number<T> &m1, const me_number<T> &m2){
    return m2 < m1;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>(const me_number<T> &m, T v){
    return me_number<T>(v) < m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>(T v, const me_number<T> &m){
    return m < me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator==(const me_number<T> &m1, const me_number<T> &m2){
    return m1.mantissa() == m2.mantissa() && m1.exponent() == m2.exponent();
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator==(const me_number<T> &m, T v){
    return m == me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator==(T v, const me_number<T> &m){
    return me_number<T>(v) == m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator!=(const me_number<T> &m1, const me_number<T> &m2){
    return !(m1 == m2);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator!=(const me_number<T> &m, T v){
    return !(m == me_number<T>(v));
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator!=(T v, const me_number<T> &m){
    return !(me_number<T>(v) == m);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<=(const me_number<T> &m1, const me_number<T> &m2){
    return m1 < m1 || m1 == m2;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<=(const me_number<T> &m, T v){
    return m <= me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator<=(T v, const me_number<T> &m){
    return me_number<T>(v) <= m;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>=(const me_number<T> &m1, const me_number<T> &m2){
    return m2 < m1 || m1 == m2;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>=(const me_number<T> &m, T v){
    return m >= me_number<T>(v);
}

template<typename T>
__DEVICE __HOST
CONSTEXPR bool operator>=(T v, const me_number<T> &m){
    return me_number<T>(v) >= m;
}

template<typename T>
CONSTEXPR std::ostream& operator<<(std::ostream &o, const me_number<T> &m){
    return o << '(' << m.mantissa() << ',' << m.exponent() << ')';
}

_KIAM_MATH_END
