#pragma once

#include "assignment.hpp"
#include "context.hpp"
#include "vector_proxy.hpp"
#include "kiam_math_alg.h"

_KIAM_MATH_BEGIN

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator=(const value_type &value){
    MATH_FILL(vector_type::begin(), vector_type::end(), value);
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator=(const vector_grid_function &other)
{
    assert(other.local_size() == m_local_size);
    vector_type::operator=(other);
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator+=(const value_type &value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 + value
#else
#define FUNC math_bind2nd(_KIAM_MATH::plus<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator-=(const value_type &value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 - value
#else
#define FUNC math_bind2nd(_KIAM_MATH::minus<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator*=(const value_type &value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 * value
#else
#define FUNC math_bind2nd(_KIAM_MATH::multiplies<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator/=(const value_type &value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 / value
#else
#define FUNC math_bind2nd(_KIAM_MATH::divides<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator+=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 + ::boost::compute::lambda::_2
#else
#define FUNC _KIAM_MATH::plus<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, std::cbegin(other), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator-=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 - ::boost::compute::lambda::_2
#else
#define FUNC _KIAM_MATH::minus<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, std::cbegin(other), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator*=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 * ::boost::compute::lambda::_2
#else
#define FUNC _KIAM_MATH::multiplies<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, std::cbegin(other), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator/=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 / ::boost::compute::lambda::_2
#else
#define FUNC _KIAM_MATH::divides<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, std::cbegin(other), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
template<class EO>
void vector_grid_function<TAG, T>::operator+=(const EOBJ(EO) &eobj)
{
    static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tag types should be the same");
    *this = *this + eobj;
}

template<typename TAG, class T>
template<class EO>
void vector_grid_function<TAG, T>::operator-=(const EOBJ(EO) &eobj)
{
    static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tag types should be the same");
    *this = *this - eobj;
}

template<typename TAG, class T>
template<class EO>
void vector_grid_function<TAG, T>::operator*=(const EOBJ(EO) &eobj)
{
    static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tag types should be the same");
    *this = *this * eobj;
}

template<typename TAG, class T>
template<class EO>
void vector_grid_function<TAG, T>::operator/=(const EOBJ(EO) &eobj)
{
    static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tag types should be the same");
    *this = *this / eobj;
}

template<typename TAG, typename T>
struct vector_grid_function_proxy : vector_proxy<T>
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_proxy<value_type> super;
    typedef typename super::reference reference;
    typedef typename super::const_reference const_reference;
    typedef vector_grid_function<tag_type, value_type> vector_grid_function_type;

    vector_grid_function_proxy(vector_grid_function_type &func) : super(func), m_local_size(func.m_local_size){}
    vector_grid_function_proxy(const vector_grid_function_type &func) : super(func), m_local_size(func.m_local_size){}

    __DEVICE
    CONSTEXPR size_t local_size() const { return m_local_size; }

    __DEVICE
    reference operator[](size_t i){
        return super::operator[](i);
    }

    __DEVICE
    CONSTEXPR const_reference operator[](size_t i) const {
        return super::operator[](i);
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR const_reference operator()(size_t i, const context<tag_type, CONTEXT> &context) const {
        return super::operator[](i);
    }

    template<typename CONTEXT>
    __DEVICE
    reference operator()(size_t i, const context<tag_type, CONTEXT> &context){
        return super::operator[](i);
    }

private:
    const size_t m_local_size;
};

template<typename T, class EO>
struct vector_grid_assignment : assignment<typename EO::tag_type, vector_grid_assignment<T, EO> >
{
    typedef typename EO::tag_type tag_type;
    typedef T value_type;
    typedef vector_grid_assignment type;
    typedef assignment<tag_type, type> super;
    typedef EOBJ(EO) eobj_type;
    typedef vector_grid_function<tag_type, value_type> func_type;

    vector_grid_assignment(func_type &func, const eobj_type &eobj) :
        func_proxy(func), eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    void operator[](size_t i){
        //func_proxy[i] = eobj_proxy[i];
        eobj_proxy.assign(func_proxy[i], i);
    }

    __DEVICE
    void operator()(size_t i){
        //func_proxy(i) = eobj_proxy(i);
        eobj_proxy.assign(func_proxy(i), i);
    }

    template<typename CONTEXT>
    __DEVICE
    void operator()(size_t i, const context<tag_type, CONTEXT> &context){
        //func_proxy[i] = eobj_proxy(i, context);
        eobj_proxy.assign(func_proxy[i], i, context);
    }

    typename func_type::proxy_type func_proxy;
    const typename EO::proxy_type eobj_proxy;
};

template<typename T, class EO>
vector_grid_assignment<T, EO>
operator<<=(
    vector_grid_function<typename EO::tag_type, T> &func,
    const EOBJ(EO) &eobj
){
    return vector_grid_assignment<T, EO>(func, eobj);
}

template<typename TAG, typename T>
template<class EO>
void vector_grid_function<TAG, T>::operator=(const EOBJ(EO) &eobj)
{
    typedef typename EO::tag_type EO_tag_type;
    static_assert(std::is_same<EO_tag_type, TAG>::value, "Tag types should be the same");
    math_assign<TAG>(*this <<= eobj).exec(size_t(), m_local_size);
}

template<typename TAG, typename T>
struct vector_grid_assignment_value : assignment<TAG, vector_grid_assignment_value<TAG, T> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_assignment_value type;
    typedef assignment<tag_type, value_type, type> super;
    typedef vector_grid_function<tag_type, value_type> func_type;

    vector_grid_assignment_value(func_type &func, const value_type &value) : func_proxy(func), value(value){}

    __DEVICE
    void operator[](size_t i){
        func_proxy[i] = value;
    }

    __DEVICE
    void operator()(size_t i){
        func_proxy(i) = value;
    }

    template<typename CONTEXT>
    __DEVICE
    void operator()(size_t i, const context<tag_type, CONTEXT> &context){
        func_proxy[i] = value;
    }

private:
    typename func_type::proxy_type func_proxy;
    const value_type value;
};

template<typename TAG, typename T>
vector_grid_assignment_value<TAG, T>
operator<<=(vector_grid_function<TAG, T> &func, const T &value){
    return vector_grid_assignment_value<TAG, T>(func, value);
}

_KIAM_MATH_END
