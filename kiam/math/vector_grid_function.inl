#pragma once

#include "assignment.hpp"
#include "context.hpp"
#include "vector_proxy.hpp"
#include "kiam_math_alg.h"
#include "math_operator.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator=(value_type const& value){
    MATH_FILL(vector_type::begin(), vector_type::end(), value);
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator=(const vector_grid_function &other)
{
    assert(other.local_size() == m_local_size);
    vector_type::operator=(other);
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator+=(value_type const& value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 + value
#else
#define FUNC math_bind2nd(math_plus<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator-=(value_type const& value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 - value
#else
#define FUNC math_bind2nd(math_minus<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator*=(value_type const& value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 * value
#else
#define FUNC math_bind2nd(math_multiplies<value_type>(), value)
#endif
    MATH_TRANSFORM(vector_type::cbegin(), vector_type::cbegin() + m_local_size, vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator/=(value_type const& value)
{
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 / value
#else
#define FUNC math_bind2nd(math_divides<value_type>(), value)
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
#define FUNC math_plus<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, other.cbegin(), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator-=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 - ::boost::compute::lambda::_2
#else
#define FUNC math_minus<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, other.cbegin(), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator*=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 * ::boost::compute::lambda::_2
#else
#define FUNC math_multiplies<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, other.cbegin(), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
void vector_grid_function<TAG, T>::operator/=(const vector_grid_function &other)
{
    assert(m_local_size == other.local_size());
#ifdef __OPENCL__
#define FUNC ::boost::compute::lambda::_1 / ::boost::compute::lambda::_2
#else
#define FUNC math_divides<value_type>()
#endif
    MATH_TRANSFORM2(vector_type::cbegin(), vector_type::cbegin() + m_local_size, other.cbegin(), vector_type::begin(), FUNC);
#undef FUNC
}

template<typename TAG, class T>
template<class GEXP>
std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value>
vector_grid_function<TAG, T>::operator+=(GRID_EXPR(GEXP) const& gexp){
    *this = *this + gexp;
}

template<typename TAG, class T>
template<class GEXP>
std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value>
vector_grid_function<TAG, T>::operator-=(GRID_EXPR(GEXP) const& gexp){
    *this = *this - gexp;
}

template<typename TAG, class T>
template<class GEXP>
std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value>
vector_grid_function<TAG, T>::operator*=(GRID_EXPR(GEXP) const& gexp){
    *this = *this * gexp;
}

template<typename TAG, class T>
template<class GEXP>
std::enable_if_t<std::is_same<TAG, typename GEXP::tag_type>::value>
vector_grid_function<TAG, T>::operator/=(GRID_EXPR(GEXP) const& gexp){
    *this = *this / gexp;
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

    IMPLEMENT_DEFAULT_COPY_CONSRUCTOR(vector_grid_function_proxy);

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
    CONSTEXPR const_reference operator()(size_t i, context<CONTEXT> const& context) const {
        return super::operator[](i);
    }

    template<typename CONTEXT>
    __DEVICE
    reference operator()(size_t i, context<CONTEXT> const& context){
        return super::operator[](i);
    }

private:
    size_t const m_local_size;
};

template<typename T, class GEXP>
struct vector_grid_assignment : assignment<typename GEXP::tag_type, vector_grid_assignment<T, GEXP> >
{
    typedef typename GEXP::tag_type tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> func_type;
    typedef GRID_EXPR(GEXP) gexp_type;

    vector_grid_assignment(func_type &func, gexp_type const& gexp) :
        func_proxy(func), gexp_proxy(gexp.get_proxy()){}

    __DEVICE
    void operator()(size_t i) const {
        const_cast<value_type&>(func_proxy[i]) = gexp_proxy[i];
        //gexp_proxy.assign(const_cast<value_type&>(func_proxy[i]), i);
    }

    template<typename CONTEXT>
    __DEVICE
    void operator()(size_t i, context<CONTEXT> const& ctx) const {
        const_cast<value_type&>(func_proxy[i]) = gexp_proxy(i, ctx());
        //gexp_proxy.assign(const_cast<value_type&>(func_proxy[i]), i, ctx);
    }

private:
    typename func_type::proxy_type func_proxy;
    typename GEXP::proxy_type const gexp_proxy;
};

template<typename T, class GEXP>
vector_grid_assignment<T, GEXP>
operator<<=(vector_grid_function<typename GEXP::tag_type, T> &func, GRID_EXPR(GEXP) const& gexp){
    return vector_grid_assignment<T, GEXP>(func, gexp);
}

template<typename TAG, typename T>
template<class GEXP>
void vector_grid_function<TAG, T>::operator=(GRID_EXPR(GEXP) const& gexp)
{
    static_assert(std::is_same<typename GEXP::tag_type, TAG>::value, "Tag types should be the same");
    math_assign<TAG>(*this <<= gexp).exec(size_t(), m_local_size);
}

template<class GEXP>
struct func_callback {
    explicit func_callback(GRID_EXPR(GEXP) const& gexp) : gexp_proxy(gexp.get_proxy()){}

    __DEVICE void operator()(size_t i) const {
        gexp_proxy[i](i);
    }

private:
    typename GEXP::proxy_type const gexp_proxy;
};

template<class GEXP>
static constexpr func_callback<GEXP> get_func_callback(GRID_EXPR(GEXP) const& gexp){
    return func_callback<GEXP>(gexp);
}

//template<typename TAG, typename T>
//template<class GEXP>
//void vector_grid_function<TAG, T>::operator=(func_grid_expression<typename GEXP::tag_type, GEXP, typename GEXP::proxy_type> const& fgexp0)
//{
//    static_assert(std::is_same<typename GEXP::tag_type, TAG>::value, "Tag types should be the same");
//    using _FUNCPROG2::operator*;
//    auto const fgexp = fgexp0 * (*this);
//    auto const callback = get_func_callback(fgexp);
//    default_executor<void>()(m_local_size, callback);
//}

template<typename TAG, typename T>
struct vector_grid_assignment_value : assignment<TAG, vector_grid_assignment_value<TAG, T> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> func_type;

    vector_grid_assignment_value(func_type &func, value_type const& value) : func_proxy(func), value(value){}

    __DEVICE
    void operator()(size_t i) const {
        const_cast<typename func_type::proxy_type&>(func_proxy)[i] = value;
    }

    template<typename CONTEXT>
    __DEVICE
    void operator()(size_t i, context<CONTEXT> const& context) const {
        const_cast<typename func_type::proxy_type&>(func_proxy)[i] = value;
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

template<typename TAG, typename T, class MO>
struct vector_grid_apply : assignment<TAG, vector_grid_apply<TAG, T, MO> >
{
    typedef TAG tag_type;
    typedef T value_type;
    typedef vector_grid_function<tag_type, value_type> func_type;
    typedef MATH_OP(MO) oper_type;

    vector_grid_apply(func_type &func, oper_type const& oper) : func_proxy(func), oper_proxy(oper.get_proxy()){}

    __DEVICE
    void operator()(size_t i){
        oper_proxy(i, func_proxy);
    }

private:
    typename func_type::proxy_type func_proxy;
    typename MO::proxy_type const oper_proxy;
};

template<typename TAG, typename T, class MO>
vector_grid_apply<TAG, T, MO> operator>>(vector_grid_function<TAG, T> &func, MATH_OP(MO) const& oper){
    return vector_grid_apply<TAG, T, MO>(func, oper);
}

#define VECTOR_GRID_APPLY_VALUE_TYPE(z, n, unused) typedef T##n value_type##n;
#define VECTOR_GRID_APPLY_FUNC_TYPE(z, n, unused) typedef vector_grid_function<tag_type, value_type##n> func##n##_type;
#define VECTOR_GRID_APPLY_FUNC(z, n, unused) func##n##_type& func##n
#define VECTOR_GRID_APPLY_FUNC_PROXY_INIT(z, n, unused) func##n##_proxy(func##n)
#define VECTOR_GRID_APPLY_FUNC_PROXY(z, n, unused) func##n##_proxy
#define VECTOR_GRID_APPLY_FUNC_PROXY_DEF(z, n, unused) typename func##n##_type::proxy_type func##n##_proxy;
#define VECTOR_GRID_APPLY_FUNC_REF(z, n, unused) vector_grid_function<TAG, T##n>&
#define VECTOR_GRID_APPLY_FUNC_GET(z, n, unused) std::get<n>(funcs)
#define VECTOR_GRID_APPLY_FUNC_GET_ARRAY(z, n, unused) *funcs[n]
#define VECTOR_GRID_APPLY_T(z, n, unused) T

#define VECTOR_GRID_APPLY(z, n, unused) \
    template<typename TAG, BOOST_PP_ENUM_PARAMS(n, typename T), class MO> \
    struct vector_grid_apply##n : assignment<TAG, vector_grid_apply##n<TAG, BOOST_PP_ENUM_PARAMS(n, T), MO> > { \
        typedef TAG tag_type; \
        BOOST_PP_REPEAT(n, VECTOR_GRID_APPLY_VALUE_TYPE, ~) \
        BOOST_PP_REPEAT(n, VECTOR_GRID_APPLY_FUNC_TYPE, ~) \
        typedef MATH_OP(MO) oper_type; \
        vector_grid_apply##n(BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC, ~), oper_type const& oper) : \
            BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC_PROXY_INIT, ~), oper_proxy(oper.get_proxy()){} \
        __DEVICE void operator()(size_t i){ \
            oper_proxy(i, BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC_PROXY, ~)); \
        } \
    private: \
        BOOST_PP_REPEAT(n, VECTOR_GRID_APPLY_FUNC_PROXY_DEF, ~) \
        typename MO::proxy_type const oper_proxy; \
    }; \
    template<typename TAG, BOOST_PP_ENUM_PARAMS(n, typename T), class MO> \
    vector_grid_apply##n<TAG, BOOST_PP_ENUM_PARAMS(n, T), MO> operator>>( \
        std::tuple<BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC_REF, ~)> &funcs, MATH_OP(MO) const& oper){ \
        return vector_grid_apply##n<TAG, BOOST_PP_ENUM_PARAMS(n, T), MO>(BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC_GET, ~), oper); \
    } \
    template<typename TAG, typename T, class MO> \
    vector_grid_apply##n<TAG, BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_T, ~), MO> operator>>( \
        std::array<vector_grid_function<TAG, T>*, n> &funcs, MATH_OP(MO) const& oper){ \
        return vector_grid_apply##n<TAG, BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_T, ~), MO>(BOOST_PP_ENUM(n, VECTOR_GRID_APPLY_FUNC_GET_ARRAY, ~), oper); \
    }

BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), VECTOR_GRID_APPLY, ~)

#undef VECTOR_GRID_APPLY_VALUE_TYPE
#undef VECTOR_GRID_APPLY_FUNC_TYPE
#undef VECTOR_GRID_APPLY_FUNC
#undef VECTOR_GRID_APPLY_FUNC_PROXY_INIT
#undef VECTOR_GRID_APPLY_FUNC_PROXY
#undef VECTOR_GRID_APPLY_FUNC_PROXY_DEF
#undef VECTOR_GRID_APPLY_FUNC_REF
#undef VECTOR_GRID_APPLY_FUNC_GET
#undef VECTOR_GRID_APPLY_FUNC_GET_ARRAY
#undef VECTOR_GRID_APPLY_T

_KIAM_MATH_END
