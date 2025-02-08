#pragma once

#include "math_object.hpp"
#include "math_mpl.hpp"
#include "math_utils.hpp"
#include "pow.hpp"

_KIAM_MATH_BEGIN

template<typename E>
struct expression : math_object_base<E>{};

template<typename E>
using is_expression = std::is_base_of<expression<typename std::remove_const<E>::type>, typename std::remove_const<E>::type>;

template<typename T>
struct constant : expression<constant<T> >
{
    typedef T value_type;

    constant(value_type const& value) : value(value){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const&) const {
        return value;
    }

    value_type const value;
};

template<typename T>
constant<T> _(T const& value){
    return constant<T>(value);
}

template<typename T>
struct variable : expression<variable<T> >
{
    typedef T value_type;

    variable(const char *name) : name(name){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const& eval) const
    {
        static_assert(std::is_same< get_value_type_t<EVAL> , value_type>::value, "Bad value type of type EVAL");
        return eval(pt, name);
    }

    math_string const name;
};

template<typename OP, typename E>
struct unary_expression : expression<unary_expression<OP, E> >
{
    typedef get_value_type_t<E> value_type;

    static_assert(std::is_same<typename OP::result_type, value_type>::value, "Bad result type of type OP");

    unary_expression(expression<E> const& e) : op(), e(e()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return op(e(pt, eval));
    }

    OP const op;
    E const e;
};

template<typename E>
unary_expression<math_negate<get_value_type_t<E> >, E>
operator-(expression<E> const& e){
    return unary_expression<math_negate<get_value_type_t<E> >, E>(e);
}

template<typename E1, typename OP, typename E2>
struct binary_expression : expression<binary_expression<E1, OP, E2> >
{
    typedef get_value_type_t<E1> value_type;

    static_assert(std::is_same< get_value_type_t<E2>, value_type>::value, "Value types should be the same");
    static_assert(std::is_same<typename OP::result_type, value_type>::value, "Bad result type of type OP");

    binary_expression(expression<E1> const& e1, expression<E2> const& e2) : e1(e1()), op(), e2(e2()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return op(e1(pt, eval), e2(pt, eval));
    }

    E1 const e1;
    OP const op;
    E2 const e2;
};

#define DECLARE_BINARY_EXPRESSION(op, oper) \
    template<typename E1, typename E2> \
    typename std::enable_if<is_expression<E1>::value && is_expression<E2>::value, binary_expression<E1, oper<get_value_type_t<E1> >, E2> >::type \
    operator op(expression<E1> const& e1, expression<E2> const& e2) \
    { \
        static_assert(std::is_same<get_value_type_t<E1>, get_value_type_t<E2> >::value, "Value types should be the same"); \
        return binary_expression<E1, oper<get_value_type_t<E1> >, E2>(e1, e2); \
    } \
    template<typename E1, typename E2> \
    typename std::enable_if<is_expression<E1>::value && !is_expression<E2>::value, binary_expression<E1, oper<get_value_type_t<E1> >, constant<E2> > >::type \
    operator op(expression<E1> const& e1, E2 const& e2) \
    { \
        static_assert(std::is_same<get_value_type_t<E1>, E2>::value, "Value types should be the same"); \
        return binary_expression<E1, oper<get_value_type_t<E1> >, constant<E2> >(e1, constant<E2>(e2)); \
    }

DECLARE_BINARY_EXPRESSION(+, math_plus)
DECLARE_BINARY_EXPRESSION(-, math_minus)
DECLARE_BINARY_EXPRESSION(*, math_multiplies)
DECLARE_BINARY_EXPRESSION(/, math_divides)

template<typename E, unsigned N>
struct pow_expression : expression<pow_expression<E, N> >
{
    typedef get_value_type_t<E> value_type;

    pow_expression(expression<E> const& e) : e(e()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return math_pow<N>(e(pt, eval));
    }

    E const e;
};

template<unsigned N, typename E>
pow_expression<E, N> _pow(expression<E> const& e){
    return pow_expression<E, N>(e);
}

template<typename C>
struct condition : math_object_base<C>{};

template<typename C>
using is_condition = std::is_base_of<condition<typename std::remove_const<C>::type>, typename std::remove_const<C>::type>;

template<typename E1, typename PRED, typename E2>
struct logical_condition : condition<logical_condition<E1, PRED, E2> >
{
    typedef get_value_type_t<E1> value_type;

    static_assert(std::is_same<get_value_type_t<E2>, value_type>::value, "Value types should be the same");

    logical_condition(expression<E1> const& e1, expression<E2> const& e2) : e1(e1()), pred(), e2(e2()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR bool operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return pred(e1(pt, eval), e2(pt, eval));
    }

    E1 const e1;
    PRED const pred;
    E2 const e2;
};

#define DECLARE_CONDITION(op, pred) \
    template<typename E1, typename E2> \
    typename std::enable_if<is_expression<E1>::value && is_expression<E2>::value, logical_condition<E1, pred<get_value_type_t<E1> >, E2> >::type \
    operator op(expression<E1> const& e1, expression<E2> const& e2) \
    { \
        static_assert(std::is_same<get_value_type_t<E1>, get_value_type_t<E2> >::value, "Value types should be the same"); \
        return logical_condition<E1, pred<get_value_type_t<E1> >, E2>(e1, e2); \
    } \
    template<typename E1, typename E2> \
    typename std::enable_if<is_expression<E1>::value && !is_expression<E2>::value, logical_condition<E1, pred<get_value_type_t<E1> >, constant<E2> > >::type \
    operator op(expression<E1> const& e1, E2 const& e2) \
    { \
        static_assert(std::is_same<get_value_type_t<E1>, E2>::value, "Value types should be the same"); \
        return logical_condition<E1, pred<get_value_type_t<E1> >, constant<E2> >(e1, constant<E2>(e2)); \
    }

DECLARE_CONDITION(== , math_equal_to)
DECLARE_CONDITION(!= , math_not_equal_to)
DECLARE_CONDITION(< , math_less)
DECLARE_CONDITION(<=, math_less_equal)
DECLARE_CONDITION(>, math_greater)
DECLARE_CONDITION(>= , math_greater_equal)

template<typename OP, typename C>
struct unary_condition : condition<unary_condition<OP, C> >
{
    unary_condition(condition<C> const& c) : op(), c(c()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR bool operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return op(c(pt, eval));
    }

    OP const op;
    C const c;
};

template<typename C>
unary_condition<math_logical_not, C> operator!(condition<C> const& c){
    return unary_condition<math_logical_not, C>(c);
}

template<typename C1, typename OP, typename C2>
struct binary_condition : condition<binary_condition<C1, OP, C2> >
{
    binary_condition(condition<C1> const& c1, condition<C2> const& c2) : c1(c1()), op(), c2(c2()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR bool operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return op(c1(pt, eval), c2(pt, eval));
    }

    C1 const c1;
    OP const op;
    C2 const c2;
};

template<typename C1, typename C2>
binary_condition<C1, math_logical_and, C2> operator&&(condition<C1> const& c1, condition<C2> const& c2){
    return binary_condition<C1, math_logical_and, C2>(c1, c2);
}

template<typename C1, typename C2>
binary_condition<C1, math_logical_or, C2> operator||(condition<C1> const& c1, condition<C2> const& c2){
    return binary_condition<C1, math_logical_or, C2>(c1, c2);
}

template<typename C, typename E1, typename E2>
struct conditional_expression : expression<conditional_expression<C, E1, E2> >
{
    typedef get_value_type_t<E1> value_type;
    static_assert(std::is_same< get_value_type_t<E2>, value_type>::value, "Value types should be the same");

    conditional_expression(condition<C> const& cond, expression<E1> const& e1, expression<E2> const& e2) :
        cond(cond()), e1(e1()), e2(e2()){}

    template<typename VECTOR_TYPE, typename EVAL>
    __DEVICE __HOST
    CONSTEXPR value_type operator()(VECTOR_TYPE const& pt, EVAL const& eval) const {
        return cond(pt, eval) ? e1(pt, eval) : e2(pt, eval);
    }

    C const cond;
    E1 const e1;
    E2 const e2;
};

template<typename C, typename E1, typename E2>
typename std::enable_if<is_expression<E1>::value && is_expression<E2>::value, conditional_expression<C, E1, E2> >::type
_if(condition<C> const& cond, expression<E1> const& e1, expression<E2> const& e2){
    return conditional_expression<C, E1, E2>(cond, e1, e2);
}

template<typename C, typename E1, typename E2>
typename std::enable_if<is_expression<E1>::value && !is_expression<E2>::value, conditional_expression<C, E1, constant<E2> > >::type
_if(condition<C> const& cond, expression<E1> const& e1, E2 const& e2)
{
    static_assert(std::is_same<get_value_type_t<E1>, E2>::value, "Value types should be the same");
    return conditional_expression<C, E1, constant<E2> >(cond, e1, constant<E2>(e2));
}

template<typename C, typename E1, typename E2>
typename std::enable_if<!is_expression<E1>::value && is_expression<E2>::value, conditional_expression<C, constant<E1>, E2> >::type
_if(condition<C> const& cond, E1 const& e1, expression<E2> const& e2)
{
    static_assert(std::is_same<E1, get_value_type_t<E2> >::value, "Value types should be the same");
    return conditional_expression<C, constant<E1>, E2>(cond, constant<E1>(e1), e2);
}

template<typename C, typename E1, typename E2>
typename std::enable_if<!is_expression<E1>::value && !is_expression<E2>::value, conditional_expression<C, constant<E1>, constant<E2> > >::type
_if(condition<C> const& cond, E1 const& e1, E2 const& e2)
{
    static_assert(std::is_same<E1, E2>::value, "Value types should be the same");
    return conditional_expression<C, constant<E1>, constant<E2> >(cond, constant<E1>(e1), constant<E2>(e2));
}

_KIAM_MATH_END
