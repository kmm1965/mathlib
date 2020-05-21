#pragma once

#include "kiam_math_func.hpp"
#include "evaluable_object.hpp"
#include "type_traits.hpp"

_KIAM_MATH_BEGIN

#define DECARE_FUNC_EVAL_OBJ(func_name, func_impl) \
    template<class EO> \
    struct func_name##_func_evaluable_object : evaluable_object<typename EO::tag_type, func_name##_func_evaluable_object<EO> > { \
        typedef get_value_type_t<EO> value_type; \
        static_assert(is_dimensionless<value_type>::value, "Should be dimensionless"); \
        func_name##_func_evaluable_object(EOBJ(EO) const& eobj) : eobj_proxy(eobj.get_proxy()){} \
        __DEVICE value_type operator[](size_t i) const { return func_impl(eobj_proxy[i]); } \
        private: const typename EO::proxy_type eobj_proxy; \
    }; \
    template<class EO> \
    typename std::enable_if<is_dimensionless<get_value_type_t<EO> >::value,  \
        func_name##_func_evaluable_object<EO> \
    >::type func_name(EOBJ(EO) const& eobj){ \
        return func_name##_func_evaluable_object<EO>(eobj); \
    }

#define DECARE_STD_FUNC_EVAL_OBJ(name) DECARE_FUNC_EVAL_OBJ(name, func::name)

DECARE_STD_FUNC_EVAL_OBJ(sin)
DECARE_STD_FUNC_EVAL_OBJ(cos)
DECARE_STD_FUNC_EVAL_OBJ(tan)
DECARE_STD_FUNC_EVAL_OBJ(asin)
DECARE_STD_FUNC_EVAL_OBJ(acos)
DECARE_STD_FUNC_EVAL_OBJ(atan)
DECARE_STD_FUNC_EVAL_OBJ(sinh)
DECARE_STD_FUNC_EVAL_OBJ(cosh)
DECARE_STD_FUNC_EVAL_OBJ(tanh)
DECARE_STD_FUNC_EVAL_OBJ(ceil)
DECARE_STD_FUNC_EVAL_OBJ(floor)
DECARE_STD_FUNC_EVAL_OBJ(exp)
DECARE_STD_FUNC_EVAL_OBJ(log)
DECARE_STD_FUNC_EVAL_OBJ(log10)

#ifdef __CUDACC__
DECARE_STD_FUNC_EVAL_OBJ(sinpi)
DECARE_STD_FUNC_EVAL_OBJ(cospi)
DECARE_STD_FUNC_EVAL_OBJ(exp2)
DECARE_STD_FUNC_EVAL_OBJ(log1p)
DECARE_STD_FUNC_EVAL_OBJ(log2)
#endif  // __CUDACC__

// Special case for sqrt and sqr
template<class EO>
struct sqrt_func_evaluable_object : evaluable_object<typename EO::tag_type, sqrt_func_evaluable_object<EO> >
{
    typedef sqrt_result_type_t<get_value_type_t<EO> > value_type;

    sqrt_func_evaluable_object(EOBJ(EO) const& eobj) : eobj_proxy(eobj.get_proxy()) {}

    __DEVICE
    value_type operator[](size_t i) const {
        return func::sqrt(eobj_proxy[i]);
    }

private:
    const typename EO::proxy_type eobj_proxy;
};

template<class EO>
sqrt_func_evaluable_object<EO> sqrt(EOBJ(EO) const& eobj) {
    return sqrt_func_evaluable_object<EO>(eobj);
}

template<class EO>
struct sqr_func_evaluable_object : evaluable_object<typename EO::tag_type, sqr_func_evaluable_object<EO> >
{
    typedef sqr_result_type_t<get_value_type_t<EO> > value_type;

    sqr_func_evaluable_object(EOBJ(EO) const& eobj) : eobj_proxy(eobj.get_proxy()) {}

    __DEVICE
    value_type operator[](size_t i) const {
        return func::sqr(eobj_proxy[i]);
    }

private:
    const typename EO::proxy_type eobj_proxy;
};

template<class EO>
sqr_func_evaluable_object<EO> sqr(EOBJ(EO) const& eobj) {
    return sqr_func_evaluable_object<EO>(eobj);
}

#define DECARE_FUNC2_EVAL_OBJ(func_name, func_impl) \
    template<class EO1, class EO2> \
    struct func_name##_func2_evaluable_object : evaluable_object<typename EO1::tag_type, func_name##_func2_evaluable_object<EO1, EO2> > { \
        typedef get_value_type_t<EO1> value_type; \
        static_assert(is_dimensionless<value_type>::value, "Should be dimensionless"); \
        func_name##_func2_evaluable_object(EOBJ(EO1) const& eobj1, EOBJ(EO2) const& eobj2) : \
            eobj1_proxy(eobj1.get_proxy()), eobj2_proxy(eobj2.get_proxy()){} \
        __DEVICE value_type operator[](size_t i) const { return func_impl(eobj1_proxy[i], eobj2_proxy[i]); } \
        private: \
            const typename EO1::proxy_type eobj1_proxy; \
            const typename EO2::proxy_type eobj2_proxy; \
    }; \
    template<class EO1, class EO2> \
    typename std::enable_if<is_dimensionless<get_value_type_t<EO1> >::value,  \
        func_name##_func2_evaluable_object<EO1, EO2> \
    >::type func_name(EOBJ(EO1) const& eobj1, EOBJ(EO2) const& eobj2){ \
        return func_name##_func2_evaluable_object<EO1, EO2>(eobj1, eobj2); \
    }

_KIAM_MATH_END
