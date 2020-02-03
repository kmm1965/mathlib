#pragma once

#include "kiam_math_func.hpp"
#include "evaluable_object.hpp"
#include "type_traits.hpp"

_KIAM_MATH_BEGIN

#define DECARE_FUNC_EVAL_OBJ(name) \
    template<class EO> \
    struct name##_func_evaluable_object : evaluable_object<typename EO::tag_type, name##_func_evaluable_object<EO> > { \
        typedef get_value_type_t<EO> value_type; \
        static_assert(is_dimensionless<value_type>::value, "Should be dimensionless"); \
        name##_func_evaluable_object(EOBJ(EO) const& eobj) : eobj_proxy(eobj.get_proxy()){} \
        __DEVICE value_type operator[](size_t i) const { return func::name(eobj_proxy[i]); } \
        private: const typename EO::proxy_type eobj_proxy; \
    }; \
    template<class EO> \
    typename std::enable_if<is_dimensionless<get_value_type_t<EO> >::value,  \
        name##_func_evaluable_object<EO> \
    >::type name(EOBJ(EO) const& eobj){ \
        return name##_func_evaluable_object<EO>(eobj); \
    }

DECARE_FUNC_EVAL_OBJ(sin)
DECARE_FUNC_EVAL_OBJ(cos)
DECARE_FUNC_EVAL_OBJ(tan)
DECARE_FUNC_EVAL_OBJ(asin)
DECARE_FUNC_EVAL_OBJ(acos)
DECARE_FUNC_EVAL_OBJ(atan)
DECARE_FUNC_EVAL_OBJ(sinh)
DECARE_FUNC_EVAL_OBJ(cosh)
DECARE_FUNC_EVAL_OBJ(tanh)
DECARE_FUNC_EVAL_OBJ(ceil)
DECARE_FUNC_EVAL_OBJ(floor)
DECARE_FUNC_EVAL_OBJ(exp)
DECARE_FUNC_EVAL_OBJ(log)
DECARE_FUNC_EVAL_OBJ(log10)

#ifdef __CUDACC__
DECARE_FUNC_EVAL_OBJ(sinpi)
DECARE_FUNC_EVAL_OBJ(cospi)
DECARE_FUNC_EVAL_OBJ(exp2)
DECARE_FUNC_EVAL_OBJ(log1p)
DECARE_FUNC_EVAL_OBJ(log2)
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

_KIAM_MATH_END
