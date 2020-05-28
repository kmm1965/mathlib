#pragma once

#include "evaluable_object.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class EO>
struct negate_evaluable_object : evaluable_object<typename EO::tag_type, negate_evaluable_object<EO> >
{
    typedef typename EO::tag_type tag_type;
    typedef EOBJ(EO) eobj_type;
    typedef get_value_type_t<EO> value_type;

    negate_evaluable_object(const eobj_type& eobj) : eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return -eobj_proxy[i];
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return -eobj_proxy(i);
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, const context<tag_type, CONTEXT>& context) const {
        return -eobj_proxy(i, context);
    }

private:
    const typename EO::proxy_type eobj_proxy;
};

template<class EO>
negate_evaluable_object<EO> operator-(EOBJ(EO) const& eobj){
    return negate_evaluable_object<EO>(eobj);
}

_KIAM_MATH_END
