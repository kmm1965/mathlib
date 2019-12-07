#pragma once

#include "math_operator.hpp"

_KIAM_MATH_BEGIN

template<class MOP, class EOP>
struct composition_operator_eobj_proxy
{
	typedef typename MOP::template get_value_type<get_value_type_t<EOP>>::type value_type;
	typedef typename MOP::tag_type tag_type;

	__DEVICE
    CONSTEXPR composition_operator_eobj_proxy(const MOP &op_proxy, const EOP &eobj_proxy) : op_proxy(op_proxy), eobj_proxy(eobj_proxy){}

	__DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
		return op_proxy(i, eobj_proxy);
	}

	__DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
		return op_proxy(i, eobj_proxy);
	}

private:
	const MOP &op_proxy;
	const EOP &eobj_proxy;
};

template<class OP1, class OP2>
struct composition_operator : math_operator<composition_operator<OP1, OP2> >
{
    template<typename EO_TAG>
    struct get_tag_type
    {
        typedef typename OP1::template get_tag_type<typename OP2::template get_tag_type<EO_TAG>::type>::type type;
    };

	template<class T>
	struct get_value_type
	{
		typedef typename OP1::template get_value_type<
			typename OP2::template get_value_type<T>::type
		>::type type;
	};

	composition_operator(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2) :
        op1_proxy(op1.get_proxy()), op2_proxy(op2.get_proxy()){}

	template<class EOP>
	__DEVICE
	typename get_value_type<get_value_type_t<EOP>>::type
    CONSTEXPR operator()(size_t i, const EOP &eobj_proxy) const {
		return op1_proxy(i, composition_operator_eobj_proxy<typename OP2::proxy_type, EOP>(op2_proxy, eobj_proxy));
	}

	IMPLEMENT_MATH_EVAL_OPERATOR(composition_operator)

private:
	const typename OP1::proxy_type op1_proxy;
	const typename OP2::proxy_type op2_proxy;
};

template<class OP1, class OP2>
composition_operator<OP1, OP2> operator&(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
	return composition_operator<OP1, OP2>(op1, op2);
}

_KIAM_MATH_END
