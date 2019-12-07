#pragma once

#include "grid_function.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, class F>
struct math_computable_grid_function_proxy;

template<typename TAG, class F>
struct computable_grid_function :
	grid_function<TAG,
		computable_grid_function<TAG, F>,
		math_computable_grid_function_proxy<TAG, F>
	>
{
	typedef computable_grid_function type;
	typedef grid_function<TAG, type, math_computable_grid_function_proxy<TAG, F> > super;
	typedef get_value_type<F> value_type;

	computable_grid_function(const F &f) : m_f(f){}

	const F& get_f() const { return m_f; }

private:
	const F &m_f;
};

template<typename TAG, class F>
struct math_computable_grid_function_proxy
{
	typedef TAG tag_type;
	typedef get_value_type<F> value_type;

	math_computable_grid_function_proxy(const computable_grid_function<tag_type, F> &func) : f_proxy(func.get_f().get_proxy()){}

	__DEVICE
	value_type operator[](size_t i) const {
		return f_proxy[i];
	}

	__DEVICE
	value_type operator()(size_t i) const {
		return f_proxy(i);
	}

	template<class CONTEXT>
	__DEVICE
	value_type operator()(size_t i, const context<tag_type, CONTEXT> &context) const {
		return f_proxy[i];
	}

private:
	const typename F::proxy_type f_proxy;
};

_KIAM_MATH_END
