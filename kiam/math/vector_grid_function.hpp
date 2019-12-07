#pragma once

#include "grid_function.hpp"
#include "math_vector.hpp"
#include "host_vector.hpp"

_KIAM_MATH_BEGIN

template<typename TAG, typename T>
struct vector_grid_function_proxy;

template<typename TAG, typename T>
struct vector_grid_function : grid_function<TAG, vector_grid_function<TAG, T>, vector_grid_function_proxy<TAG, T> >, math_vector<T>
{
	typedef vector_grid_function type;
	typedef TAG tag_type;
	typedef T value_type;
	typedef math_vector<value_type> vector_type;
	typedef typename vector_type::reference reference;
	typedef typename vector_type::const_reference const_reference;
	typedef typename vector_type::pointer pointer;
	typedef typename vector_type::const_pointer const_pointer;
	typedef grid_function<tag_type, type, vector_grid_function_proxy<tag_type, value_type> > super;

	vector_grid_function(const vector_grid_function&) = delete;
    vector_grid_function(vector_grid_function&&) = delete;

	vector_grid_function(size_t size, size_t local_size = 0, const value_type &init = value_type()) :
		vector_type(size, init), m_local_size(local_size == 0 ? size : local_size){}

	void operator=(const value_type &value);
	void operator=(const vector_grid_function &other);

	void assign(const HOST_VECTOR_T<value_type> &hvec)
	{
		assert(hvec.size() == vector_type::size());
		vector_type::operator=(hvec);
	}

	template<typename EO>
	void operator=(const EOBJ(EO) &eobj0);

	void resize(size_t new_size, size_t local_size = 0, const value_type &init = value_type())
	{
		vector_type::resize(new_size, init);
		m_local_size = local_size > 0 ? local_size : new_size;
	}

	size_t local_size() const { return m_local_size; }

	reference operator[](size_t i){
		return vector_type::operator[](i);
	}

	const_reference operator[](size_t i) const {
		return vector_type::operator[](i);
	}

	void operator+=(const value_type &value);
	void operator-=(const value_type &value);
	void operator*=(const value_type &value);
	void operator/=(const value_type &value);

	void operator+=(const vector_grid_function &other);
	void operator-=(const vector_grid_function &other);
	void operator*=(const vector_grid_function &other);
	void operator/=(const vector_grid_function &other);

	template<class EO>
	void operator+=(const EOBJ(EO) &eobj);

	template<class EO>
	void operator-=(const EOBJ(EO) &eobj);

	template<class EO>
	void operator*=(const EOBJ(EO) &eobj);

	template<class EO>
	void operator/=(const EOBJ(EO) &eobj);

	const size_t m_local_size;
};

_KIAM_MATH_END

#define REIMPLEMENT_GRID_FUNCTION_OPERATORS() \
	typedef typename super::tag_type tag_type; \
	void operator=(const value_type &value){ super::operator=(value); } \
	void operator=(const super &func){ super::operator=(func); } \
	/*void operator=(const host_vector<value_type> &hvec){ super::operator=(hvec); }*/ \
	template<class EO> \
	void operator=(const EOBJ(EO) &eobj){ super::operator=(eobj); } \
	void operator+=(const value_type &value){ super::operator+=(value); } \
	void operator-=(const value_type &value){ super::operator-=(value); } \
	void operator*=(const value_type &value){ super::operator*=(value); } \
	void operator/=(const value_type &value){ super::operator/=(value); } \
	void operator+=(const super &other){ super::operator+=(other); } \
	void operator-=(const super &other){ super::operator-=(other); } \
	void operator*=(const super &other){ super::operator*=(other); } \
	void operator/=(const super &other){ super::operator/=(other); } \
	template<class EO> \
	void operator+=(const EOBJ(EO) &eobj){ super::operator+=(eobj); } \
	template<class EO> \
	void operator-=(const EOBJ(EO) &eobj){ super::operator-=(eobj); } \
	template<class EO> \
	void operator*=(const EOBJ(EO) &eobj){ super::operator*=(eobj); } \
	template<class EO> \
	void operator/=(const EOBJ(EO) &eobj){ super::operator/=(eobj); }

#define DECLARE_MATH_VECTOR_GRID_FUNCTION(name) \
	template<typename T> \
	using name##_vector_grid_function = _KIAM_MATH::vector_grid_function<name##_tag, T>

#include "vector_grid_function.inl"
