#pragma once

#include "../gridop2_src/cyclic_index.hpp"

template<class OP>
struct first_derivative_operator_x : cyclic_index_operator_x<OP>
{
    typedef cyclic_index_operator_x<OP> super;

    template<typename T>
    struct get_value_type;

    template<class Unit, class Y>
    struct get_value_type<units::quantity<Unit, Y> >
    {
        typedef typename units::divide_typeof_helper<units::quantity<Unit, Y>, length_type>::type type;
    };

protected:
    first_derivative_operator_x(typename super::index_type const& index) : super(index){}
};

template<class OP>
struct first_derivative_operator_y : cyclic_index_operator_y<OP>
{
    typedef cyclic_index_operator_y<OP> super;

    template<typename T>
    struct get_value_type;

    template<class Unit, class Y>
    struct get_value_type<units::quantity<Unit, Y> >
    {
        typedef typename units::divide_typeof_helper<units::quantity<Unit, Y>, length_type>::type type;
    };

protected:
    first_derivative_operator_y(typename super::index_type const& index) : super(index) {}
};

template<class OP>
struct forward_first_derivative_operator_x : first_derivative_operator_x<OP>
{
	typedef first_derivative_operator_x<OP> super;

protected:
	forward_first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index), h(h){}

	template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, const EOP &eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix + 1, iy)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix, iy)]]) / h;
    }

private:
	const length_type h;
};

template<class OP>
struct backward_first_derivative_operator_x : first_derivative_operator_x<OP>
{
    typedef first_derivative_operator_x<OP> super;

protected:
	backward_first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index), h(h){}

	template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, const EOP &eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix, iy)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix - 1, iy)]]) / h;
    }

private:
    const length_type h;
};

template<class OP>
struct central_first_derivative_operator_x : first_derivative_operator_x<OP>
{
    typedef first_derivative_operator_x<OP> super;

protected:
	central_first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index), h2(h * 2.){}

	template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, EOP &eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix + 1, iy)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix - 1, iy)]]) / h2;
    }

private:
    const length_type h2;
};

template<typename TAG_X>
struct first_derivative_operator_x;

template<>
struct first_derivative_operator_x<tag_main> : forward_first_derivative_operator_x<first_derivative_operator_x<tag_main> >
{
	typedef forward_first_derivative_operator_x<first_derivative_operator_x> super;

    template<typename TAG>
    struct get_tag_type;

    template<typename tag_y>
    struct get_tag_type<std::tuple<tag_main, tag_y> >
    {
        typedef std::tuple<tag_aux, tag_y> type;
    };

	first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index, h) {};

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)
    REIMPLEMENT_OPERATOR_EXEC()
};

template<>
struct first_derivative_operator_x<tag_aux> : backward_first_derivative_operator_x<first_derivative_operator_x<tag_aux> >
{
	typedef backward_first_derivative_operator_x<first_derivative_operator_x> super;

    template<typename TAG>
    struct get_tag_type;

    template<typename tag_y>
    struct get_tag_type<std::tuple<tag_aux, tag_y> >
    {
        typedef std::tuple<tag_main, tag_y> type;
    };

	first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index, h){};

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)
    REIMPLEMENT_OPERATOR_EXEC()
};

/////////////
template<class OP>
struct forward_first_derivative_operator_y : first_derivative_operator_y<OP>
{
	typedef first_derivative_operator_y<OP> super;

protected:
	forward_first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index), h(h) {}

    template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix, iy + 1)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix, iy)]]) / h;
    }

private:
    const length_type h;
};

template<class OP>
struct backward_first_derivative_operator_y : first_derivative_operator_y<OP>
{
    typedef first_derivative_operator_y<OP> super;

protected:
	backward_first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index), h(h) {}

	template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix, iy)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix, iy - 1)]]) / h;
    }

private:
    const length_type h;
};

template<class OP>
struct central_first_derivative_operator_y : first_derivative_operator_y<OP>
{
    typedef first_derivative_operator_y<OP> super;

protected:
	central_first_derivative_operator_y(const length_type &h) : h2(h * 2.) {}

	template<typename EOP>
    __DEVICE
    typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i_, EOP& eobj_proxy) const
    {
        const typename super::index_value_type i = super::m_index.value(i_);
        const typename super::cyclic_index_value_type ix = i.first, iy = i.second;
        return (eobj_proxy[super::m_index[typename super::index_value_type(ix, iy + 1)]] -
			eobj_proxy[super::m_index[typename super::index_value_type(ix, iy - 1)]]) / h2;
    }

private:
    const length_type h2;
};

template<typename TAG_Y>
struct first_derivative_operator_y;

template<>
struct first_derivative_operator_y<tag_main> : forward_first_derivative_operator_y<first_derivative_operator_y<tag_main> >
{
	typedef forward_first_derivative_operator_y<first_derivative_operator_y> super;

    template<typename TAG>
    struct get_tag_type;

    template<typename tag_x>
    struct get_tag_type<std::tuple<tag_x, tag_main> >
    {
        typedef std::tuple<tag_x, tag_aux> type;
    };

	first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index, h) {};

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)
    REIMPLEMENT_OPERATOR_EXEC()
};

template<>
struct first_derivative_operator_y<tag_aux> : backward_first_derivative_operator_y<first_derivative_operator_y<tag_aux> >
{
	typedef backward_first_derivative_operator_y<first_derivative_operator_y> super;

    template<typename TAG>
    struct get_tag_type;

    template<typename tag_x>
    struct get_tag_type<std::tuple<tag_x, tag_aux> >
    {
        typedef std::tuple<tag_x, tag_main> type;
    };

	first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index, h) {};

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)
    REIMPLEMENT_OPERATOR_EXEC()
};
