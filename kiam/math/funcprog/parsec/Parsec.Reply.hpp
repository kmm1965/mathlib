#pragma once

#include "Parsec.Error.hpp"
#include "../Functor.hpp"

_PARSEC_BEGIN

template<typename S, typename U, typename A>
struct c_Ok
{
    using stream_type = S;
    using user_state_type = U;
    using value_type = A;
    using state_type = State<stream_type, user_state_type>;

    c_Ok(value_type const& value, state_type const& state, ParseError const& error) : value(value), state(state), error(error){}

    value_type const value;
    state_type const state;
    ParseError const error;
};

#define OK_(S, U, A) BOOST_IDENTITY_TYPE((_PARSEC::c_Ok<S, U, A>))
#define OK(S, U, A) typename OK_(S, U, A)

struct c_Error
{
    c_Error(ParseError const& error) : error(error){}

    ParseError const error;
};

enum { Ok_, Error_ };

template<typename S, typename U, typename A>
struct Reply;

template<typename S, typename U>
struct _Reply
{
    using base_class = _Reply;

    template<typename A>
    using type = Reply<S, U, A>;
};

#define _REPLY_(S, U) BOOST_IDENTITY_TYPE((_PARSEC::_Reply<S, U>))
#define _REPLY(S, U) typename _REPLY_(S, U)

template<typename S, typename U, typename A>
struct Reply : variant_t<c_Ok<S, U, A>, c_Error>, _Reply<S, U>
{
    using super = variant_t<c_Ok<S, U, A>, c_Error>;

    Reply(c_Ok<S, U, A> const& value) : super(value){}
    Reply(c_Error const& value) : super(value){}

    constexpr c_Ok<S, U, A> const& ok() const
    {
        assert(super::index() == Ok_);
        return super::template get<c_Ok<S, U, A> >();
    }

    constexpr c_Error const& error() const
    {
        assert(super::index() == Error_);
        return super::template get<c_Error>();
    }
};

#define REPLY_(S, U, A) BOOST_IDENTITY_TYPE((_PARSEC::Reply<S, U, A>))
#define REPLY(S, U, A) typename REPLY_(S, U, A)

DECLARE_FUNCTION_3(3, REPLY(T0, T1, T2), __Ok, T2 const&, STATE(T0, T1) const&, ParseError const&)
FUNCTION_TEMPLATE(3) constexpr REPLY(T0, T1, T2) __Ok(T2 const& value, STATE(T0, T1) const& state, ParseError const& error) {
    return OK(T0, T1, T2)(value, state, error);
}

template<typename S, typename U, typename A>
constexpr Reply<S, U, A> __Error(ParseError const& error){
    return c_Error(error);
}

_PARSEC_END

_FUNCPROG_BEGIN

// Functor
template<typename S, typename U>
struct _is_functor<parsec::_Reply<S, U> > : std::true_type {};

template<typename S, typename U>
struct Functor<parsec::_Reply<S, U> > : _Functor<parsec::_Reply<S, U> >
{
    template<typename Ret, typename Arg>
    static constexpr parsec::Reply<S, U, Ret>
    fmap(function_t<Ret(Arg)> const& f, parsec::Reply<S, U, fdecay<Arg> > const& v)
    {
        if (v.index() == parsec::Ok_){
            parsec::c_Ok<S, U, fdecay<Arg> > const& ok = v.ok();
            return parsec::c_Ok<S, U, Ret>(f(ok.value), ok.state, ok.error);
        } else return v.error();
    }
};

_FUNCPROG_END
