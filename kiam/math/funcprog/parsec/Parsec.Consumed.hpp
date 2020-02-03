#pragma once

#include "../Functor.hpp"

_PARSEC_BEGIN

template<typename A>
struct ConsumedBase
{
    using value_type = A;

protected:
    ConsumedBase(value_type const& value) : value(value) {}
    ConsumedBase(f0<value_type> const& fvalue) : value(fvalue) {}

public:
    value_type operator*() const {
        return value();
    }

private:
    const fdata<value_type> value;
};

template<typename A>
struct c_Consumed : ConsumedBase<A>
{
    c_Consumed(A const& value) : ConsumedBase<A>(value) {}
    c_Consumed(f0<A> const& fvalue) : ConsumedBase<A>(fvalue) {}
};

template<typename A>
struct c_Empty : ConsumedBase<A> 
{
    c_Empty(A const& value) : ConsumedBase<A>(value) {}
    c_Empty(f0<A> const& fvalue) : ConsumedBase<A>(fvalue) {}
};

enum { Consumed_, Empty_ };

template<typename A>
struct Consumed;

struct _Consumed
{
    using base_class = _Consumed;

    template<typename A>
    using type = Consumed<A>;
};

template<typename A>
struct Consumed : variant_t<c_Consumed<A>, c_Empty<A> >, _Consumed
{
    using super = variant_t<c_Consumed<A>, c_Empty<A> >;

    Consumed(const c_Consumed<A> &value) : super(value) {}
    Consumed(const c_Empty<A> &value) : super(value) {}

    const c_Consumed<A>& consumed() const
    {
        assert(super::index() == Consumed_);
        return super::template get<c_Consumed<A> >();
    }

    const c_Empty<A>& empty() const
    {
        assert(super::index() == Empty_);
        return super::template get<c_Empty<A> >();
    }
};

template<typename T>
Consumed<T> __Consumed(T const& value) {
    return c_Consumed<T>(value);
}

template<typename T>
Consumed<T> __Empty(T const& value) {
    return c_Empty<T>(value);
}

_PARSEC_END

_FUNCPROG_BEGIN

// Functor
IMPLEMENT_FUNCTOR(parsec::Consumed, parsec::_Consumed)

template<>
struct Functor<parsec::_Consumed>
{
    DECLARE_FUNCTOR_CLASS(parsec::Consumed)
};

//instance Functor Consumed where
//    fmap f (Consumed x) = Consumed (f x)
//    fmap f (Empty x)    = Empty (f x)
template<typename Ret, typename Arg, typename... Args>
parsec::Consumed<remove_f0_t<function_t<Ret(Args...)> > >
Functor<parsec::_Consumed>::fmap(function_t<Ret(Arg, Args...)> const& f, parsec::Consumed<fdecay<Arg> > const& v) {
    using Ret_type = remove_f0_t<function_t<Ret(Args...)> >;
    return parsec::Consumed<Ret_type>(v.index() == parsec::Consumed_ ?
        parsec::c_Consumed<Ret_type>(f << v.consumed().value) :
        parsec::c_Empty<Ret_type>(f << v.empty().value));
}

_FUNCPROG_END
