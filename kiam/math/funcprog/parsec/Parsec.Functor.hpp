#pragma once

_PARSEC_BEGIN

/*
parsecMap :: (a -> b) -> ParsecT s u m a -> ParsecT s u m b
parsecMap f p
    = ParsecT $ \s cok cerr eok eerr ->
      unParser p s (cok . f) cerr (eok . f) eerr
*/
template<typename S, typename U, typename _M, typename P, typename Ret, typename Arg, typename... Args>
struct parsecMap_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, remove_f0_t<function_t<Ret(Args...)> > >;

    parsecMap_unParser(function_t<Ret(Arg, Args...)> const& f, ParsecT<S, U, _M, fdecay<Arg>, P> const& p) : f(f), p(p){}

    IMPLEMENT_UNPARSER_RUN(return p.template run<B>(s, cok & f, cerr, eok & f, eerr);)

private:
    function_t<Ret(Arg, Args...)> const f;
    ParsecT<S, U, _M, fdecay<Arg>, P> const p;
};

#define PARSECMAP_UNPARSER_(S, U, _M, P, Ret, Arg) BOOST_IDENTITY_TYPE((_PARSEC::parsecMap_unParser<S, U, _M, P, Ret, Arg, Args...>))
#define PARSECMAP_UNPARSER(S, U, _M, P, Ret, Arg) typename PARSECMAP_UNPARSER_(S, U, _M, P, Ret, Arg)

DECLARE_FUNCTION_2_ARGS(6, PARSECT(T0, T1, T2, remove_f0_t<function_t<T4(Args...)> >, PARSECMAP_UNPARSER(T0, T1, T2, T3, T4, T5)), parsecMap,
    function_t<T4(T5, Args...)> const&, PARSECT(T0, T1, T2, fdecay<T5>, T3) const&)
FUNCTION_TEMPLATE_ARGS(6) constexpr PARSECT(T0, T1, T2, remove_f0_t<function_t<T4(Args...)> >, PARSECMAP_UNPARSER(T0, T1, T2, T3, T4, T5)) parsecMap(
    function_t<T4(T5, Args...)> const& f, PARSECT(T0, T1, T2, fdecay<T5>, T3) const& p)
{
    return PARSECMAP_UNPARSER(T0, T1, T2, T3, T4, T5)(f, p);
}

template<typename S, typename U, typename _M, typename P, typename Ret, typename Arg, typename... Args>
using parsec_fmap_type = ParsecT<S, U, _M, remove_f0_t<function_t<Ret(Args...)> >,
    parsecMap_unParser<S, U, _M, P, Ret, Arg, Args...> >;

template<typename S, typename U, typename _M, typename P, typename Ret, typename Arg, typename... Args>
constexpr parsec_fmap_type<S, U, _M, P, Ret, Arg, Args...> fmap(function_t<Ret(Arg, Args...)> const& f,
    ParsecT<S, U, _M, fdecay<Arg>, P> const& p){
    return Functor<_ParsecT<S, U, _M> >::fmap(f, p);
}

template<typename S, typename U, typename _M, typename P, typename Ret, typename Arg, typename... Args>
constexpr parsec_fmap_type<S, U, _M, P, Ret, Arg, Args...> operator/(function_t<Ret(Arg, Args...)> const& f,
    ParsecT<S, U, _M, fdecay<Arg>, P> const& p){
    return fmap<S, U, _M>(f, p);
}

// liftA == fmap
template<typename S, typename U, typename _M, typename P, typename Ret, typename Arg, typename... Args>
constexpr parsec_fmap_type<S, U, _M, P, Ret, Arg, Args...> liftA(function_t<Ret(Arg, Args...)> const& f,
    ParsecT<S, U, _M, fdecay<Arg>, P> const& p){
    return fmap<S, U, _M>(f, p);
}

_PARSEC_END

_FUNCPROG_BEGIN

// Functor
template<typename S, typename U, typename _M>
struct _is_functor<parsec::_ParsecT<S, U, _M> > : _is_functor<_M> {};

template<typename S, typename U, typename _M, typename A, typename P>
struct is_parser<parsec::ParsecT<S, U, _M, A, P> > : std::true_type {};

template<typename S, typename U, typename _M>
struct Functor<parsec::_ParsecT<S, U, _M> >
{
    template<typename P, typename Ret, typename Arg, typename... Args>
    static constexpr parsec::ParsecT<S, U, _M, remove_f0_t<function_t<Ret(Args...)> >, parsec::parsecMap_unParser<S, U, _M, P, Ret, Arg, Args...> >
    fmap(function_t<Ret(Arg, Args...)> const& f, parsec::ParsecT<S, U, _M, fdecay<Arg>, P> const& p){
        return parsec::parsecMap<S, U, _M, P, Ret, Arg, Args...>(f, p);
    }
};

_FUNCPROG_END
