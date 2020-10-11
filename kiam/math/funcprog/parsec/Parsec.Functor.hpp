#pragma once

_PARSEC_BEGIN

/*
parsecMap :: (a -> b) -> ParsecT s u m a -> ParsecT s u m b
parsecMap f p
    = ParsecT $ \s cok cerr eok eerr ->
      unParser p s (cok . f) cerr (eok . f) eerr
*/
template<typename S, typename U, typename _M, typename Ret, typename T, typename P>
struct parsecMap_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, Ret>;

    parsecMap_unParser(function_t<Ret(T const&)> const& f, ParsecT<S, U, _M, T, P> const& p) : f(f), p(p) {}

    IMPLEMENT_UNPARSER_RUN(return p.template run<B>(s, cok & f, cerr, eok & f, eerr);)

private:
    function_t<Ret(T)> const f;
    ParsecT<S, U, _M, T, P> const p;
};

#define PARSECMAP_UNPARSER_(S, U, _M, Ret, T, P) BOOST_IDENTITY_TYPE((_PARSEC::parsecMap_unParser<S, U, _M, Ret, T, P>))
#define PARSECMAP_UNPARSER(S, U, _M, Ret, T, P) typename PARSECMAP_UNPARSER_(S, U, _M, Ret, T, P)

DEFINE_FUNCTION_2(6, PARSECT(T0, T1, T2, T3, PARSECMAP_UNPARSER(T0, T1, T2, T3, fdecay<T4>, T5)), parsecMap,
    function_t<T3(T4)> const&, f, PARSECT(T0, T1, T2, fdecay<T4>, T5) const&, p,
    return PARSECT(T0, T1, T2, T3, PARSECMAP_UNPARSER(T0, T1, T2, T3, fdecay<T4>, T5))(PARSECMAP_UNPARSER(T0, T1, T2, T3, fdecay<T4>, T5)(f, p));)

_PARSEC_END

_FUNCPROG_BEGIN

// Functor
template<typename S, typename U, typename _M>
struct _is_functor<parsec::_ParsecT<S, U, _M> > : _is_functor<_M> {};

template<typename S, typename U, typename _M>
struct Functor<parsec::_ParsecT<S, U, _M> >
{
    template<typename Ret, typename Arg, typename P>
    static constexpr auto fmap(function_t<Ret(Arg)> const& f, parsec::ParsecT<S, U, _M, fdecay<Arg>, P> const& p) {
        return parsec::parsecMap<S, U, _M, Ret, fdecay<Arg>, P>(f, p);
    }
};

template<typename S, typename U, typename _M, typename Ret, typename Arg, typename P>
using parsec_fmap_type = parsec::ParsecT<S, U, _M, Ret, parsec::parsecMap_unParser<S, U, _M, Ret, fdecay<Arg>, P> >;

#define PARSEC_FMAP_TYPE_(S, U, _M, Ret, Arg, P) BOOST_IDENTITY_TYPE((parsec_fmap_type<S, U, _M, Ret, Arg, P>))
#define PARSEC_FMAP_TYPE(S, U, _M, Ret, Arg, P) typename PARSEC_FMAP_TYPE_(S, U, _M, Ret, Arg, P)

DEFINE_FUNCTION_2(6, PARSEC_FMAP_TYPE(T0, T1, T2, T3, T4, T5), fmap, function_t<T3(T4)> const&, f,
    PARSECT(T0, T1, T2, fdecay<T4>, T5) const&, p, return (Functor<parsec::_ParsecT<T0, T1, T2> >::template fmap<T3, T4, T5>(f, p));)

template<typename S, typename U, typename _M, typename Ret, typename Arg, typename P>
constexpr parsec_fmap_type<S, U, _M, Ret, Arg, P> operator/(function_t<Ret(Arg)> const& f, parsec::ParsecT<S, U, _M, fdecay<Arg>, P> const& p) {
    return fmap<S, U, _M, Ret, Arg, P>(f, p);
}

// liftA == fmap
DEFINE_FUNCTION_2(6, PARSEC_FMAP_TYPE(T0, T1, T2, T3, T4, T5), liftM, function_t<T3(T4)> const&, f,
    PARSECT(T0, T1, T2, fdecay<T4>, T5) const&, p, return (fmap<T0, T1, T2, T3, T4, T5>(f, p));)

_FUNCPROG_END
