#pragma once

#include "Parsec.Functor.hpp"

_PARSEC_BEGIN

/*
parserReturn :: a -> ParsecT s u m a
parserReturn x
    = ParsecT $ \s _ _ eok _ ->
      eok x s (unknownError s)
*/
template<typename S, typename U, typename _M, typename A>
struct parserReturn_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    parserReturn_unParser(A const& x) : x(x) {}

    IMPLEMENT_UNPARSER_RUN(return eok(x, s, unknownError(s));)

private:
    const A x;
};

template<typename S, typename U, typename _M, typename A>
ParsecT<S, U, _M, A, parserReturn_unParser<S, U, _M, A> > parserReturn(A const& x) {
    return ParsecT<S, U, _M, A, parserReturn_unParser<S, U, _M, A> >(parserReturn_unParser<S, U, _M, A>(x));
}

/*
parserBind :: ParsecT s u m a -> (a -> ParsecT s u m b) -> ParsecT s u m b
{-# INLINE parserBind #-}
parserBind m k
    = ParsecT $ \s cok cerr eok eerr ->
    let
        -- consumed-okay case for m
        mcok x s err =
            let
                    -- if (k x) consumes, those go straigt up
                    pcok = cok
                    pcerr = cerr

                    -- if (k x) doesn't consume input, but is okay,
                    -- we still return in the consumed continuation
                    peok x s err' = cok x s (mergeError err err')

                    -- if (k x) doesn't consume input, but errors,
                    -- we return the error in the 'consumed-error'
                    -- continuation
                    peerr err' = cerr (mergeError err err')
            in  unParser (k x) s pcok pcerr peok peerr

        -- empty-ok case for m
        meok x s err =
            let
                -- in these cases, (k x) can return as empty
                pcok = cok
                peok x s err' = eok x s (mergeError err err')
                pcerr = cerr
                peerr err' = eerr (mergeError err err')
            in  unParser (k x) s pcok pcerr peok peerr
        -- consumed-error case for m
        mcerr = cerr

        -- empty-error case for m
        meerr = eerr

    in unParser m s mcok mcerr meok meerr
*/
template<typename S, typename U, typename _M, typename A, typename PA, typename B, typename PB>
struct parserBind_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, B>;
    using ParsecT_A_base_t = ParsecT_base<S, U, _M, A>;
    using State_t = State<S, U>;

    parserBind_unParser(ParsecT<S, U, _M, A, PA> const& m, function_t<ParsecT<S, U, _M, B, PB>(A const&)> const& k) : m(m), k(k) {}

    template<typename BB>
    typename ParsecT_base_t::template return_type<BB> run(
        const State_t &s,
        typename ParsecT_base_t::template ok_type<BB> const& cok,
        typename ParsecT_base_t::template err_type<BB> const& cerr,
        typename ParsecT_base_t::template ok_type<BB> const& eok,
        typename ParsecT_base_t::template err_type<BB> const& eerr) const
    {
        // consumed-okay case for m
        const typename ParsecT_A_base_t::template ok_type<BB> mcok = [this, &cok, &cerr]
        (A const& x, const State_t &s, ParseError const& err) {
            /*
                let
                        -- if (k x) consumes, those go straigt up
                        pcok = cok
                        pcerr = cerr

                        -- if (k x) doesn't consume input, but is okay,
                        -- we still return in the consumed continuation
                        peok x s err' = cok x s (mergeError err err')

                        -- if (k x) doesn't consume input, but errors,
                        -- we return the error in the 'consumed-error'
                        -- continuation
                        peerr err' = cerr (mergeError err err')
                in  unParser (k x) s pcok pcerr peok peerr
            */
            // if (k x) consumes, those go straigt up
            // if (k x) doesn't consume input, but is okay,
            // we still return in the consumed continuation
            // peok x s err' = cok x s (mergeError err err')
            const typename ParsecT_base_t::template ok_type<BB> peok =
                [cok, err](const B &y, const State_t &s, ParseError const& err_) {
                    return cok(y, s, mergeError(err, err_));
                };

            // if (k x) doesn't consume input, but errors,
            // we return the error in the 'consumed-error' continuation
            // peerr err' = cerr (mergeError err err')
            const typename ParsecT_base_t::template err_type<BB> peerr =
                [cerr, err](ParseError const& err_) {
                    return cerr(mergeError(err, err_));
                };
            return k(x).template run<BB>(s, cok, cerr, peok, peerr);
        };
        // empty-ok case for m
        const typename ParsecT_A_base_t::template ok_type<BB> meok =
            [this, &cok, &cerr, &eok, &eerr](A const& x, const State_t &s, ParseError const& err)
        {
            /*
                let
                    -- in these cases, (k x) can return as empty
                    pcok = cok
                    peok x s err' = eok x s (mergeError err err')
                    pcerr = cerr
                    peerr err' = eerr (mergeError err err')
                in  unParser (k x) s pcok pcerr peok peerr
            */
            // in these cases, (k x) can return as empty
            const typename ParsecT_base_t::template ok_type<BB> peok =
                [eok, err](const B &y, const State_t &s, ParseError const& err_) {
                    return eok(y, s, mergeError(err, err_));
                };
            const typename ParsecT_base_t::template err_type<BB> peerr =
                [eerr, err](ParseError const& err_) {
                    return eerr(mergeError(err, err_));
                };
            return k(x).template run<BB>(s, cok, cerr, peok, peerr);
        };
        return m.template run<BB>(s, mcok, cerr, meok, eerr);
    }

private:
    const ParsecT<S, U, _M, A, PA> m;
    const function_t<ParsecT<S, U, _M, B, PB>(A const&)> k;
};

// parserBind::ParsecT s u m a -> (a->ParsecT s u m b)->ParsecT s u m b
DEFINE_FUNCTION_2(7, PARSECT(T0, T1, T2, T5, PARSERBIND_UNPARSER(T0, T1, T2, T3, T4, T5, T6)),
    parserBind, const PARSECT(T0, T1, T2, T3, T4)&, m, function_t<PARSECT(T0, T1, T2, T5, T6)(T3 const&)> const&, k,
    return PARSECT(T0, T1, T2, T5, PARSERBIND_UNPARSER(T0, T1, T2, T3, T4, T5, T6))(
        PARSERBIND_UNPARSER(T0, T1, T2, T3, T4, T5, T6)(m, k));)

template<typename S, typename U, typename _M, typename A, typename PA, typename B, typename PB>
using seq_exec_unParser_t =
    parserBind_unParser<S, U, _M, A, PA,
        B, typename Monad<_ParsecT<S, U, _M> >::template liftM_unParser_t<B, PB, B>
    >;

template<typename S, typename U, typename _M, typename A, typename PA, typename B, typename PB>
ParsecT<S, U, _M, B, seq_exec_unParser_t<S, U, _M, A, PA, B, PB> >
operator>>(ParsecT<S, U, _M, A, PA> const& p1, ParsecT<S, U, _M, B, PB> const& p2) {
    return _do2(__unused__, p1, x, p2, return (Monad<_ParsecT<S, U, _M> >::mreturn(x)););
}

// p1 *> p2 = p1 `parserBind` const p2
template<typename S, typename U, typename _M, typename A, typename PA, typename B, typename PB>
ParsecT<S, U, _M, B, parserBind_unParser<S, U, _M, A, PA, B, PB> >
operator*=(ParsecT<S, U, _M, A, PA> const& p1, ParsecT<S, U, _M, B, PB> const& p2) {
    return parserBind<S, U, _M, A, PA, B, PB>(p1, _const_<A>(p2));
}

_PARSEC_END

_FUNCPROG_BEGIN

// Applicative
template<typename S, typename U, typename _M>
struct _is_applicative<parsec::_ParsecT<S, U, _M> > : _is_applicative<_M> {};

template<typename S, typename U, typename _M>
struct Applicative<parsec::_ParsecT<S, U, _M> > : Functor<parsec::_ParsecT<S, U, _M> >
{
    using super = Functor<parsec::_ParsecT<S, U, _M> >;

    template<typename T>
    static parsec::ParsecT<S, U, _M, T, parsec::parserReturn_unParser<S, U, _M, T> > pure(T const& x) {
        return parsec::parserReturn<S, U, _M>(x);
    }
};

// Monad
template<typename S, typename U, typename _M>
struct _is_monad<parsec::_ParsecT<S, U, _M> > : _is_monad<_M> {};

template<typename S, typename U, typename _M>
struct Monad<parsec::_ParsecT<S, U, _M> > : Applicative<parsec::_ParsecT<S, U, _M> >
{
    using super = Applicative<parsec::_ParsecT<S, U, _M> >;

    // mreturn == pure
    template<typename T>
    static parsec::ParsecT<S, U, _M, T, parsec::parserReturn_unParser<S, U, _M, T> > mreturn(T const& x) {
        return super::pure(x);
    }
    
    template<typename PA, typename B, typename PB, typename Arg>
    static parsec::ParsecT<S, U, _M, B, parsec::parserBind_unParser<S, U, _M, fdecay<Arg>, PA, B, PB> >
    mbind(parsec::ParsecT<S, U, _M, fdecay<Arg>, PA> const& m, function_t<parsec::ParsecT<S, U, _M, B, PB>(Arg)> const& k) {
        return parsec::parserBind<S, U, _M, fdecay<Arg>, PA, B, PB>(m, k);
    }

    template<typename A, typename PA, typename B>
    using liftM_unParser_t = parsec::parserBind_unParser<S, U, _M, A, PA, B, parsec::parserReturn_unParser<S, U, _M, B> >;

    template<typename A, typename PA, typename B>
    using liftM_type = parsec::ParsecT<S, U, _M, B, liftM_unParser_t<A, PA, B> >;
};

template<typename S, typename U, typename _M, typename PA, typename B, typename PB, typename Arg>
parsec::ParsecT<S, U, _M, B, parsec::parserBind_unParser<S, U, _M, fdecay<Arg>, PA, B, PB> >
operator>>=(parsec::ParsecT<S, U, _M, fdecay<Arg>, PA> const& m, function_t<parsec::ParsecT<S, U, _M, B, PB>(Arg)> const& k){
    return Monad<parsec::_ParsecT<S, U, _M> >::mbind(m, k);
}

template<typename S, typename U, typename _M, typename PA, typename B, typename PB, typename Arg>
parsec::ParsecT<S, U, _M, B, parsec::parserBind_unParser<S, U, _M, fdecay<Arg>, PA, B, PB> >
operator<<=(function_t<parsec::ParsecT<S, U, _M, B, PB>(Arg)> const& k, parsec::ParsecT<S, U, _M, fdecay<Arg>, PA> const& m){
    return m >>= k;
}

template<typename S, typename U, typename _M, typename A, typename PA, typename B>
using parsec_liftM_type = typename Monad<parsec::_ParsecT<S, U, _M> >::template liftM_type<A, PA, B>;

#define PARSEC_LIFTM_TYPE_(S, U, _M, A, PA, B) BOOST_IDENTITY_TYPE((parsec_liftM_type<S, U, _M, A, PA, B>))
#define PARSEC_LIFTM_TYPE(S, U, _M, A, PA, B) typename PARSEC_LIFTM_TYPE_(S, U, _M, A, PA, B)

DEFINE_FUNCTION_2(6, PARSEC_LIFTM_TYPE(T0, T1, T2, T3, T4, T5), liftMM, function_t<T5(T3 const&)> const&, f,
    PARSECT(T0, T1, T2, T3, T4) const&, m,
    return _do(x, m, return (Monad<parsec::_ParsecT<T0, T1, T2> >::mreturn(f(x))););)

_FUNCPROG_END

