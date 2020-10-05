#pragma once

#include "Parsec.Setup.h"
#include "Parsec.State.hpp"
#include "Parsec.Error.hpp"
#include "Parsec.Consumed.hpp"
#include "Parsec.Reply.hpp"
#include "Parsec.Stream.hpp"
#include "../Identity.hpp"
#include "../Maybe.hpp"
#include "../Either.hpp"

_PARSEC_BEGIN

struct parse_error : std::runtime_error
{
    parse_error(const char *msg) : std::runtime_error(msg) {}
};

template<typename S, typename U, typename _M, typename A, class P>
struct ParsecT;

#define PARSECT_(S, U, _M, A, P) BOOST_IDENTITY_TYPE((_PARSEC::ParsecT<S, U, _M, A, P>))
#define PARSECT(S, U, _M, A, P) typename PARSECT_(S, U, _M, A, P)

template<typename S, typename U, typename _M, typename A>
struct parserReturn_unParser;

#define PARSERRETURN_UNPARSER_(S, U, _M, A) BOOST_IDENTITY_TYPE((_PARSEC::parserReturn_unParser<S, U, _M, A>))
#define PARSERRETURN_UNPARSER(S, U, _M, A) typename PARSERRETURN_UNPARSER_(S, U, _M, A)

template<typename S, typename U, typename _M, typename A>
ParsecT<S, U, _M, A, parserReturn_unParser<S, U, _M, A> > parserReturn(A const&);

template<typename S, typename U, typename _M, typename A, typename PA, typename B, typename PB>
struct parserBind_unParser;

#define PARSERBIND_UNPARSER_(S, U, _M, A, PA, B, PB) BOOST_IDENTITY_TYPE((_PARSEC::parserBind_unParser<S, U, _M, A, PA, B, PB>))
#define PARSERBIND_UNPARSER(S, U, _M, A, PA, B, PB) typename PARSERBIND_UNPARSER_(S, U, _M, A, PA, B, PB)

DECLARE_FUNCTION_2(7, PARSECT(T0, T1, T2, T5, PARSERBIND_UNPARSER(T0, T1, T2, T3, T4, T5, T6)),
    parserBind, const PARSECT(T0, T1, T2, T3, T4)&, function_t<PARSECT(T0, T1, T2, T5, T6)(const T3&)> const&);

template<typename S, typename U, typename _M, typename A>
struct parserZero_unParser;

template<typename S, typename U, typename _M, typename A>
ParsecT<S, U, _M, A, parserZero_unParser<S, U, _M, A> > parserZero();

template<typename S, typename U, typename _M, typename A, typename PM, typename PN>
struct parserPlus_unParser;

#define PARSERPLUS_UNPARSER_(S, U, _M, A, PM, PN) BOOST_IDENTITY_TYPE((_PARSEC::parserPlus_unParser<S, U, _M, A, PM, PN>))
#define PARSERPLUS_UNPARSER(S, U, _M, A, PM, PN) typename PARSERPLUS_UNPARSER_(S, U, _M, A, PM, PN)

DECLARE_FUNCTION_2(6, PARSECT(T0, T1, T2, T3, PARSERPLUS_UNPARSER(T0, T1, T2, T3, T4, T5)),
    parserPlus, PARSECT(T0, T1, T2, T3, T4) const&, PARSECT(T0, T1, T2, T3, T5) const&);

template<typename S, typename U, typename _M>
struct _ParsecT
{
    static_assert(is_monad<_M>::value, "M should be a monad");
    using base_class = _ParsecT;
    
    template<typename A>
    using type = ParsecT<S, U, _M, A, parserReturn_unParser<S, U, _M, A> >;

    template<typename B>
    using return_type = typename _M::template type<B>;
};

#define _PARSECT_(S, U, _M) BOOST_IDENTITY_TYPE((_PARSEC::_ParsecT<S, U, _M>))
#define _PARSECT(S, U, _M) typename _PARSECT_(S, U, _M)

template<typename S, typename U>
struct __ParsecT
{
    template<typename _M>
    using mt_type = _ParsecT<S, U, _M>;
};

template<typename S, typename U, typename _M, typename A>
struct ParsecT_base : _ParsecT<S, U, _M>
{
    using super = _ParsecT<S, U, _M>;
    using value_type = A;

    template<typename B>
    using ok_type = function_t<typename super::template return_type<B>(A const&, State<S, U> const&, ParseError const&)>;

    template<typename B>
    using err_type = function_t<typename super::template return_type<B>(ParseError const&)>;
};

#define PARSECT_BASE_(S, U, _M, A) BOOST_IDENTITY_TYPE((_PARSEC::ParsecT_base<S, U, _M, A>))
#define PARSECT_BASE(S, U, _M, A) typename PARSECT_BASE_(S, U, _M, A)

template<typename S, typename U, typename _M, typename A, class P>
struct ParsecT : ParsecT_base<S, U, _M, A>
{
    using super = ParsecT_base<S, U, _M, A>;

    ParsecT(P const& unParser) : unParser(unParser) {}

    template<typename B>
    typename super::template return_type<B> run(State<S, U> const& s,
        typename super::template ok_type<B> const& cok,
        typename super::template err_type<B> const& cerr,
        typename super::template ok_type<B> const& eok,
        typename super::template err_type<B> const& eerr) const
    {
        return unParser.template run<B>(s, cok, cerr, eok, eerr);
    }

    /*
    -- | Low-level unpacking of the ParsecT type. To run your parser, please look to
    -- runPT, runP, runParserT, runParser and other such functions.
    runParsecT :: Monad m => ParsecT s u m a -> State s u -> m (Consumed (m (Reply s u a)))
    runParsecT p s = unParser p s cok cerr eok eerr
        where cok a s' err = return . Consumed . return $ Ok a s' err
              cerr err = return . Consumed . return $ Error err
              eok a s' err = return . Empty . return $ Ok a s' err
              eerr err = return . Empty . return $ Error err
    */

    typename _M::template type<Consumed<typename _M::template type<Reply<S, U, A> > > >
    run(State<S, U> const& s) const
    {
        using Reply_t = Reply<S, U, A>;
        using mrep_type = typename _M::template type<Reply_t>;
        using B = Consumed<mrep_type>;
        const typename ParsecT<S, U, _M, A, P>::template ok_type<B>
            cok = [](A const& a, State<S, U> const& s_, ParseError const& err) {
                return Monad<_M>::mreturn(B(c_Consumed<mrep_type>(Monad<_M>::mreturn(Reply_t(c_Ok<S, U, A>(a, s_, err))))));
            },
            eok = [](A const& a, State<S, U> const& s_, ParseError const& err) {
                return Monad<_M>::mreturn(B(c_Empty<mrep_type>(Monad<_M>::mreturn(Reply_t(c_Ok<S, U, A>(a, s_, err))))));
            };
        const typename ParsecT<S, U, _M, A, P>::template err_type<B>
            cerr = [](ParseError const& err) {
                return Monad<_M>::mreturn(B(c_Consumed<mrep_type>(Monad<_M>::mreturn(Reply_t(c_Error(err))))));
            },
            eerr = [](ParseError const& err) {
                return Monad<_M>::mreturn(B(c_Empty<mrep_type>(Monad<_M>::mreturn(Reply_t(c_Error(err))))));
            };
        return unParser.template run<B>(s, cok, cerr, eok, eerr);
    }

private:
    const P unParser;
};

template<typename S, typename U, typename _M, typename A, class P>
ParsecT<S, U, _M, A, P> getParsecT(P const& unParser) {
    return ParsecT<S, U, _M, A, P>(unParser);
}

template<typename S, typename U, typename _M, typename A, typename P>
typename _M::template type<Consumed<typename _M::template type<Reply<S, U, A> > > >
runParsecT(ParsecT<S, U, _M, A, P> const& p, State<S, U> const& s) {
    return p.run(s);
}

template<typename S, typename U, typename A, typename P>
using Parsec = ParsecT<S, U, _Identity, A, P>;

using _FUNCPROG::None;

template<typename A, typename P>
using Parser = Parsec<String, None, A, P>;

#define IMPLEMENT_UNPARSER_RUN(impl) \
    template<typename B> \
    typename ParsecT_base_t::template return_type<B> run( \
        State<S, U> const& s, \
        typename ParsecT_base_t::template ok_type<B> const& cok, \
        typename ParsecT_base_t::template err_type<B> const& cerr, \
        typename ParsecT_base_t::template ok_type<B> const& eok, \
        typename ParsecT_base_t::template err_type<B> const& eerr) const \
    { impl }

/*
-- | The parser @unexpected msg@ always fails with an unexpected error
-- message @msg@ without consuming any input.
--
-- The parsers 'fail', ('<?>') and @unexpected@ are the three parsers
-- used to generate error messages. Of these, only ('<?>') is commonly
-- used. For an example of the use of @unexpected@, see the definition
-- of 'Text.Parsec.Combinator.notFollowedBy'.

unexpected :: (Stream s m t) => String -> ParsecT s u m a
unexpected msg
    = ParsecT $ \s _ _ _ eerr ->
      eerr $ newErrorMessage (UnExpect msg) (statePos s)
*/

template<typename S, typename U, typename _M, typename A>
struct unexpected_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    unexpected_unParser(std::string const& msg) : msg(msg) {}

    IMPLEMENT_UNPARSER_RUN(return eerr(newErrorMessage(Message(UnExpect, msg), s.pos));)

private:
    std::string msg;
};

template<typename S, typename U, typename _M, typename A>
ParsecT<S, U, _M, A, unexpected_unParser<S, U, _M, A> > unexpected(std::string const& msg) {
    return ParsecT<S, U, _M, A, unexpected_unParser<S, U, _M, A> >(unexpected_unParser<S, U, _M, A>(msg));
}

// sysUnExpectError :: String -> SourcePos -> Reply s u a
// sysUnExpectError msg pos  = Error (newErrorMessage (SysUnExpect msg) pos)
template<typename S, typename U, typename A>
Reply<S, U, A> sysUnExpectError(std::string const& msg, SourcePos const& pos) {
    return c_Error(newErrorMessage(Message(SysUnExpect, msg), pos));
}

_PARSEC_END

_FUNCPROG_BEGIN

template<typename S, typename U, typename _M, typename A, typename P1, typename P2>
struct is_same_class<parsec::ParsecT<S, U, _M, A, P1>, parsec::ParsecT<S, U, _M, A, P2> > : std::true_type {};

_FUNCPROG_END

#include "Parsec.MonadPlus.hpp"
#include "Parsec.Run.hpp"
#include "Parsec.Token.hpp"
#include "Parsec.Many.hpp"
#include "Parsec.Try.hpp"
#include "Parsec.LookAhead.hpp"
#include "Parsec.Label.hpp"
#include "Parsec.Char.hpp"
#include "Parsec.Combinator.hpp"
