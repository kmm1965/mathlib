#pragma once

#include "ParsecT.hpp"
#include "../detail/show.hpp"

_PARSEC_BEGIN

/*
-- | @choice ps@ tries to apply the parsers in the list @ps@ in order,
-- until one of them succeeds. Returns the value of the succeeding
-- parser.

choice :: (Stream s m t) => [ParsecT s u m a] -> ParsecT s u m a
choice ps           = foldr (<|>) mzero ps
*/
template<typename S, typename U, typename _M, typename A, typename P>
using choice_unParser_t = parserPlus_unParser<S, U, _M, A, parserZero_unParser<S, U, _M, A>, P >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, A, choice_unParser_t<S, U, _M, A, P> >
choice(List<ParsecT<S, U, _M, A, P> > const& ps){
    return foldr(parserPlus<S, U, _M, A, parserZero_unParser<S, U, _M, A>, P>(), parserZero<S, U, _M, A>(), ps);
}

/*
-- | @option x p@ tries to apply parser @p@. If @p@ fails without
-- consuming input, it returns the value @x@, otherwise the value
-- returned by @p@.
--
-- >  priority  = option 0 (do{ d <- digit
-- >                          ; return (digitToInt d)
-- >                          })

option :: (Stream s m t) => a -> ParsecT s u m a -> ParsecT s u m a
option x p          = p <|> return x
*/
template<typename S, typename U, typename _M, typename A, typename P>
using option_unParser_t = parserPlus_unParser<S, U, _M, A, P, parserReturn_unParser<S, U, _M, A> >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, A, option_unParser_t<S, U, _M, A, P> >
option(A const& x, ParsecT<S, U, _M, A, P> const& p) {
    return p | Monad<_ParsecT<S, U, _M> >::mreturn(x);
}

/*
-- | @optionMaybe p@ tries to apply parser @p@.  If @p@ fails without
-- consuming input, it return 'Nothing', otherwise it returns
-- 'Just' the value returned by @p@.

optionMaybe :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m (Maybe a)
optionMaybe p       = option Nothing (liftM Just p)
*/
template<typename S, typename U, typename _M, typename A, typename P>
using optionMaybe_type = ParsecT < S, U, _M, Maybe<A>,
    option_unParser_t<S, U, _M, Maybe<A>,
    typename Monad<_ParsecT<S, U, _M> >::template liftM_unParser_t<A, P, Maybe<A> >
    >
>;

template<typename S, typename U, typename _M, typename A, typename P>
optionMaybe_type<S, U, _M, A, P> optionMaybe(ParsecT<S, U, _M, A, P> const& p) {
    return option(Nothing<A>(), liftMM<S, U, _M, A, P>(_(Just<A>), p));
}

/*
-- | @optional p@ tries to apply parser @p@.  It will parse @p@ or nothing.
-- It only fails if @p@ fails after consuming input. It discards the result
-- of @p@.

optional :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m ()
optional p          = do{ _ <- p; return ()} <|> return ()
*/
template<typename S, typename U, typename _M, typename A, typename P>
using optional_unParser_t =
    parserPlus_unParser<S, U, _M, EmptyData<A>,
        parserBind_unParser<S, U, _M, A, P,
            EmptyData<A>, parserReturn_unParser<S, U, _M, EmptyData<A> >
        >,
        parserReturn_unParser<S, U, _M, EmptyData<A> >
    >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, EmptyData<A>, optional_unParser_t<S, U, _M, A, P> >
optional(ParsecT<S, U, _M, A, P> const& p) {
    return _do(__unused__, p, return (Monad<_ParsecT<S, U, _M> >::mreturn(EmptyData<A>()));
        ) | Monad<_ParsecT<S, U, _M> >::mreturn(EmptyData<A>());
}

/*
-- | @between open close p@ parses @open@, followed by @p@ and @close@.
-- Returns the value returned by @p@.
--
-- >  braces  = between (symbol "{") (symbol "}")

between :: (Stream s m t) => ParsecT s u m open -> ParsecT s u m close
            -> ParsecT s u m a -> ParsecT s u m a
between open close p
                    = do{ _ <- open; x <- p; _ <- close; return x }
*/
template<typename S, typename U, typename _M, typename A, typename P, typename O, typename PO, typename C, typename PC>
using between_unParser_t =
    parserBind_unParser<S, U, _M, O, PO,
        A, parserBind_unParser<S, U, _M, A, P,
            A, parserBind_unParser<S, U, _M, C, PC,
                A, parserReturn_unParser<S, U, _M, A>
            >
        >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename O, typename PO, typename C, typename PC>
ParsecT<S, U, _M, A, between_unParser_t<S, U, _M, A, P, O, PO, C, PC> >
between(ParsecT<S, U, _M, O, PO> const& open, ParsecT<S, U, _M, C, PC> const& close, ParsecT<S, U, _M, A, P> const& p) {
    return _do3(__unused__, open, x, p, __unused2__, close, return (Monad<_ParsecT<S, U, _M> >::mreturn(x)););
}

/*
-- | @skipMany1 p@ applies the parser @p@ /one/ or more times, skipping
-- its result.

skipMany1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m ()
skipMany1 p         = do{ _ <- p; skipMany p }
{-
skipMany p          = scan
                    where
                      scan  = do{ p; scan } <|> return ()
-}
*/
template<typename S, typename U, typename _M, typename A, typename P>
using skipMany1_unParser_t =
    parserBind_unParser<S, U, _M, A, P, EmptyData<A>, skipMany_unParser_t<S, U, _M, A, P> >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, EmptyData<A>, skipMany1_unParser_t<S, U, _M, A, P> >
skipMany1(ParsecT<S, U, _M, A, P> const& p) {
    return _do(__unused__, p, return skipMany(p););
}
/*
-- | @many1 p@ applies the parser @p@ /one/ or more times. Returns a
-- list of the returned values of @p@.
--
-- >  word  = many1 letter

many1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m [a]
many1 p             = do{ x <- p; xs <- many p; return (x:xs) }
*/
template<typename S, typename U, typename _M, typename A, typename P>
using many1_unParser_t = seq_exec_unParser_t<S, U, _M, A, P, List<A>, many_unParser_t<S, U, _M, A, P> >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, List<A>, many1_unParser_t<S, U, _M, A, P> >
many1(ParsecT<S, U, _M, A, P> const& p) {
    return _do2(x, p, xs, many(p), return (Monad<_ParsecT<S, U, _M> >::mreturn(x >> xs)););
}

/*
-- | @sepBy1 p sep@ parses /one/ or more occurrences of @p@, separated
-- by @sep@. Returns a list of values returned by @p@.

sepBy1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
sepBy1 p sep        = do{ x <- p
                        ; xs <- many (sep >> p)
                        ; return (x:xs)
                        }
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
using sepBy1_unParser_t =
    seq_exec_unParser_t<S, U, _M, A, P,
        List<A>, many_unParser_t<S, U, _M, A,
            seq_exec_unParser_t<S, U, _M, SEP, PSEP, A, P>
        >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, sepBy1_unParser_t<S, U, _M, A, P, SEP, PSEP> >
sepBy1(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return _do2(x, p, xs, many(sep >> p), return (Monad<_ParsecT<S, U, _M> >::mreturn(x >> xs)););
}

/*
-- | @sepBy p sep@ parses /zero/ or more occurrences of @p@, separated
-- by @sep@. Returns a list of values returned by @p@.
--
-- >  commaSep p  = p `sepBy` (symbol ",")

sepBy :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
sepBy p sep         = sepBy1 p sep <|> return []
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
using sepBy_unParser_t =
    parserPlus_unParser<S, U, _M, List<A>,
        sepBy1_unParser_t<S, U, _M, A, P, SEP, PSEP>,
        parserReturn_unParser<S, U, _M, List<A> >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, sepBy_unParser_t<S, U, _M, A, P, SEP, PSEP> >
sepBy(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return sepBy1(p, sep) | (Monad<_ParsecT<S, U, _M> >::mreturn(List<A>()));
}

/*
-- | @sepEndBy1 p sep@ parses /one/ or more occurrences of @p@,
-- separated and optionally ended by @sep@. Returns a list of values
-- returned by @p@.

sepEndBy1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
sepEndBy1 p sep     = do{ x <- p
                        ; do{ _ <- sep
                            ; xs <- sepEndBy p sep
                            ; return (x:xs)
                            }
                          <|> return [x]
                        }
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
struct sepEndBy1_unParser;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
using sepEndBy_unParser_t =
    parserPlus_unParser<S, U, _M, List<A>,
        sepEndBy1_unParser<S, U, _M, A, P, SEP, PSEP>,
        parserReturn_unParser<S, U, _M, List<A> >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, sepEndBy_unParser_t<S, U, _M, A, P, SEP, PSEP> >
sepEndBy(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep);

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
struct sepEndBy1_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, List<A> >;

    sepEndBy1_unParser(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) : p(p), sep(sep) {}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(
        State<S, U> const& s,
        typename ParsecT_base_t::template ok_type<B> const& cok,
        typename ParsecT_base_t::template err_type<B> const& cerr,
        typename ParsecT_base_t::template ok_type<B> const& eok,
        typename ParsecT_base_t::template err_type<B> const& eerr) const
    {
        return _do(x, p,
            return _do2(__unused__, sep,
                xs, sepEndBy(p, sep),
                return (Monad<_ParsecT<S, U, _M> >::mreturn(x >> xs));
            ) | (Monad<_ParsecT<S, U, _M> >::mreturn(List<A>(x)));
        ).template run<B>(s, cok, cerr, eok, eerr);
    }

private:
    const ParsecT<S, U, _M, A, P> p;
    const ParsecT<S, U, _M, SEP, PSEP> sep;
};

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, sepEndBy1_unParser<S, U, _M, A, P, SEP, PSEP> >
sepEndBy1(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return sepEndBy1_unParser<S, U, _M, A, P, SEP, PSEP>(p, sep);
}

/*
-- | @sepEndBy p sep@ parses /zero/ or more occurrences of @p@,
-- separated and optionally ended by @sep@, ie. haskell style
-- statements. Returns a list of values returned by @p@.
--
-- >  haskellStatements  = haskellStatement `sepEndBy` semi

sepEndBy :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
sepEndBy p sep      = sepEndBy1 p sep <|> return []
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, sepEndBy_unParser_t<S, U, _M, A, P, SEP, PSEP> >
sepEndBy(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return sepEndBy1(p, sep) | Monad<_ParsecT<S, U, _M> >::mreturn(List<A>());
}

/*
-- | @endBy1 p sep@ parses /one/ or more occurrences of @p@, separated
-- and ended by @sep@. Returns a list of values returned by @p@.

endBy1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
endBy1 p sep        = many1 (do{ x <- p; _ <- sep; return x })
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
using endBy1_unParser_t =
    many1_unParser_t<S, U, _M, A,
        parserBind_unParser<S, U, _M, A, P,
            SEP, parserBind_unParser<S, U, _M, SEP, PSEP,
                A, parserReturn_unParser<S, U, _M, A>
            >
        >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, endBy1_unParser_t<S, U, _M, A, P, SEP, PSEP> >
endBy1(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return many1(_do2(x, p, __unused__, sep, return (Monad<_ParsecT<S, U, _M> >::mreturn(x));));
}

/*
-- | @endBy p sep@ parses /zero/ or more occurrences of @p@, separated
-- and ended by @sep@. Returns a list of values returned by @p@.
--
-- >   cStatements  = cStatement `endBy` semi

endBy :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m sep -> ParsecT s u m [a]
endBy p sep         = many (do{ x <- p; _ <- sep; return x })
*/
template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
using endBy_unParser_t =
    many_unParser_t<S, U, _M, A,
        parserBind_unParser<S, U, _M, A, P,
            SEP, parserBind_unParser<S, U, _M, SEP, PSEP,
                A, parserReturn_unParser<S, U, _M, A>
            >
        >
    >;

template<typename S, typename U, typename _M, typename A, typename P, typename SEP, typename PSEP>
ParsecT<S, U, _M, List<A>, endBy_unParser_t<S, U, _M, A, P, SEP, PSEP> >
endBy(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, SEP, PSEP> const& sep) {
    return many(_do2(x, p, __unused__, sep, return (Monad<_ParsecT<S, U, _M> >::mreturn(x));));
}

/*
-- | @count n p@ parses @n@ occurrences of @p@. If @n@ is smaller or
-- equal to zero, the parser equals to @return []@. Returns a list of
-- @n@ values returned by @p@.

count :: (Stream s m t) => Int -> ParsecT s u m a -> ParsecT s u m [a]
count n p           | n <= 0    = return []
                    | otherwise = sequence (replicate n p)
*/
template<typename S, typename U, typename _M, typename A>
using count_unParser_t = parserReturn_unParser<S, U, _M, List<A> >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, List<A>, count_unParser_t<S, U, _M, A> >
count(int n, ParsecT<S, U, _M, A, P> const& p){
    return n <= 0 ? _PARSECT(S, U, _M)::mreturn(List<A>()) : sequence(replicate(n, p));
}

/*
-- | @chainr1 p op x@ parses /one/ or more occurrences of |p|,
-- separated by @op@ Returns a value obtained by a /right/ associative
-- application of all functions returned by @op@ to the values returned
-- by @p@.

chainr1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m (a -> a -> a) -> ParsecT s u m a
chainr1 p op        = scan
                    where
                      scan      = do{ x <- p; rest x }

                      rest x    = do{ f <- op
                                    ; y <- scan
                                    ; return (f x y)
                                    }
                                <|> return x
*/
template<typename S, typename U, typename _M, typename A, typename P, typename POP>
struct chainr1_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;
    using bin_function_t = function_t<A(A const&, A const&)>;

    chainr1_unParser(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, bin_function_t, POP> const& op) : p(p), op(op) {}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(
        State<S, U> const& s,
        typename ParsecT_base_t::template ok_type<B> const& cok,
        typename ParsecT_base_t::template err_type<B> const& cerr,
        typename ParsecT_base_t::template ok_type<B> const& eok,
        typename ParsecT_base_t::template err_type<B> const& eerr) const
    {
        return _do(x, p,
            return _do2(f, op,
                y, PARSECT(S, U, _M, A, chainr1_unParser)(*this),
                return _PARSECT(S, U, _M)::mreturn(f(x, y));
            ) | _PARSECT(S, U, _M)::mreturn(x);
        ).template run<B>(s, cok, cerr, eok, eerr);
    }

private:
    const ParsecT<S, U, _M, A, P> p;
    const ParsecT<S, U, _M, bin_function_t, POP> op;
};

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
ParsecT<S, U, _M, A, chainr1_unParser<S, U, _M, A, P, POP> >
chainr1(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, function_t<A(A const&, A const&)>, POP> const& op){
    return chainr1_unParser<S, U, _M, A, P, POP>(p, op);
}

/*
-- | @chainr p op x@ parses /zero/ or more occurrences of @p@,
-- separated by @op@ Returns a value obtained by a /right/ associative
-- application of all functions returned by @op@ to the values returned
-- by @p@. If there are no occurrences of @p@, the value @x@ is
-- returned.

chainr :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m (a -> a -> a) -> a -> ParsecT s u m a
chainr p op x       = chainr1 p op <|> return x
*/
template<typename S, typename U, typename _M, typename A, typename P, typename POP>
using chainr_unParser_t =
    parserPlus_unParser<S, U, _M, A,
        chainr1_unParser<S, U, _M, A, P, POP>,
        parserReturn_unParser<S, U, _M, A> >;

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
ParsecT<S, U, _M, A, chainr_unParser_t<S, U, _M, A, P, POP> >
chainr(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, function_t<A(A const&, A const&)>, POP> const& op, A const& x) {
    return chainr1(p, op) | _PARSECT(S, U, _M)::mreturn(x);
}

/*
-- | @chainl p op x@ parses /zero/ or more occurrences of @p@,
-- separated by @op@. Returns a value obtained by a /left/ associative
-- application of all functions returned by @op@ to the values returned
-- by @p@. If there are zero occurrences of @p@, the value @x@ is
-- returned.

-- | @chainl1 p op@ parses /one/ or more occurrences of @p@,
-- separated by @op@ Returns a value obtained by a /left/ associative
-- application of all functions returned by @op@ to the values returned
-- by @p@. This parser can for example be used to eliminate left
-- recursion which typically occurs in expression grammars.
--
-- >  expr    = term   `chainl1` addop
-- >  term    = factor `chainl1` mulop
-- >  factor  = parens expr <|> integer
-- >
-- >  mulop   =   do{ symbol "*"; return (*)   }
-- >          <|> do{ symbol "/"; return (div) }
-- >
-- >  addop   =   do{ symbol "+"; return (+) }
-- >          <|> do{ symbol "-"; return (-) }

chainl1 :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m (a -> a -> a) -> ParsecT s u m a
chainl1 p op        = do{ x <- p; rest x }
                    where
                      rest x    = do{ f <- op
                                    ; y <- p
                                    ; rest (f x y)
                                    }
                                <|> return x
*/

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
struct chainl1_rest_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;
    using bin_function_t = function_t<A(A const&, A const&)>;

    chainl1_rest_unParser(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, bin_function_t, POP> const& op, A const& x) : p(p), op(op), x(x){}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(
        State<S, U> const& s,
        typename ParsecT_base_t::template ok_type<B> const& cok,
        typename ParsecT_base_t::template err_type<B> const& cerr,
        typename ParsecT_base_t::template ok_type<B> const& eok,
        typename ParsecT_base_t::template err_type<B> const& eerr) const
    {
        return (
            _do2(f, op, y, p,
                return PARSECT(S, U, _M, A, chainl1_rest_unParser)(chainl1_rest_unParser(p, op, f(x, y)));
            ) | (Monad<_ParsecT<S, U, _M> >::mreturn(x))
        ).template run<B>(s, cok, cerr, eok, eerr);
    }

private:
    const ParsecT<S, U, _M, A, P> p;
    const ParsecT<S, U, _M, bin_function_t, POP> op;
    const A x;
};

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
using chainl1_unParser_t = parserBind_unParser<S, U, _M, A, P, A, chainl1_rest_unParser<S, U, _M, A, P, POP> >;

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
ParsecT<S, U, _M, A, chainl1_unParser_t<S, U, _M, A, P, POP> >
chainl1(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, function_t<A(A const&, A const&)>, POP> const& op)
{
    using chainl1_rest_unParser_t = chainl1_rest_unParser<S, U, _M, A, P, POP>;
    return _do(x, p, return PARSECT(S, U, _M, A, chainl1_rest_unParser_t)(chainl1_rest_unParser_t(p, op, x)););
}

/*
chainl :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m (a -> a -> a) -> a -> ParsecT s u m a
chainl p op x       = chainl1 p op <|> return x
*/
template<typename S, typename U, typename _M, typename A, typename P, typename POP>
using chainl_unParser_t =
    parserPlus_unParser<S, U, _M, A,
        chainl1_unParser_t<S, U, _M, A, P, POP>,
        parserReturn_unParser<S, U, _M, A> >;

template<typename S, typename U, typename _M, typename A, typename P, typename POP>
ParsecT<S, U, _M, A, chainl_unParser_t<S, U, _M, A, P, POP> >
chainl(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, function_t<A(A const&, A const&)>, POP> const& op, A const& x) {
    return chainl1(p, op) | Monad<_ParsecT<S, U, _M> >::mreturn(x);
}

/*
-----------------------------------------------------------
-- Tricky combinators
-----------------------------------------------------------
-- | The parser @anyToken@ accepts any kind of token. It is for example
-- used to implement 'eof'. Returns the accepted token.

anyToken :: (Stream s m t, Show t) => ParsecT s u m t
anyToken            = tokenPrim show (\pos _tok _toks -> pos) Just
*/
/*
    using showToken_type = function_t<std::string(const T&)>;
    using nextpos_type = function_t<SourcePos(SourcePos const&, T const&, S const&)>;
    using test_type = function_t<Maybe<A>(T const&)>;
*/
template<typename U, typename _M, typename T>
using anyToken_unParser_t = tokenPrimEx_unParser<U, _M, T, T>;

template<typename U, typename _M, typename T>
ParsecT<List<T>, U, _M, T, anyToken_unParser_t<U, _M, T> >
anyToken() {
    return tokenPrim<U, _M, T, T>(show<T>, _([](SourcePos const& pos, T const& _tok, List<T> const& _toks) { return pos; }), _(Just<T>));
}

/*
-- | @notFollowedBy p@ only succeeds when parser @p@ fails. This parser
-- does not consume any input. This parser can be used to implement the
-- \'longest match\' rule. For example, when recognizing keywords (for
-- example @let@), we want to make sure that a keyword is not followed
-- by a legal identifier character, in which case the keyword is
-- actually an identifier (for example @lets@). We can program this
-- behaviour as follows:
--
-- >  keywordLet  = try (do{ string "let"
-- >                       ; notFollowedBy alphaNum
-- >                       })
--
-- __NOTE__: Currently, 'notFollowedBy' exhibits surprising behaviour
-- when applied to a parser @p@ that doesn't consume any input;
-- specifically
--
--  - @'notFollowedBy' . 'notFollowedBy'@ is /not/ equivalent to 'lookAhead', and
--
--  - @'notFollowedBy' 'eof'@ /never/ fails.
--
-- See [haskell/parsec#8](https://github.com/haskell/parsec/issues/8)
-- for more details.

notFollowedBy :: (Stream s m t, Show a) => ParsecT s u m a -> ParsecT s u m ()
notFollowedBy p     = try (do{ c <- try p; unexpected (show c) }
                           <|> return ()
                          )
*/
template<typename S, typename U, typename _M, typename A, typename P>
using notFollowedBy_unParser_t =
    try_unParser<S, U, _M, EmptyData<A>,
        parserPlus_unParser<S, U, _M, EmptyData<A>,
            parserBind_unParser<S, U, _M,
                A, try_unParser<S, U, _M, A, P>,
                EmptyData<A>, unexpected_unParser<S, U, _M, EmptyData<A> >
            >,
            parserReturn_unParser<S, U, _M, EmptyData<A> >
        >
    >;

template<typename S, typename U, typename _M, typename A, typename P>
ParsecT<S, U, _M, EmptyData<A>, notFollowedBy_unParser_t<S, U, _M, A, P> >
notFollowedBy(ParsecT<S, U, _M, A, P> const& p)
{
    const auto unexp = unexpected<S, U, _M, EmptyData<A> >;
    return _try_(_do(c, _try_(p), return unexp(show(c));) | Monad<_ParsecT<S, U, _M> >::template mreturn(EmptyData<A>()));
}

/*
-- | This parser only succeeds at the end of the input. This is not a
-- primitive parser but it is defined using 'notFollowedBy'.
--
-- >  eof  = notFollowedBy anyToken <?> "end of input"

eof :: (Stream s m t, Show t) => ParsecT s u m ()
eof                 = notFollowedBy anyToken <?> "end of input"
*/
template<typename U, typename _M, typename T>
using eof_unParser_t =
    labels_unParser<List<T>, U, _M, EmptyData<T>,
        notFollowedBy_unParser_t<List<T>, U, _M, T,
            anyToken_unParser_t<U, _M, T>
        >
    >;

template<typename U, typename _M, typename T>
ParsecT<List<T>, U, _M, EmptyData<T>, eof_unParser_t<U, _M, T> >
eof() {
    return notFollowedBy(anyToken<U, _M, T>()) & "end of input";
}

/*
-- | @manyTill p end@ applies parser @p@ /zero/ or more times until
-- parser @end@ succeeds. Returns the list of values returned by @p@.
-- This parser can be used to scan comments:
--
-- >  simpleComment   = do{ string "<!--"
-- >                      ; manyTill anyChar (try (string "-->"))
-- >                      }
--
--    Note the overlapping parsers @anyChar@ and @string \"-->\"@, and
--    therefore the use of the 'try' combinator.

manyTill :: (Stream s m t) => ParsecT s u m a -> ParsecT s u m end -> ParsecT s u m [a]
manyTill p end      = scan
                    where
                      scan  = do{ _ <- end; return [] }
                            <|>
                              do{ x <- p; xs <- scan; return (x:xs) }
*/
template<typename S, typename U, typename _M, typename A, typename P, typename E, typename PE>
struct manyTill_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, List<A> >;

    manyTill_unParser(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, E, PE> const& end) : p(p), end(end) {}

    template<typename B>
    typename ParsecT_base_t::template return_type<B> run(
        State<S, U> const& s,
        typename ParsecT_base_t::template ok_type<B> const& cok,
        typename ParsecT_base_t::template err_type<B> const& cerr,
        typename ParsecT_base_t::template ok_type<B> const& eok,
        typename ParsecT_base_t::template err_type<B> const& eerr) const
    {
        ParsecT<S, U, _M, List<A>, manyTill_unParser> scan(*this);
        return (
            _do(__unused__, end, return (Monad<_ParsecT<S, U, _M> >::mreturn(List<A>()));) |
            _do2(x, p, xs, scan, return (Monad<_ParsecT<S, U, _M> >::mreturn(x >> xs));)
        ).template run<B>(s, cok, cerr, eok, eerr);
    }

private:
    const ParsecT<S, U, _M, A, P> p;
    const ParsecT<S, U, _M, E, PE> end;
};

template<typename S, typename U, typename _M, typename A, typename P, typename E, typename PE>
ParsecT<S, U, _M, List<A>, manyTill_unParser<S, U, _M, A, P, E, PE> >
manyTill(ParsecT<S, U, _M, A, P> const& p, ParsecT<S, U, _M, E, PE> const& end){
    return ParsecT<S, U, _M, List<A>, manyTill_unParser<S, U, _M, A, P, E, PE> >(manyTill_unParser<S, U, _M, A, P, E, PE>(p, end));
}

template<typename U, typename _M>
using simpleComment_unParser_t =
    seq_exec_unParser_t<String, U, _M,
        String, tokens_unParser<String, U, _M, char>,
        String, manyTill_unParser<String, U, _M,
            char, anyChar_unParser_t<U, _M>,
            String, try_unParser<String, U, _M, String,
                tokens_unParser<String, U, _M, char>
            >
        >
    >;

template<typename U, typename _M>
ParsecT<String, U, _M, String, simpleComment_unParser_t<U, _M> >
simpleComment() {
    return _string<U, _M>("<!--") >> manyTill(anyChar<U, _M>(), _try_(_string<U, _M>("-->")));
}

/*
-- | @parserTrace label@ is an impure function, implemented with "Debug.Trace" that
-- prints to the console the remaining parser state at the time it is invoked.
-- It is intended to be used for debugging parsers by inspecting their intermediate states.
--
-- > *> parseTest (oneOf "aeiou"  >> parserTrace "label") "atest"
-- > label: "test"
-- > ...
--
-- @since 3.1.12.0
parserTrace :: (Show t, Stream s m t) => String -> ParsecT s u m ()
parserTrace s = pt <|> return ()
    where
        pt = try $ do
           x <- try $ many1 anyToken
           trace (s++": " ++ show x) $ try $ eof
           fail (show x)
*/
/*
-- | @parserTraced label p@ is an impure function, implemented with "Debug.Trace" that
-- prints to the console the remaining parser state at the time it is invoked.
-- It then continues to apply parser @p@, and if @p@ fails will indicate that
-- the label has been backtracked.
-- It is intended to be used for debugging parsers by inspecting their intermediate states.
--
-- > *>  parseTest (oneOf "aeiou"  >> parserTraced "label" (oneOf "nope")) "atest"
-- > label: "test"
-- > label backtracked
-- > parse error at (line 1, column 2):
-- > ...
--
-- @since 3.1.12.0
parserTraced :: (Stream s m t, Show t) => String -> ParsecT s u m b -> ParsecT s u m b
parserTraced s p = do
  parserTrace s
  p <|> trace (s ++ " backtracked") (fail s)
*/

_PARSEC_END
