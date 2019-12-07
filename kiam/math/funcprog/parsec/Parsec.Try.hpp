#pragma once

_PARSEC_BEGIN

/*
-- | The parser @try p@ behaves like parser @p@, except that it
-- pretends that it hasn't consumed any input when an error occurs.
--
-- This combinator is used whenever arbitrary look ahead is needed.
-- Since it pretends that it hasn't consumed any input when @p@ fails,
-- the ('<|>') combinator will try its second alternative even when the
-- first parser failed while consuming input.
--
-- The @try@ combinator can for example be used to distinguish
-- identifiers and reserved words. Both reserved words and identifiers
-- are a sequence of letters. Whenever we expect a certain reserved
-- word where we can also expect an identifier we have to use the @try@
-- combinator. Suppose we write:
--
-- >  expr        = letExpr <|> identifier <?> "expression"
-- >
-- >  letExpr     = do{ string "let"; ... }
-- >  identifier  = many1 letter
--
-- If the user writes \"lexical\", the parser fails with: @unexpected
-- \'x\', expecting \'t\' in \"let\"@. Indeed, since the ('<|>') combinator
-- only tries alternatives when the first alternative hasn't consumed
-- input, the @identifier@ parser is never tried (because the prefix
-- \"le\" of the @string \"let\"@ parser is already consumed). The
-- right behaviour can be obtained by adding the @try@ combinator:
--
-- >  expr        = letExpr <|> identifier <?> "expression"
-- >
-- >  letExpr     = do{ try (string "let"); ... }
-- >  identifier  = many1 letter

try :: ParsecT s u m a -> ParsecT s u m a
try p =
    ParsecT $ \s cok _ eok eerr ->
    unParser p s cok eerr eok eerr
*/
template<typename S, typename U, typename M, typename A, typename P>
struct try_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, M, A>;

    try_unParser(ParsecT<S, U, M, A, P> const& p) : p(p) {}

    IMPLEMENT_UNPARSER_RUN(return p.template run<B>(s, cok, eerr, eok, eerr);)

private:
    const ParsecT<S, U, M, A, P> p;
};

template<typename S, typename U, typename M, typename A, typename P>
ParsecT<S, U, M, A, try_unParser<S, U, M, A, P> >
_try_(ParsecT<S, U, M, A, P> const& p) {
    return ParsecT<S, U, M, A, try_unParser<S, U, M, A, P> >(try_unParser<S, U, M, A, P>(p));
}

_PARSEC_END
