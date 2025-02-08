#pragma once

_PARSEC_BEGIN

// manyErr :: a
// manyErr = error "Text.ParserCombinators.Parsec.Prim.many: combinator 'many' is applied to a parser that accepts an empty string."
template<typename S, typename U, typename _M, typename A, typename B>
constexpr auto manyErr(){
    return _([](A const&, State<S, U> const&, ParseError const&){
        throw parse_error("Text.ParserCombinators.Parsec.Prim.many: combinator 'many' is applied to a parser that accepts an empty string.");
        return *(typename ParsecT_base<S, U, _M, A>::template return_type<B>*) nullptr;
    });
}

/*
manyAccum :: (a -> [a] -> [a])
          -> ParsecT s u m a
          -> ParsecT s u m [a]
manyAccum acc p =
    ParsecT $ \s cok cerr eok _eerr ->
    let walk xs x s' _err =
            unParser p s'
              (seq xs $ walk $ acc x xs)  -- consumed-ok
              cerr                        -- consumed-err
              manyErr                     -- empty-ok
              (\e -> cok (acc x xs) s' e) -- empty-err
    in unParser p s (walk []) cerr manyErr (\e -> eok [] s e)
*/
template<typename S, typename U, typename _M, typename A, typename P>
struct manyAccum_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, List<A> >;
    using acc_type = function_t<List<A>(A const&, List<A> const&)>;

    DECLARE_OK_ERR_TYPES();

    manyAccum_unParser(const acc_type &acc, ParsecT<S, U, _M, A, P> const& p) : acc(acc), p(p){}

    template<typename B>
    constexpr auto run(State<S, U> const& s, ok_type<B> const& cok, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        function_t<typename ParsecT_base_t::template return_type<B>(
            List<A> const&, A const&, State<S, U> const&, ParseError const&)> const walk =
            [this, &walk, &cok, &cerr](List<A> const& xs, A const& x, State<S, U> const& s_, ParseError const& _err){
            return p.template run<B>(s_,
                walk << acc(x, xs),          // consumed-ok
                cerr,                        // consumed-err
                manyErr<S, U, _M, A, B>(),   // empty-ok
                _([this, &cok, &xs, &x, &s_](ParseError const& e){ return cok(acc(x, xs), s_, e); }));
        };
        return p.template run<B>(s, walk << List<A>(), cerr,
            manyErr<S, U, _M, A, B>(), _([s, eok](ParseError const& e){ return eok(List<A>(), s, e); }));
    }

private:
    acc_type const acc;
    ParsecT<S, U, _M, A, P> const p;
};

template<typename S, typename U, typename _M, typename A, typename P>
constexpr ParsecT<S, U, _M, List<A>, manyAccum_unParser<S, U, _M, A, P> > 
manyAccum(const typename manyAccum_unParser<S, U, _M, A, P>::acc_type &acc, ParsecT<S, U, _M, A, P> const& p){
    return manyAccum_unParser<S, U, _M, A, P>(acc, p);
}

/*
-- | @many p@ applies the parser @p@ /zero/ or more times. Returns a
--    list of the returned values of @p@.
--
-- >  identifier  = do{ c  <- letter
-- >                  ; cs <- many (alphaNum <|> char '_')
-- >                  ; return (c:cs)
-- >                  }

many :: ParsecT s u m a -> ParsecT s u m [a]
many p
  = do xs <- manyAccum (:) p
       return (reverse xs)
*/
template<typename S, typename U, typename _M, typename A, typename P>
constexpr auto many(ParsecT<S, U, _M, A, P> const& p){
    return _do(xs, manyAccum(_(cons<A>), p), return (Monad<_ParsecT<S, U, _M> >::return_(reverse(xs))););
}

/*
-- | @skipMany p@ applies the parser @p@ /zero/ or more times, skipping
-- its result.
--
-- >  spaces  = skipMany space

skipMany :: ParsecT s u m a -> ParsecT s u m ()
skipMany p
  = do _ <- manyAccum (\_ _ -> []) p
       return ()
*/
template<typename S, typename U, typename _M, typename A, typename P>
constexpr auto skipMany(ParsecT<S, U, _M, A, P> const& p)
{
    auto const f = _([](A const&, List<A> const&){ return List<A>(); });
    return _do(_, manyAccum(f, p), return (Monad<_ParsecT<S, U, _M> >::return_(EmptyData<A>())););
}

_PARSEC_END
