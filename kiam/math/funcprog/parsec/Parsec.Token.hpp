#pragma once

_PARSEC_BEGIN

/*
tokenPrimEx :: (Stream s m t)
            => (t -> String)
            -> (SourcePos -> t -> s -> SourcePos)
            -> Maybe (SourcePos -> t -> s -> u -> u)
            -> (t -> Maybe a)
            -> ParsecT s u m a
{-# INLINE tokenPrimEx #-}
tokenPrimEx showToken nextpos Nothing test
  = ParsecT $ \(State input pos user) cok _cerr _eok eerr -> do
      r <- uncons input
      case r of
        Nothing -> eerr $ unexpectError "" pos
        Just (c,cs)
         -> case test c of
              Just x -> let newpos = nextpos pos c cs
                            newstate = State cs newpos user
                        in seq newpos $ seq newstate $
                           cok x newstate (newErrorUnknown newpos)
              Nothing -> eerr $ unexpectError (showToken c) pos
tokenPrimEx showToken nextpos (Just nextState) test
  = ParsecT $ \(State input pos user) cok _cerr _eok eerr -> do
      r <- uncons input
      case r of
        Nothing -> eerr $ unexpectError "" pos
        Just (c,cs)
         -> case test c of
              Just x -> let newpos = nextpos pos c cs
                            newUser = nextState pos c cs user
                            newstate = State cs newpos newUser
                        in seq newpos $ seq newstate $
                           cok x newstate $ newErrorUnknown newpos
              Nothing -> eerr $ unexpectError (showToken c) pos
*/
template<typename U, typename _M, typename A, typename T>
struct tokenPrimEx_unParser
{
    using S = List<T>;
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;
    using State_t = State<S, U>;
    using Stream_t = Stream<_M, T>;
    using pair_type = typename Stream<_M, T>::pair_type;

    using showToken_type = function_t<std::string(const T&)>;
    using nextpos_type = function_t<SourcePos(SourcePos const&, T const&, S const&)>;
    using next_state_function_type = function_t<U(SourcePos const&, T const&, S const&, U const&)>;
    using nextState_type = Maybe<next_state_function_type>;
    using test_type = function_t<Maybe<A>(const T&)>;

    DECLARE_OK_ERR_TYPES();

    tokenPrimEx_unParser(const showToken_type &showToken, const nextpos_type &nextpos, const nextState_type &nextState, const test_type &test) :
        showToken(showToken), nextpos(nextpos), nextState(nextState), test(test) {}

    // showToken nextpos(Just nextState) test
    template<typename B>
    constexpr auto run(State<S, U> const& s, ok_type<B> const& cok, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        return _do(r, Stream_t::uncons(s.input),
            if (!r)
                return eerr(unexpectError("", s.pos));
            pair_type const& p = r.value();
            T const& c = p.first;
            List<T> const& cs = p.second;
            Maybe<A> const ma = test(c);
            if (!ma)
                return eerr(unexpectError(showToken(c), s.pos));
            A const& x = ma.value();
            SourcePos const newpos = nextpos(s.pos, c, cs);
            State_t const newstate(cs, newpos, nextState ? nextState.value()(s.pos, c, cs, s.user) : s.user);
            return cok(x, newstate, newErrorUnknown(newpos)););
    }

private:
    const showToken_type showToken;
    const nextpos_type nextpos;
    const nextState_type nextState;
    const test_type test;
};

template<typename U, typename _M, typename A, typename T>
constexpr auto tokenPrimEx(
    const typename tokenPrimEx_unParser<U, _M, A, T>::showToken_type &showToken,
    const typename tokenPrimEx_unParser<U, _M, A, T>::nextpos_type &nextpos,
    const typename tokenPrimEx_unParser<U, _M, A, T>::nextState_type &nextState,
    const typename tokenPrimEx_unParser<U, _M, A, T>::test_type &test)
{
    return ParsecT<List<T>, U, _M, A, tokenPrimEx_unParser<U, _M, A, T> >(
        tokenPrimEx_unParser<U, _M, A, T>(showToken, nextpos, nextState, test));
}

/*
-- | The parser @tokenPrim showTok nextPos testTok@ accepts a token @t@
-- with result @x@ when the function @testTok t@ returns @'Just' x@. The
-- token can be shown using @showTok t@. The position of the /next/
-- token should be returned when @nextPos@ is called with the current
-- source position @pos@, the current token @t@ and the rest of the
-- tokens @toks@, @nextPos pos t toks@.
--
-- This is the most primitive combinator for accepting tokens. For
-- example, the 'Text.Parsec.Char.char' parser could be implemented as:
--
-- >  char c
-- >    = tokenPrim showChar nextPos testChar
-- >    where
-- >      showChar x        = "'" ++ x ++ "'"
-- >      testChar x        = if x == c then Just x else Nothing
-- >      nextPos pos x xs  = updatePosChar pos x

tokenPrim :: (Stream s m t)
          => (t -> String)                      -- ^ Token pretty-printing function.
          -> (SourcePos -> t -> s -> SourcePos) -- ^ Next position calculating function.
          -> (t -> Maybe a)                     -- ^ Matching function for the token to parse.
          -> ParsecT s u m a
{-# INLINE tokenPrim #-}
tokenPrim showToken nextpos test = tokenPrimEx showToken nextpos Nothing test
*/

template<typename U, typename _M, typename A, typename T>
constexpr auto tokenPrim(
    const typename tokenPrimEx_unParser<U, _M, A, T>::showToken_type &showToken,
    const typename tokenPrimEx_unParser<U, _M, A, T>::nextpos_type &nextpos,
    const typename tokenPrimEx_unParser<U, _M, A, T>::test_type &test)
{
    return tokenPrimEx<U, _M, A, T>(showToken, nextpos, Nothing<typename tokenPrimEx_unParser<U, _M, A, T>::next_state_function_type>(), test);
}

/*
-- | The parser @token showTok posFromTok testTok@ accepts a token @t@
-- with result @x@ when the function @testTok t@ returns @'Just' x@. The
-- source position of the @t@ should be returned by @posFromTok t@ and
-- the token can be shown using @showTok t@.
--
-- This combinator is expressed in terms of 'tokenPrim'.
-- It is used to accept user defined token streams. For example,
-- suppose that we have a stream of basic tokens tupled with source
-- positions. We can then define a parser that accepts single tokens as:
--
-- >  mytoken x
-- >    = token showTok posFromTok testTok
-- >    where
-- >      showTok (pos,t)     = show t
-- >      posFromTok (pos,t)  = pos
-- >      testTok (pos,t)     = if x == t then Just t else Nothing

token :: (Stream s Identity t)
      => (t -> String)            -- ^ Token pretty-printing function.
      -> (t -> SourcePos)         -- ^ Computes the position of a token.
      -> (t -> Maybe a)           -- ^ Matching function for the token to parse.
      -> Parsec s u a
token showToken tokpos test = tokenPrim showToken nextpos test
    where
        nextpos _ tok ts = case runIdentity (uncons ts) of
                             Nothing -> tokpos tok
                             Just (tok',_) -> tokpos tok'
*/
template<typename U, typename _M, typename A, typename T>
constexpr Parsec<List<T>, U, A, tokenPrimEx_unParser<U, Identity<T>, A, T> >
token(
    typename tokenPrimEx_unParser<U, _M, A, T>::showToken_type const& showToken,
    function_t<SourcePos(const T&)> const& tokpos,
    typename tokenPrimEx_unParser<U, _M, A, T>::test_type const& test)
{
    using S = Stream<_Identity, T>;
    const typename tokenPrimEx_unParser<U, _M, A, T>::nextpos_type nextpos =
        [tokpos](SourcePos const&, T const& tok, typename S::stream_type const& ts)
    {
        const Maybe<typename S::pair_type> r = uncons(ts).run();
        return tokpos(r ? r.value().first : tok);
    };
    return tokenPrim<U, _M, A, T>(showToken, nextpos, test);
}

template<typename S, typename U, typename _M, typename T>
struct tokens_unParser
{
    using A = List<T>;
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    using showTokens_type = function_t<std::string(List<T> const&)>;
    using nextposs_type = function_t<SourcePos(SourcePos const&, List<T> const&)>;

    DECLARE_OK_ERR_TYPES();

    tokens_unParser(showTokens_type const& showTokens, nextposs_type const& nextposs, List<T> const& tts) :
        showTokens(showTokens), nextposs(nextposs), tts(tts) {}

    template<typename B>
    constexpr auto run(State<S, U> const& s, ok_type<B> const& cok, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        /*
        tokens _ _ []
            = ParsecT $ \s _ _ eok _ ->
              eok [] s $ unknownError s
        tokens showTokens nextposs tts@(tok:toks)
            = ParsecT $ \(State input pos u) cok cerr _eok eerr ->
            let
                errEof = (setErrorMessage (Expect (showTokens tts))
                          (newErrorMessage (SysUnExpect "") pos))

                errExpect x = (setErrorMessage (Expect (showTokens tts))
                               (newErrorMessage (SysUnExpect (showTokens [x])) pos))

                walk []     rs = ok rs
                walk (t:ts) rs = do
                  sr <- uncons rs
                  case sr of
                    Nothing                 -> cerr $ errEof
                    Just (x,xs) | t == x    -> walk ts xs
                                | otherwise -> cerr $ errExpect x

                ok rs = let pos' = nextposs pos tts
                            s' = State rs pos' u
                        in cok tts s' (newErrorUnknown pos')
            in do
                sr <- uncons input
                case sr of
                    Nothing         -> eerr $ errEof
                    Just (x,xs)
                        | tok == x  -> walk toks xs
                        | otherwise -> eerr $ errExpect x
        */
        if (null(tts))
            return eok(tts, s, unknownError(s));
        //errEof = (setErrorMessage (Expect (showTokens tts))
        //    (newErrorMessage (SysUnExpect "") pos))
        const f0<ParseError> errEof = [this, &s]() {
            return setErrorMessage(Message(Expect, showTokens(tts)), newErrorMessage(Message(SysUnExpect, ""), s.pos));
        };
        //errExpect x = (setErrorMessage (Expect (showTokens tts))
        //    (newErrorMessage (SysUnExpect (showTokens [x])) pos))
        const function_t<ParseError(const T&)> errExpect = [this, &s](T const& x) {
            return setErrorMessage(Message(Expect, showTokens(tts)), newErrorMessage(Message(SysUnExpect, showTokens(List<T>({ x }))), s.pos));
        };
        //ok rs = let pos' = nextposs pos tts
        //            s' = State rs pos' u
        //        in cok tts s' (newErrorUnknown pos')
        const function_t<typename ParsecT_base_t::template return_type<B>(S const& rs)> ok =
            [this, &s, &cok](S const& rs)
        {
            const SourcePos pos_ = nextposs(s.pos, tts);
            const State<S, U> s_(rs, pos_, s.user);
            return cok(tts, s_, newErrorUnknown(pos_));
        };
        /*
                walk []     rs = ok rs
                walk (t:ts) rs = do
                  sr <- uncons rs
                  case sr of
                    Nothing                 -> cerr $ errEof
                    Just (x,xs) | t == x    -> walk ts xs
                                | otherwise -> cerr $ errExpect x
        */
        const function_t<typename ParsecT_base_t::template return_type<B>(List<T> const&, S const&)> walk =
            [&ok, &cerr, &errEof, &errExpect, &walk](List<T> const& l, S const& rs)
        {
            if (null(l))
                return ok(rs);
            T const& t = head(l);
            const List<T> ts = tail(l);
            const Maybe<pair_t<T, List<T> > > sr = uncons(rs);
            if (!sr) return cerr(errEof());
            auto const& [x, xs] = sr.value();
            return t == x ? walk(ts, xs) : cerr(errExpect(x));
        };
        /*
              do
                sr <- uncons input
                case sr of
                    Nothing         -> eerr $ errEof
                    Just (x,xs)
                        | tok == x  -> walk toks xs
                        | otherwise -> eerr $ errExpect x
        */
        T const& tok = head(tts);
        const List<T> toks = tail(tts);
        const Maybe<pair_t<T, List<T> > > sr = uncons(s.input);
        if (!sr) return eerr(errEof());
        auto const& [x, xs] = sr.value();
        return tok == x ? walk(toks, xs) : eerr(errExpect(x));
    }

private:
    const function_t<std::string(List<T> const&)> showTokens;
    const function_t<SourcePos(SourcePos const&, List<T> const&)> nextposs;
    const List<T> tts;
};

/*
tokens :: (Stream s m t, Eq t)
       => ([t] -> String)      -- Pretty print a list of tokens
       -> (SourcePos -> [t] -> SourcePos)
       -> [t]                  -- List of tokens to parse
       -> ParsecT s u m [t]
tokens showTokens nextposs tts@(tok:toks)
*/
template<typename S, typename U, typename _M, typename T>
constexpr auto tokens(
    const typename tokens_unParser<S, U, _M, T>::showTokens_type &showTokens,
    const typename tokens_unParser<S, U, _M, T>::nextposs_type &nextposs,
    List<T> const& tts)
{
    return ParsecT<S, U, _M, List<T>, tokens_unParser<S, U, _M, T> >(
        tokens_unParser<S, U, _M, T>(showTokens, nextposs, tts));
}

_PARSEC_END
