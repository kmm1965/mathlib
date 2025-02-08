#pragma once

_PARSEC_BEGIN

/*
-- | Low-level creation of the ParsecT type. You really shouldn't have to do this.
mkPT :: Monad m => (State s u -> m (Consumed (m (Reply s u a)))) -> ParsecT s u m a
mkPT k = ParsecT $ \s cok cerr eok eerr -> do
           cons <- k s
           case cons of
             Consumed mrep -> do
                       rep <- mrep
                       case rep of
                         Ok x s' err -> cok x s' err
                         Error err -> cerr err
             Empty mrep -> do
                       rep <- mrep
                       case rep of
                         Ok x s' err -> eok x s' err
                         Error err -> eerr err
*/
template<typename S, typename U, typename _M, typename A>
struct mkPT_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;
    using k_type = function_t<typename _M::template type<Consumed<typename _M::template type<Reply<S, U, A> > > >(State<S, U> const&)>;

    mkPT_unParser(const k_type &k) : k(k){}

    IMPLEMENT_UNPARSER_RUN(
        return _do(cons, k(s),
            return cons.index() == Consumed_ ?
                _do(rep, cons.consumed().value,
                    if (rep.index() == Ok_){
                        const OK(S, U, A) &ok = rep.ok();
                        return cok(ok.value, ok.state, ok.error);
                    } else return cerr(rep.error().error);
                ) : // Empty_
                _do(rep, cons.empty().value,
                    if (rep.index() == Ok_){
                        const OK(S, U, A) &ok = rep.ok();
                        return eok(ok.value, ok.state, ok.error);
                    } else return eerr(rep.error().error);
                );
        );)

private:
    const k_type k;
};

template<typename S, typename U, typename _M, typename A>
constexpr ParsecT<S, U, _M, A, mkPT_unParser<S, U, _M, A> >
mkPT(const typename mkPT_unParser<S, U, _M, A>::k_type &k){
    return mkPT_unParser<S, U, _M, A>(k);
}

/*
-- < Running a parser: monadic (runPT) and pure (runP)

runPT :: (Stream s m t)
      => ParsecT s u m a -> u -> SourceName -> s -> m (Either ParseError a)
runPT p u name s
    = do res <- runParsecT p (State s (initialPos name) u)
         r <- parserReply res
         case r of
           Ok x _ _  -> return (Right x)
           Error err -> return (Left err)
    where
        parserReply res
            = case res of
                Consumed r -> r
                Empty    r -> r
*/
template<typename S, typename U, typename _M, typename A, class P>
constexpr auto runPT(ParsecT<S, U, _M, A, P> const& p, U const& u, SourceName const& name, S const& s)
{
    using Reply_t = Reply<S, U, A>;
    using Either_t = Either<ParseError, A>;
    using mrep_type = typename _M::template type<Reply_t>;
    auto const parserReply = [](Consumed<mrep_type> const& res){
        return res.index() == Consumed_ ? *(res.consumed()) : *(res.empty());
    };
    return _do2(res, p.run(State<S, U>(s, initialPos(name), u)), r, parserReply(res),
        return Monad_t<_M>::mreturn(r.index() == Ok_ ?
            Either_t(_Right<A>(r.ok().value)) :
            Either_t(_Left<ParseError>(r.error().error)));
    );
}

/*
runP :: (Stream s Identity t)
     => Parsec s u a -> u -> SourceName -> s -> Either ParseError a
runP p u name s = runIdentity $ runPT p u name s
*/
template<typename S, typename U, typename A, class P>
constexpr auto runP(Parsec<S, U, A, P> const& p, U const& u, SourceName const& name, S const& s){
    return runPT(p, u, name, s).run();
}

/*
-- | The most general way to run a parser. @runParserT p state filePath
-- input@ runs parser @p@ on the input list of tokens @input@,
-- obtained from source @filePath@ with the initial user state @st@.
-- The @filePath@ is only used in error messages and may be the empty
-- string. Returns a computation in the underlying monad @m@ that return either a 'ParseError' ('Left') or a
-- value of type @a@ ('Right').

runParserT :: (Stream s m t)
           => ParsecT s u m a -> u -> SourceName -> s -> m (Either ParseError a)
runParserT = runPT
*/
template<typename S, typename U, typename _M, typename A, class P>
constexpr auto runParserT(ParsecT<S, U, _M, A, P> const& p, U const& u, SourceName const& name, S const& s){
    return runPT(p, u, name, s);
}

/*
-- | The most general way to run a parser over the Identity monad. @runParser p state filePath
-- input@ runs parser @p@ on the input list of tokens @input@,
-- obtained from source @filePath@ with the initial user state @st@.
-- The @filePath@ is only used in error messages and may be the empty
-- string. Returns either a 'ParseError' ('Left') or a
-- value of type @a@ ('Right').
--
-- >  parseFromFile p fname
-- >    = do{ input <- readFile fname
-- >        ; return (runParser p () fname input)
-- >        }

runParser :: (Stream s Identity t)
          => Parsec s u a -> u -> SourceName -> s -> Either ParseError a
runParser = runP
*/
template<typename S, typename U, typename A, class P>
constexpr auto runParser(Parsec<S, U, A, P> const& p, U const& u, SourceName const& name, S const& s){
    return runP(p, u, name, s);
}

/*
-- | @parse p filePath input@ runs a parser @p@ over Identity without user
-- state. The @filePath@ is only used in error messages and may be the
-- empty string. Returns either a 'ParseError' ('Left')
-- or a value of type @a@ ('Right').
--
-- >  main    = case (parse numbers "" "11, 2, 43") of
-- >             Left err  -> print err
-- >             Right xs  -> print (sum xs)
-- >
-- >  numbers = commaSep integer

parse :: (Stream s Identity t)
      => Parsec s () a -> SourceName -> s -> Either ParseError a
parse p = runP p ()
*/

template<typename S, typename A, class P>
constexpr auto parse(Parsec<S, None, A, P> const& p, SourceName const& name, S const& s){
    return runP(p, None(), name, s);
}

_PARSEC_END
