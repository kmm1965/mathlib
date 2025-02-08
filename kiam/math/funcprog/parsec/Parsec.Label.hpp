#pragma once

_PARSEC_BEGIN

/*
labels :: ParsecT s u m a -> [String] -> ParsecT s u m a
labels p msgs =
    ParsecT $ \s cok cerr eok eerr ->
    let eok' x s' error = eok x s' $ if errorIsUnknown error
                  then error
                  else setExpectErrors error msgs
        eerr' err = eerr $ setExpectErrors err msgs

    in unParser p s cok cerr eok' eerr'

 where
   setExpectErrors err []         = setErrorMessage (Expect "") err
   setExpectErrors err [msg]      = setErrorMessage (Expect msg) err
   setExpectErrors err (msg:msgs)
       = foldr (\msg' err' -> addErrorMessage (Expect msg') err')
         (setErrorMessage (Expect msg) err) msgs
*/
template<typename S, typename U, typename _M, typename A, typename P>
struct labels_unParser
{
    using ParsecT_base_t = ParsecT_base<S, U, _M, A>;

    DECLARE_OK_ERR_TYPES();

    labels_unParser(ParsecT<S, U, _M, A, P> const& p, List<std::string> const& msgs) : p(p), msgs(msgs){}

    template<typename B>
    constexpr auto run(State<S, U> const& s, ok_type<B> const& cok, err_type<B> const& cerr, ok_type<B> const& eok, err_type<B> const& eerr) const
    {
        auto const setExpectErrors = _([](ParseError const& err, List<std::string> const& msgs){
            const int lmsgs = length(msgs);
            if (lmsgs == 0) return setErrorMessage(Message(Expect, ""), err);
            else if (lmsgs == 1) return setErrorMessage(Message(Expect, msgs.front()), err);
            else return foldr(_([](std::string const& msg_, ParseError const& err_){
                    return addErrorMessage(Message(Expect, msg_), err_); }),
                setErrorMessage(Message(Expect, head(msgs)), err), tail(msgs));
        });
        //let eok' x s' error = eok x s' $ if errorIsUnknown error
        //        then error
        //        else setExpectErrors error msgs
        //eerr' err = eerr $ setExpectErrors err msgs
        ok_type<B> const eok_ = [this, &eok, &setExpectErrors](A const& x, State<S, U> const& s_, ParseError const& error){
            return eok(x, s_, errorIsUnknown(error) ? error : setExpectErrors(error, msgs));
        };
        err_type<B> const eerr_ = [this, &eerr, &setExpectErrors](ParseError const& err){
            return eerr(setExpectErrors(err, msgs));
        };
        return p.template run<B>(s, cok, cerr, eok_, eerr_);
    }

private:
    ParsecT<S, U, _M, A, P> const p;
    List<std::string> const msgs;
};

template<typename S, typename U, typename _M, typename A, typename P>
constexpr auto labels(ParsecT<S, U, _M, A, P> const& p, List<std::string> const& msgs){
    return ParsecT<S, U, _M, A, labels_unParser<S, U, _M, A, P> >(labels_unParser<S, U, _M, A, P>(p, msgs));
}

/*
-- | A synonym for @\<?>@, but as a function instead of an operator.
label :: ParsecT s u m a -> String -> ParsecT s u m a
label p msg
  = labels p [msg]
*/
template<typename S, typename U, typename _M, typename A, typename P>
constexpr auto label(ParsecT<S, U, _M, A, P> const& p, std::string const& msg){
    return labels(p, List<std::string>({ msg }));
}

template<typename S, typename U, typename _M, typename A, typename P>
constexpr auto operator&(ParsecT<S, U, _M, A, P> const& p, std::string const& msg){
    return label(p, msg);
}

_PARSEC_END
