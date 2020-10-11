#pragma once

#include "Parsec.Pos.hpp"
#include "Parsec.Message.hpp"
#include "../List.hpp"

_PARSEC_BEGIN

/*
-- | The abstract data type @ParseError@ represents parse errors. It
-- provides the source position ('SourcePos') of the error
-- and a list of error messages ('Message'). A @ParseError@
-- can be returned by the function 'Text.Parsec.Prim.parse'. @ParseError@ is an
-- instance of the 'Show' and 'Eq' classes.

data ParseError = ParseError !SourcePos [Message]
    deriving ( Typeable )
*/

struct ParseError
{
    ParseError(SourcePos const& pos, List<Message> const& msgs = List<Message>()) : pos(pos), msgs(msgs){}

    const SourcePos pos;
    const List<Message> msgs;
};

/*
-- | Extracts the source position from the parse error

errorPos :: ParseError -> SourcePos
errorPos (ParseError pos _msgs)
    = pos

*/
SourcePos errorPos(ParseError const& err) {
    return err.pos;
}

/*
-- | Extracts the list of error messages from the parse error

errorMessages :: ParseError -> [Message]
errorMessages (ParseError _pos msgs)
    = sort msgs
*/
inline List<Message> errorMessages(ParseError const& err) {
    return sort(err.msgs);
}

/*
errorIsUnknown :: ParseError -> Bool
errorIsUnknown (ParseError _pos msgs)
    = null msgs

*/
inline constexpr bool errorIsUnknown(ParseError const& err) {
    return null(err.msgs);
}

/*
-- < Create parse errors

newErrorUnknown :: SourcePos -> ParseError
newErrorUnknown pos
    = ParseError pos []

*/
inline ParseError newErrorUnknown(SourcePos const& pos) {
    return ParseError(pos);
}

/*
newErrorMessage :: Message -> SourcePos -> ParseError
newErrorMessage msg pos
    = ParseError pos [msg]
*/
DEFINE_FUNCTION_2_NOTEMPL_NC(ParseError, newErrorMessage, Message const&, msg, SourcePos const&, pos,
    return ParseError(pos, List<Message>({ msg }));)

/*
addErrorMessage :: Message -> ParseError -> ParseError
addErrorMessage msg (ParseError pos msgs)
    = ParseError pos (msg:msgs)
*/
DEFINE_FUNCTION_2_NOTEMPL_NC(ParseError, addErrorMessage, Message const&, msg, ParseError const&, err,
    return ParseError(err.pos, msg >> err.msgs);)

/*
setErrorPos :: SourcePos -> ParseError -> ParseError
setErrorPos pos (ParseError _ msgs)
    = ParseError pos msgs
*/
DEFINE_FUNCTION_2_NOTEMPL_NC(ParseError, setErrorPos, SourcePos const&, pos, ParseError const&, err,
    return ParseError(pos, err.msgs);)

/*
setErrorMessage :: Message -> ParseError -> ParseError
setErrorMessage msg (ParseError pos msgs)
    = ParseError pos (msg : filter (msg /=) msgs)
*/
DEFINE_FUNCTION_2_NOTEMPL_NC(ParseError, setErrorMessage, Message const&, msg, ParseError const&, err,
    return ParseError(err.pos, msg >> filter(_neq(msg), err.msgs));)

/*
mergeError :: ParseError -> ParseError -> ParseError
mergeError e1@(ParseError pos1 msgs1) e2@(ParseError pos2 msgs2)
    -- prefer meaningful errors
    | null msgs2 && not (null msgs1) = e1
    | null msgs1 && not (null msgs2) = e2
    | otherwise
    = case pos1 `compare` pos2 of
        -- select the longest match
        EQ -> ParseError pos1 (msgs1 ++ msgs2)
        GT -> e1
        LT -> e2
*/
DEFINE_FUNCTION_2_NOTEMPL_NC(ParseError, mergeError, ParseError const&, err1, ParseError const&, err2,
    return
        err2.msgs.empty() && !err1.msgs.empty() ? err1 :
        err1.msgs.empty() && !err2.msgs.empty() ? err2 :
        err1.pos == err2.pos ? ParseError(err1.pos, err1.msgs + err2.msgs) :
        err1.pos < err2.pos ? err2 : err1;)

/*
instance Eq ParseError where
    l == r
        = errorPos l == errorPos r && messageStrs l == messageStrs r
        where
          messageStrs = map messageString . errorMessages
*/
inline bool operator==(ParseError const& l, ParseError const& r)
{
    const function_t<List<_FUNCPROG::String>(ParseError const&)> messageStrs =
        _map(_(messageString)) & _(errorMessages);
    return l.pos == r.pos && messageStrs(l) == messageStrs(r);
}

inline bool operator!=(ParseError const& l, ParseError const& r) {
    return !(l == r);
}

inline _FUNCPROG::String showErrorMessages(const char *msgOr, const char *msgUnknown, const char *msgExpecting,
    const char *msgUnExpected, const char *msgEndOfInput, List<Message> const& msgs)
{
    /*
        | null msgs = msgUnknown
        | otherwise = concat $ map ("\n"++) $ clean $
                     [showSysUnExpect,showUnExpect,showExpect,showMessages]
    */
    if (null(msgs))
        return msgUnknown;
    // helpers
    // clean = nub . filter (not . null)
    const function_t<List<_FUNCPROG::String>(List<_FUNCPROG::String> const&)> clean =
        [](List<_FUNCPROG::String> const& l) {
            return nub(filter(not_(_(sempty)), l));
        };
    // separate   _ []     = ""
    // separate   _ [m]    = m
    // separate sep (m:ms) = m ++ sep ++ separate sep ms
    const function_t<_FUNCPROG::String(const char*, List<_FUNCPROG::String> const&)> separate =
        [&separate](const char *sep, List<_FUNCPROG::String> const& ms) {
            return null(ms) ? "" :
                length(ms) == 1 ? head(ms) :
                String(head(ms) + sep + separate(sep, tail(ms)));
        };

    // commaSep = separate ", " . clean
    const function_t<_FUNCPROG::String(List<_FUNCPROG::String> const&)> commaSep =
        [&clean, &separate](List<_FUNCPROG::String> const& ms) {
            return separate(", ", clean(ms));
        };
    
    // commasOr []  = ""
    // commasOr [m] = m
    // commasOr ms  = commaSep (init ms) ++ " " ++ msgOr ++ " " ++ last ms
    const function_t<_FUNCPROG::String(List<_FUNCPROG::String> const&)> commasOr =
        [&commaSep, &msgOr](List<_FUNCPROG::String> const& ms) {
            return null(ms) ? "" :
                length(ms) == 1 ? head(ms) : String(commaSep(init(ms)) + ' ' + msgOr + ' ' + last(ms));
        };

/*
showMany pre msgs3 = case clean (map messageString msgs3) of
                    [] -> ""
                    ms | null pre  -> commasOr ms
                       | otherwise -> pre ++ " " ++ commasOr ms
*/
    const function_t<_FUNCPROG::String(const char*, List<Message> const&)> showMany =
        [&clean, &commasOr](const char *pre, List<Message> const& msgs3)
        {
            const List<_FUNCPROG::String> ms = clean(map(_(messageString), msgs3));
            return null(ms) ? "" :
                pre == 0 || *pre == 0 ? commasOr(ms) :
                pre + (' ' + commasOr(ms));
        };
/*
    (sysUnExpect,msgs1) = span ((SysUnExpect "") ==) msgs
    (unExpect,msgs2)    = span ((UnExpect    "") ==) msgs1
    (expect,messages)   = span ((Expect      "") ==) msgs2
*/
    const auto [sysUnExpect, msgs1] = span(_eq<Message>(Message(SysUnExpect, "")), msgs);
    const auto [unExpect, msgs2] = span(_eq<Message>(Message(UnExpect, "")), msgs1);
    const auto [expect, messages] = span(_eq<Message>(Message(Expect, "")), msgs2);
/*
      showExpect      = showMany msgExpecting expect
      showUnExpect    = showMany msgUnExpected unExpect
      showSysUnExpect | not (null unExpect) ||
                        null sysUnExpect = ""
                      | null firstMsg    = msgUnExpected ++ " " ++ msgEndOfInput
                      | otherwise        = msgUnExpected ++ " " ++ firstMsg
          where
              firstMsg  = messageString (head sysUnExpect)

      showMessages      = showMany "" messages
*/
    const _FUNCPROG::String
        showExpect = showMany(msgExpecting, expect),
        showUnExpect = showMany(msgUnExpected, unExpect),
        firstMsg = null(sysUnExpect) ? "" : messageString(head(sysUnExpect)),
        showSysUnExpect = !null(unExpect) || null(sysUnExpect) ? "" :
            msgUnExpected + (' ' + (firstMsg.empty() ? msgEndOfInput : firstMsg)),
        showMessages = showMany("", messages);

    // concat $ map ("\n"++) $ clean $
    //    [showSysUnExpect,showUnExpect,showExpect,showMessages]
    return strconcat(map(_concat2<char>("\n"), clean(List<_FUNCPROG::String>({ showSysUnExpect, showUnExpect, showExpect, showMessages }))));
}

// unknownError :: State s u -> ParseError
// unknownError state        = newErrorUnknown (statePos state)
template<typename S, typename U>
constexpr ParseError unknownError(State<S, U> const& state) {
    return newErrorUnknown(state.pos);
}

// unexpectError::String->SourcePos->ParseError
// unexpectError msg pos = newErrorMessage(SysUnExpect msg) pos
ParseError unexpectError(_FUNCPROG::String const& msg, SourcePos const& pos) {
    return newErrorMessage(Message(SysUnExpect, msg), pos);
}

_PARSEC_END

namespace std {
    inline ostream& operator<<(ostream& os, _PARSEC::ParseError const& err) {
        return os << err.pos << ':' << _PARSEC::showErrorMessages("or", "unknown parse error", "expecting", "unexpected", "end of input", _PARSEC::errorMessages(err));
    }
}
