#pragma once

#include "ParsecT.hpp"
#include "../detail/anyall.hpp"
#include "../detail/show.hpp"

_PARSEC_BEGIN

/*
-- | The parser @satisfy f@ succeeds for any character for which the
-- supplied function @f@ returns 'True'. Returns the character that is
-- actually parsed.

-- >  digit     = satisfy isDigit
-- >  oneOf cs  = satisfy (\c -> c `elem` cs)

satisfy :: (Stream s m Char) => (Char -> Bool) -> ParsecT s u m Char
satisfy f           = tokenPrim (\c -> show [c])
                                (\pos c _cs -> updatePosChar pos c)
                                (\c -> if f c then Just c else Nothing)
*/
template<typename U, typename _M>
constexpr auto satisfy(function_t<bool(char)> const& f){
    return tokenPrim<U, _M, char, char>(
        _([](char c){ return std::string("\"") + c + "\""; }),
        _([](SourcePos const& pos, char c, const String &_cs){ return updatePosChar(pos, c); }),
        _([f](char c){ return f(c) ? Just(c) : Nothing<char>(); }));
}

/*
-- | @oneOf cs@ succeeds if the current character is in the supplied
-- list of characters @cs@. Returns the parsed character. See also
-- 'satisfy'.
--
-- >   vowel  = oneOf "aeiou"

oneOf :: (Stream s m Char) => [Char] -> ParsecT s u m Char
oneOf cs            = satisfy (\c -> elem c cs)
*/

template<typename U, typename _M>
constexpr auto oneOf(const String &cs){
    return satisfy<U, _M>(_([cs](char c){ return elem(c, cs); }));
}

/*
-- | As the dual of 'oneOf', @noneOf cs@ succeeds if the current
-- character /not/ in the supplied list of characters @cs@. Returns the
-- parsed character.
--
-- >  consonant = noneOf "aeiou"

noneOf :: (Stream s m Char) => [Char] -> ParsecT s u m Char
noneOf cs           = satisfy (\c -> not (elem c cs))
*/
template<typename U, typename _M>
constexpr auto noneOf(const String &cs){
    return satisfy<U, _M>(_([cs](char c){ return !elem(c, cs); }));
}

/*
-- | @char c@ parses a single character @c@. Returns the parsed
-- character (i.e. @c@).
--
-- >  semiColon  = char ';'

char :: (Stream s m Char) => Char -> ParsecT s u m Char
char c              = satisfy (==c)  <?> show [c]
*/
template<typename U, typename _M>
constexpr auto char_(char c){
    return satisfy<U, _M>(_([c](char c1){ return c1 == c; })) & (std::string("\"") + c + '"');
}

/*
-- | This parser succeeds for any character. Returns the parsed character.

anyChar :: (Stream s m Char) => ParsecT s u m Char
anyChar             = satisfy (const True)
*/
template<typename U, typename _M>
constexpr auto anyChar(){
    return satisfy<U, _M>(_const_<char>(true));
}

/*
-- | Parses a white space character (any character which satisfies 'isSpace')
-- Returns the parsed character.

space :: (Stream s m Char) => ParsecT s u m Char
space               = satisfy isSpace       <?> "space"
*/
template<typename U, typename _M>
constexpr auto space(){
    return satisfy<U, _M>(_([](char c){ return std::isspace(c); })) & std::string("space");
}

/*
-- | Skips /zero/ or more white space characters. See also 'skipMany'.

spaces :: (Stream s m Char) => ParsecT s u m ()
spaces              = skipMany space        <?> "white space"
*/
template<typename U, typename _M>
constexpr auto spaces(){
    return skipMany(space<U, _M>()) & std::string("white space");
}

/*
-- | Parses a newline character (\'\\n\'). Returns a newline character.

newline :: (Stream s m Char) => ParsecT s u m Char
newline             = char '\n'             <?> "lf new-line"
*/
template<typename U, typename _M>
constexpr auto newline(){
    return char_<U, _M>('\n') & std::string("lf new-line");
}

/*
-- | Parses a carriage return character (\'\\r\') followed by a newline character (\'\\n\').
-- Returns a newline character.

crlf :: (Stream s m Char) => ParsecT s u m Char
crlf                = char '\r' *> char '\n' <?> "crlf new-line"
*/
template<typename U, typename _M>
constexpr auto crlf(){
    return (char_<U, _M>('\r') *= char_<U, _M>('\n')) & std::string("crlf new-line");
}

/*
-- | Parses a CRLF (see 'crlf') or LF (see 'newline') end-of-line.
-- Returns a newline character (\'\\n\').
--
-- > endOfLine = newline <|> crlf
--

endOfLine :: (Stream s m Char) => ParsecT s u m Char
endOfLine           = newline <|> crlf       <?> "new-line"
*/
template<typename U, typename _M>
constexpr auto endOfLine(){
    return (newline<U, _M>() | crlf<U, _M>()) & std::string("new-line");
}

/*
-- | Parses a tab character (\'\\t\'). Returns a tab character.

tab :: (Stream s m Char) => ParsecT s u m Char
tab                 = char '\t'             <?> "tab"
*/
template<typename U, typename _M>
constexpr auto tab(){
    return char_<U, _M>('\t') & std::string("tab");
}

/*
-- | Parses an upper case letter (according to 'isUpper').
-- Returns the parsed character.

upper :: (Stream s m Char) => ParsecT s u m Char
upper               = satisfy isUpper       <?> "uppercase letter"
*/
template<typename U, typename _M>
constexpr auto upper(){
    return satisfy<U, _M>(_([](char c){ return std::isupper(c); })) & std::string("uppercase letter");
}

/*
-- | Parses a lower case character (according to 'isLower').
-- Returns the parsed character.

lower :: (Stream s m Char) => ParsecT s u m Char
lower               = satisfy isLower       <?> "lowercase letter"
*/
template<typename U, typename _M>
constexpr auto lower(){
    return satisfy<U, _M>(_([](char c){ return std::islower(c); })) & std::string("lowercase letter");
}

/*
-- | Parses a letter or digit (a character between \'0\' and \'9\')
-- according to 'isAlphaNum'. Returns the parsed character.

alphaNum :: (Stream s m Char => ParsecT s u m Char)
alphaNum            = satisfy isAlphaNum    <?> "letter or digit"
*/
template<typename U, typename _M>
constexpr auto alphaNum(){
    return satisfy<U, _M>(_([](char c){ return std::isalnum(c); })) & std::string("letter or digit");
}

/*
-- | Parses a letter (an upper case or lower case character according
-- to 'isAlpha'). Returns the parsed character.

letter :: (Stream s m Char) => ParsecT s u m Char
letter              = satisfy isAlpha       <?> "letter"
*/
template<typename U, typename _M>
constexpr auto letter(){
    return satisfy<U, _M>(_([](char c){ return std::isalpha(c); })) & std::string("letter");
}

/*
-- | Parses a digit. Returns the parsed character.

digit :: (Stream s m Char) => ParsecT s u m Char
digit               = satisfy isDigit       <?> "digit"

-- | Parses a hexadecimal digit (a digit or a letter between \'a\' and
-- \'f\' or \'A\' and \'F\'). Returns the parsed character.
*/
template<typename U, typename _M>
constexpr auto digit(){
    return satisfy<U, _M>(_([](char c){ return std::isdigit(c); })) & std::string("digit");
}

/*
hexDigit :: (Stream s m Char) => ParsecT s u m Char
hexDigit            = satisfy isHexDigit    <?> "hexadecimal digit"

-- | Parses an octal digit (a character between \'0\' and \'7\'). Returns
-- the parsed character.
*/
template<typename U, typename _M>
constexpr auto hexDigit(){
    return satisfy<U, _M>(_([](char c){ return std::isxdigit(c); })) & std::string("hexadecimal digit");
}

/*
octDigit :: (Stream s m Char) => ParsecT s u m Char
octDigit            = satisfy isOctDigit    <?> "octal digit"
*/
template<typename U, typename _M>
constexpr auto octDigit(){
    return satisfy<U, _M>(_([](char c){ return '0' <= c && c <= '7'; })) & std::string("octal digit");
}

/*
-- | @string s@ parses a sequence of characters given by @s@. Returns
-- the parsed string (i.e. @s@).
--
-- >  divOrMod    =   string "div"
-- >              <|> string "mod"

string :: (Stream s m Char) => String -> ParsecT s u m String
string s            = tokens show updatePosString s
*/
template<typename U, typename _M>
constexpr auto _string(const char *s){
    return tokens<String, U, _M, char>(show<String>,
        _([](SourcePos const& pos, const String &s){ return updatePosString(pos, s.c_str()); }),
        String(s));
}

_PARSEC_END
