#pragma once

#include "../List.hpp"

_PARSEC_BEGIN

using SourceName = std::string;

struct SourcePos
{
    SourcePos(SourceName const& name, int line, int column) : name(name), line(line), column(column) {}

    const SourceName name;
    const int line, column;
};

DECLARE_FUNCTION_3_NOTEMPL_NC(SourcePos, newPos, SourceName const&, int, int)
inline SourcePos newPos(SourceName const& name, int line, int column) {
    return SourcePos(name, line, column);
}

inline SourcePos initialPos(SourceName const& name) {
    return SourcePos(name, 1, 1);
}

// Increments the line number of a source position.
DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, incSourceLine, SourcePos const&, int)
inline SourcePos incSourceLine(SourcePos const& pos, int n) {
    return SourcePos(pos.name, pos.line + n, pos.column);
}

// Increments the column number of a source position.
DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, incSourceColumn, SourcePos const&, int)
inline SourcePos incSourceColumn(SourcePos const& pos, int n) {
    return SourcePos(pos.name, pos.line, pos.column + n);
}

// Set the name of the source.
DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, setSourceName, SourcePos const&, SourceName const&)
inline SourcePos setSourceName(SourcePos const& pos, SourceName const& name) {
    return SourcePos(name, pos.line, pos.column);
}

// Set the line number of a source position.
DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, setSourceLine, SourcePos const&, int)
inline SourcePos setSourceLine(SourcePos const& pos, int n) {
    return SourcePos(pos.name, n, pos.column);
}

// Set the column number of a source position.
DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, setSourceColumn, SourcePos const&, int)
inline SourcePos setSourceColumn(SourcePos const& pos, int n) {
    return SourcePos(pos.name, pos.line, n);
}

/*
-- | Update a source position given a character. If the character is a
-- newline (\'\\n\') or carriage return (\'\\r\') the line number is
-- incremented by 1. If the character is a tab (\'\t\') the column
-- number is incremented to the nearest 8'th column, ie. @column + 8 -
-- ((column-1) \`mod\` 8)@. In all other cases, the column is
-- incremented by 1.

updatePosChar   :: SourcePos -> Char -> SourcePos
updatePosChar (SourcePos name line column) c
    = case c of
        '\n' -> SourcePos name (line+1) 1
        '\t' -> SourcePos name line (column + 8 - ((column-1) `mod` 8))
        _    -> SourcePos name line (column + 1)
*/

DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, updatePosChar, SourcePos const&, char)
inline SourcePos updatePosChar(SourcePos const& pos, char c)
{
    switch (c)
    {
    case '\n': return SourcePos(pos.name, pos.line + 1, 1);
    case '\t': return SourcePos(pos.name, pos.line, pos.column + 8 - (pos.column - 1) % 8);
    default: return SourcePos(pos.name, pos.line, pos.column + 1);
    }
}

/*
-- | The expression @updatePosString pos s@ updates the source position
-- @pos@ by calling 'updatePosChar' on every character in @s@, ie.
-- @foldl updatePosChar pos string@.

updatePosString :: SourcePos -> String -> SourcePos
updatePosString pos string
    = foldl updatePosChar pos string
*/

DECLARE_FUNCTION_2_NOTEMPL_NC(SourcePos, updatePosString, SourcePos const&, const char*)
inline SourcePos updatePosString(SourcePos const& pos, const char* s) {
    return foldl(_(updatePosChar), pos, List<char>(s));
}

/*
instance Show SourcePos where
  show (SourcePos name line column)
    | null name = showLineColumn
    | otherwise = "\"" ++ name ++ "\" " ++ showLineColumn
    where
      showLineColumn    = "(line " ++ show line ++
                          ", column " ++ show column ++
                          ")"
*/

inline bool operator==(SourcePos const& l, SourcePos const& r) {
    return l.name == r.name && l.line == r.line && l.column == r.column;
}

inline bool operator!=(SourcePos const& l, SourcePos const& r) {
    return !(l == r);
}

inline bool operator<(SourcePos const& l, SourcePos const& r) {
    return l.name < r.name ||
        (l.name == r.name && l.line < r.line) ||
        (l.name == r.name && l.line == r.line && l.column < r.column);
}

_PARSEC_END

namespace std {
    inline ostream& operator<<(ostream& os, _PARSEC::SourcePos const& pos) {
        if (!pos.name.empty())
            os << '"' << pos.name << "\" ";
        return os << "(line " << pos.line << ", column " << pos.column << ')';
    }
}
