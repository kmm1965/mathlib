#pragma once

#include "../funcprog_setup.h"

_FUNCPROG_BEGIN

template<typename T>
std::string show(T const& v)
{
    std::ostringstream os;
    os << v;
    return os.str();
}

template<>
std::string show<char>(char const& c)
{
    std::ostringstream os;
    os << '\'' << c << '\'';
    return os.str();
}

template<>
std::string show<const char*>(const char* const& s)
{
    std::ostringstream os;
    os << '"' << s << '"';
    return os.str();
}

_FUNCPROG_END
