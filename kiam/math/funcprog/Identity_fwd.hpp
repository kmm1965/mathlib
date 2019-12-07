#pragma once

#include "func.hpp"

_FUNCPROG_BEGIN

struct _Identity;

template<typename A>
struct Identity;

template<typename T> T runIdentity(Identity<T> const&);

// Constructor
template<typename T>
Identity<T> Identity_(T const&);

template<typename T>
Identity<T> Identity_f(f0<T> const& fvalue);

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Identity<T> const& v);

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Identity<T> const& v);

ostream& operator<<(ostream& os, _FUNCPROG::Identity<string> const& v);
wostream& operator<<(wostream& os, _FUNCPROG::Identity<wstring> const& v);
ostream& operator<<(ostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<string> > const& v);
wostream& operator<<(wostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<wstring> > const& v);

} // namespace std
