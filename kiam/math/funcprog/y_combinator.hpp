#pragma once

#include "funcprog_setup.h"

_FUNCPROG_BEGIN

template<class Fun>
struct y_combinator_result
{
    template<class T>
    explicit y_combinator_result(T &&fun) : fun(std::forward<T>(fun)) {}

    template<class ...Args>
    decltype(auto) operator()(Args &&...args) const {
        return fun(std::ref(*this), std::forward<Args>(args)...);
    }

private:
	const Fun fun;
};

template<class Fun>
y_combinator_result<typename std::decay<Fun>::type> y_combinator(Fun &&fun) {
    return y_combinator_result<typename std::decay<Fun>::type>(std::forward<Fun>(fun));
}

_FUNCPROG_END
