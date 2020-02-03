#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<unsigned N, unsigned I, class Closure>
__DEVICE __HOST
typename std::enable_if<(I == N)>::type meta_loop_(Closure &){}

template<unsigned N, unsigned I, class Closure>
__DEVICE __HOST
typename std::enable_if<(I < N)>::type
meta_loop_(Closure &closure)
{
    closure.template apply<I>();
    meta_loop_<N, I + 1>(closure);
}

template<unsigned N, class Closure>
__DEVICE __HOST
void meta_loop(Closure &closure){
    meta_loop_<N, 0>(closure);
}

template<class Closure>
struct abstract_sum_closure
{
    typedef typename Closure::value_type value_type;

    __DEVICE __HOST
    CONSTEXPR abstract_sum_closure(const Closure &closure) : closure(closure), result(value_type()){}

    template<unsigned I>
    __DEVICE __HOST
    void apply(){
        result += closure.template value<I>();
    }

    const Closure &closure;
    value_type result;
};

template<unsigned N, class Closure>
__DEVICE __HOST
typename Closure::value_type abstract_sum(const Closure &closure)
{
    abstract_sum_closure<Closure> my_closure(closure);
    meta_loop<N>(my_closure);
    return my_closure.result;
}

_KIAM_MATH_END
