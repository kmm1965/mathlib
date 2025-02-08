#pragma once

_FUNCPROG_BEGIN

// Requires lift
template<typename __M>
struct MonadTrans;

template<typename __M, typename F>
using lift_type = monad_type<F, typename __M::template base_type<base_class_t<F> >::template type<typename F::value_type> >;

#define LIFT_TYPE_(__M, F) BOOST_IDENTITY_TYPE((lift_type<__M, F>))
#define LIFT_TYPE(__M, F) typename LIFT_TYPE_(__M, F)

template<typename __M, typename F>
lift_type<__M, F> lift(F const& m) {
    return MonadTrans<__M>::lift(m);
}

_FUNCPROG_END
