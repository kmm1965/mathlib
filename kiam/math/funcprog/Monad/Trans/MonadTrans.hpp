#pragma once

_FUNCPROG_BEGIN

// Requires lift
template<typename T>
struct MonadTrans;

template<typename __M, typename F>
using lift_type = typename std::enable_if<
    is_monad<F>::value,
    typename __M::template mt_type<base_class_t<F> >::template type<typename F::value_type>
>::type;

#define LIFT_TYPE_(__M, F) BOOST_IDENTITY_TYPE((lift_type<__M, F>))
#define LIFT_TYPE(__M, F) typename LIFT_TYPE_(__M, F)

template<typename __M, typename F>
lift_type<__M, F> lift(F const& m) {
    return MonadTrans<typename __M::template mt_type<base_class_t<F> > >::lift(m);
}

_FUNCPROG_END
