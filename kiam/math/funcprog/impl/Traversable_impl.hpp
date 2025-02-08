#pragma once

#include "../Traversable.hpp"

_FUNCPROG_BEGIN

// sequenceA :: Applicative f => t (f a) -> f (t a)
// sequenceA = traverse id
template<typename _T>
template<typename A>
constexpr applicative_type<A, typeof_t<A, typeof_t<_T, value_type_t<A> > > >
_Traversable<_T>::sequenceA(typeof_t<_T, A> const& x){
    return Traversable<_T>::traverse(_(id<A>), x);
}

FUNCTION_TEMPLATE(3) constexpr TRAVERSE_TYPE(T0, T1, T2) traverse(function_t<T1(T2)> const& f, T0 const& x) {
    return Traversable_t<T0>::traverse(f, x);
}

template<typename T>
constexpr sequenceA_type<T> sequenceA(T const& x){
    return Traversable_t<T>::sequenceA(x);
}

FUNCTION_TEMPLATE(3) constexpr MAPM_TYPE(T0, T1, T2) mapM(function_t<T1(T2)> const& f, T0 const& x) {
    return traverse(f, x);
}

template<typename T>
constexpr sequence_type<T> sequence(T const& x){
    return sequenceA(x);
}

_FUNCPROG_END
