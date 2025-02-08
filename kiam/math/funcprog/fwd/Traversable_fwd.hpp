#pragma once

#include "Applicative_fwd.hpp"
#include "Monad_fwd.hpp"

_FUNCPROG_BEGIN

template<class _T>
struct _is_traversable : std::false_type {};

template<class _T>
bool constexpr _is_traversable_v = _is_traversable<_T>::value;

template<class T>
using is_traversable = _is_traversable<base_class_t<T> >;

template<class T>
bool constexpr is_traversable_v = is_traversable<T>::value;

template<class TR, typename T = TR>
using traversable_type = std::enable_if_t<is_traversable<TR>::value, T>;

#define TRAVERSABLE_TYPE_(TR, T) BOOST_IDENTITY_TYPE((traversable_type<TR, T>))
#define TRAVERSABLE_TYPE(TR, T) typename TRAVERSABLE_TYPE_(TR, T)

// requires traverse, sequenceA
template<typename T>
struct Traversable;

template<typename T>
using Traversable_t = Traversable<base_class_t<T> >;

#define IMPLEMENT_TRAVERSABLE(_T) \
    template<> struct _is_traversable<_T> : std::true_type {}

// traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
template<typename T, typename AP, typename Arg>
using traverse_type = std::enable_if_t<
    is_traversable<T>::value&& is_applicative<AP>::value&& is_same_as_v<value_type_t<T>, Arg>,
    typeof_t<AP, typeof_t<T, value_type_t<AP> > >
>;

#define TRAVERSE_TYPE_(T, AP, ARG) BOOST_IDENTITY_TYPE((traverse_type<T, AP, ARG>))
#define TRAVERSE_TYPE(T, AP, ARG) typename TRAVERSE_TYPE_(T, AP, ARG)

DECLARE_FUNCTION_2(3, TRAVERSE_TYPE(T0, T1, T2), traverse, function_t<T1(T2)> const&, T0 const&);

// sequenceA :: Applicative f => t (f a) -> f (t a)
template<typename T>
using sequenceA_type = std::enable_if_t<
    is_traversable<T>::value && is_applicative<value_type_t<T> >::value,
    typeof_t<value_type_t<T>, typeof_t<T, value_type_t<value_type_t<T> > > >
>;

template<typename T>
constexpr sequenceA_type<T> sequenceA(T const& x);

// mapM :: Monad m => (a -> m b) -> t a -> m (t b)
template<typename T, typename M, typename ARG>
using mapM_type = std::enable_if_t<
    is_traversable<T>::value && is_monad_v<M> && is_same_as_v<value_type_t<T>, ARG>,
    typeof_t<M, typeof_t<T, value_type_t<M> > >
>;

#define MAPM_TYPE_(T, M, ARG) BOOST_IDENTITY_TYPE((mapM_type<T, M, ARG>))
#define MAPM_TYPE(T, M, ARG) typename MAPM_TYPE_(T, M, ARG)

DECLARE_FUNCTION_2(3, MAPM_TYPE(T0, T1, T2), mapM, function_t<T1(T2)> const&, T0 const&);

// sequence :: Monad m => t (m a) -> m (t a)
template<typename T>
using sequence_type = std::enable_if_t<
    is_traversable<T>::value && is_monad_v<value_type_t<T> >,
    typeof_t<value_type_t<T>, typeof_t<T, value_type_t<value_type_t<T> > > >
>;

template<typename T>
constexpr sequence_type<T> sequence(T const& x);

_FUNCPROG_END
