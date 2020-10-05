#pragma once

#include "Monad.hpp"

_FUNCPROG_BEGIN

// requires traverse, sequenceA
template<typename T>
struct Traversable;

template<typename T>
using Traversable_t = Traversable<base_class_t<T> >;

template<class T>
struct is_traversable : std::false_type {};

#define DECLARE_TRAVERSABLE_CLASS(T) \
    /* traverse :: Applicative f => (a -> f b) -> t a -> f (t b) */ \
    template<typename AP, typename Arg> \
    static typename std::enable_if<is_applicative<AP>::value, typeof_t<AP, T<value_type_t<AP> > > >::type \
    traverse(function_t<AP(Arg)> const& f, T<fdecay<Arg> > const& x); \
    \
    /* sequenceA :: Applicative f => t (f a) -> f (t a) */ \
    template<typename A> \
    static typename std::enable_if<is_applicative<A>::value, typeof_t<A, T<value_type_t<A> > > >::type \
    sequenceA(T<A> const& x);

#define DEFAULT_SEQUENCEA_IMPL(T, _T) \
    /* sequenceA = traverse id */ \
    template<typename A> \
    typename std::enable_if<is_applicative<A>::value, typeof_t<A, T<value_type_t<A> > > >::type \
    Traversable<_T>::sequenceA(T<A> const& x) { \
        return traverse(_(id<A>), x); \
    }

#define IMPLEMENT_TRAVERSABLE(T) \
    template<typename A> struct is_traversable<T<A> > : std::true_type {}

// traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
template<typename T, typename AP, typename Arg>
using traverse_type = typename std::enable_if<
    is_traversable<T>::value && is_applicative<AP>::value && is_same_as<value_type_t<T>, Arg>::value,
    typeof_t<AP, typeof_t<T, value_type_t<AP> > >
>::type;

#define TRAVERSE_TYPE_(T, AP, ARG) BOOST_IDENTITY_TYPE((traverse_type<T, AP, ARG>))
#define TRAVERSE_TYPE(T, AP, ARG) typename TRAVERSE_TYPE_(T, AP, ARG)

DEFINE_FUNCTION_2(3, TRAVERSE_TYPE(T0, T1, T2), traverse, function_t<T1(T2)> const&, f, T0 const&, x,
    return Traversable_t<T0>::traverse(f, x);)

// sequenceA :: Applicative f => t (f a) -> f (t a)
template<typename T>
using sequenceA_type = typename std::enable_if<
    is_traversable<T>::value && is_applicative<value_type_t<T> >::value,
    typeof_t<value_type_t<T>, typeof_t<T, value_type_t<value_type_t<T> > > >
>::type;

template<typename T>
sequenceA_type<T> sequenceA(T const& x) {
    return Traversable_t<T>::sequenceA(x);
}

// mapM :: Monad m => (a -> m b) -> t a -> m (t b)
template<typename T, typename M, typename ARG>
using mapM_type = typename std::enable_if<
    is_traversable<T>::value && is_monad<M>::value && is_same_as<value_type_t<T>, ARG>::value,
    typeof_t<M, typeof_t<T, value_type_t<M> > >
>::type;

#define MAPM_TYPE_(T, M, ARG) BOOST_IDENTITY_TYPE((mapM_type<T, M, ARG>))
#define MAPM_TYPE(T, M, ARG) typename MAPM_TYPE_(T, M, ARG)

DEFINE_FUNCTION_2(3, MAPM_TYPE(T0, T1, T2), mapM, function_t<T1(T2)> const&, f, T0 const&, x,
    return traverse(f, x);)

// sequence :: Monad m => t (m a) -> m (t a)
template<typename T>
using sequence_type = typename std::enable_if<
    is_traversable<T>::value && is_monad<value_type_t<T> >::value,
    typeof_t<value_type_t<T>, typeof_t<T, value_type_t<value_type_t<T> > > >
>::type;

template<typename T>
sequence_type<T> sequence(T const& x) {
    return sequenceA(x);
}

_FUNCPROG_END
