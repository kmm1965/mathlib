#pragma once

_FUNCPROG_BEGIN

/*
-- -----------------------------------------------------------------------------
-- The Alternative class definition

infixl 3 <|>

-- | A monoid on applicative functors.
--
-- If defined, 'some' and 'many' should be the least solutions
-- of the equations:
--
-- * @'some' v = (:) '<$>' v '<*>' 'many' v@
--
-- * @'many' v = 'some' v '<|>' 'pure' []@
class Applicative f => Alternative f where
    -- | The identity of '<|>'
    empty :: f a
    -- | An associative binary operation
    (<|>) :: f a -> f a -> f a

    -- | One or more.
    some :: f a -> f [a]
    some v = some_v
      where
        many_v = some_v <|> pure []
        some_v = liftA2 (:) v many_v

    -- | Zero or more.
    many :: f a -> f [a]
    many v = many_v
      where
        many_v = some_v <|> pure []
        some_v = liftA2 (:) v many_v
*/
template<class A>
struct _is_alternative : std::false_type {};

template<class A>
using is_alternative = _is_alternative<base_class_t<A> >;

template<class A, typename T = A>
using alternative_type = typename std::enable_if<is_alternative<A>::value, T>::type;

template<class _A1, class _A2>
struct _is_same_alternative : std::false_type {};

template<class _A>
struct _is_same_alternative<_A, _A> : _is_alternative<_A> {};

template<class A1, class A2>
using is_same_alternative = _is_same_alternative<base_class_t<A1>, base_class_t<A2> >;

template<class A1, class A2, typename T>
using same_alternative_type = typename std::enable_if<is_same_alternative<A1, A2>::value, T>::type;

// Requires _empty, operator|
template<typename ALT>
struct Alternative;

template<typename A>
using Alternative_t = Alternative<base_class_t<A> >;

#define IMPLEMENT_ALTERNATIVE(_ALT) \
    template<> struct _is_alternative<_ALT> : std::true_type {}

#define DECLARE_ALTERNATIVE_CLASS(ALT) \
    template<typename T> struct alt_op_result_type; \
    template<typename T> using alt_op_result_type_t = typename alt_op_result_type<T>::type; \
    template<typename T> \
    struct alt_op_result_type<ALT<T> >{ \
        using type = ALT<T>; \
    }; \
    template<typename T> static ALT<T> empty(); \
    template<typename T> static ALT<T> alt_op(ALT<T> const& op1, ALT<T> const& op2);

template<class ALT>
using alt_op_result_type = alternative_type<ALT, typename Alternative_t<ALT>::template alt_op_result_type_t<ALT> >;

template<class ALT>
alt_op_result_type<ALT> operator|(ALT const& op1, ALT const& op2) {
    return Alternative_t<ALT>::alt_op(op1, op2);
}

_FUNCPROG_END
