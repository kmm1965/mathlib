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
// Requires _empty, operator|
template<typename ALT>
struct Alternative;

template<typename A>
using Alternative_t = Alternative<base_class_t<A> >;

template<class A>
struct is_alternative : std::false_type {};

template<class A>
using is_alternative_t = is_alternative<base_class_t<A> >;

template<class A1, class A2>
struct is_same_alternative : std::false_type {};

template<class A1, class A2>
using is_same_alternative_t = is_same_alternative<base_class_t<A1>, base_class_t<A2> >;

#define IMPLEMENT_ALTERNATIVE(ALT, _ALT) \
	template<> struct is_alternative<_ALT> : std::true_type {}; \
    template<> struct is_same_alternative<_ALT, _ALT> : std::true_type {};

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
using alt_op_result_type = typename std::enable_if<
	is_alternative_t<ALT>::value,
	typename Alternative_t<ALT>::template alt_op_result_type_t<ALT>
>::type;

template<class ALT>
alt_op_result_type<ALT> operator|(ALT const& op1, ALT const& op2) {
	return Alternative_t<ALT>::alt_op(op1, op2);
}

_FUNCPROG_END
