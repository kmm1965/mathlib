#pragma once

#include "Monad.hpp"
#include "Alternative.hpp"

_FUNCPROG_BEGIN

/*
-- -----------------------------------------------------------------------------
-- The MonadPlus class definition

-- | Monads that also support choice and failure.
class (Alternative m, Monad m) => MonadPlus m where
   -- | The identity of 'mplus'.  It should also satisfy the equations
   --
   -- > mzero >>= f  =  mzero
   -- > v >> mzero   =  mzero
   --
   -- The default definition is
   --
   -- @
   -- mzero = 'empty'
   -- @
   mzero :: m a
   mzero = empty

   -- | An associative operation. The default definition is
   --
   -- @
   -- mplus = ('<|>')
   -- @
   mplus :: m a -> m a -> m a
   mplus = (<|>)
*/
template<class _MP>
struct _is_monad_plus : std::false_type {};

template<class MP>
using is_monad_plus = _is_monad_plus<base_class_t<MP> >;

template<class MP, typename T = MP>
using monad_plus_type = typename std::enable_if<is_monad_plus<MP>::value, T>::type;

template<class _MP1, class _MP2>
struct _is_same_monad_plus : std::false_type {};

template<class _MP>
struct _is_same_monad_plus<_MP, _MP> : _is_monad_plus<_MP> {};

template<class MP1, class MP2>
using is_same_monad_plus = _is_same_monad_plus<base_class_t<MP1>, base_class_t<MP2> >;

template<class MP1, class MP2, typename T>
using same_monad_plus_type = typename std::enable_if<is_same_monad_plus<MP1, MP2>::value, T>::type;

// Requires mzero (default empty), mplus (default |)
template<typename A>
struct MonadPlus;

template<typename T>
using MonadPlus_t = MonadPlus<base_class_t<T> >;

#define IMPLEMENT_MONADPLUS(_MP) \
    template<> struct _is_monad_plus<_MP> : std::true_type {}

#define DECLARE_MONADPLUS_CLASS(MP) \
    template<typename T> \
    struct mplus_result_type; \
    template<typename T> \
    using mplus_result_type_t = typename mplus_result_type<T>::type; \
    template<typename T> \
    struct mplus_result_type<MP<T> >{ \
        using type = MP<T>; \
    }; \
    template<typename T> \
    static MP<T> mzero(); \
    template<typename A> \
    static MP<A> mplus(MP<A> const& op1, MP<A> const& op2);

#define IMPLEMENT_DEFAULT_MONADPLUS(MP, _MP) \
    template<typename T> \
    MP<T> MonadPlus<_MP>::mzero(){ return Alternative<_MP>::template empty<T>(); } \
    template<typename T> \
    MP<T> MonadPlus<_MP>::mplus(MP<T> const& op1, MP<T> const& op2){ return op1 | op2; }

template<typename MP1, typename MP2>
using mplus_type = same_monad_plus_type<MP1, MP2, typename MonadPlus_t<MP1>::template mplus_result_type_t<MP2> >;

#define MPLUS_TYPE_(MP1, MP2) BOOST_IDENTITY_TYPE((mplus_type<MP1, MP2>))
#define MPLUS_TYPE(MP1, MP2) typename MPLUS_TYPE_(MP1, MP2)

DEFINE_FUNCTION_2(2, MPLUS_TYPE(T0, T1), mplus, T0 const&, x, T1 const&, y, return MonadPlus_t<T1>::mplus(x, y);)

_FUNCPROG_END
