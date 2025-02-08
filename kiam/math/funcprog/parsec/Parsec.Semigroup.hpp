#pragma once

#include "../Semigroup.hpp"

_FUNCPROG_BEGIN

//instance Semigroup.Semigroup a => Semigroup.Semigroup (ParsecT s u m a) where
//    -- | Combines two parsers like '*>', '>>' and @do {...;...}@
//    --  /but/ also combines their results with (<>) instead of
//    --  discarding the first.
//    (<>)     = Applicative.liftA2 (Semigroup.<>)

// Semigroup
template<typename S, typename U, typename _M>
struct Semigroup<parsec::_ParsecT<S, U, _M> >
{
    template<typename A, typename PX, typename PY>
    static constexpr auto sg_op(parsec::ParsecT<S, U, _M, A, PX> const& px, parsec::ParsecT<S, U, _M, A, PY> const& py)
    {
        static_assert(is_semigroup_v<A>, "Should be a Semigroup");
        return Applicative<parsec::_ParsecT<S, U, _M> >::template liftA2<PX, PY>(_(Semigroup_t<A>::template sg_op<value_type_t<A> >))(px, py);
    }
};

template<typename S, typename U, typename _M, typename A, typename PX, typename PY>
constexpr auto operator%(parsec::ParsecT<S, U, _M, A, PX> const& x, parsec::ParsecT<S, U, _M, A, PY> const& y){
    return Semigroup<parsec::_ParsecT<S, U, _M> >::sg_op(x, y);
}

_FUNCPROG_END
