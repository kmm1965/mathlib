#pragma once

#include "Parsec.Pos.hpp"

_PARSEC_BEGIN

template<typename S, typename U>
struct State
{
    using stream_type = S;
    using user_state_type = U;

    State(stream_type const& input, SourcePos const& pos, user_state_type const& user) : input(input), pos(pos), user(user){}

    stream_type const input;
    SourcePos const pos;
    user_state_type const user;
};

#define STATE_(S, U) BOOST_IDENTITY_TYPE((_PARSEC::State<S, U>))
#define STATE(S, U) typename STATE_(S, U)

template<typename S, typename U>
SourcePos const& statePos(State<S, U> const& state){
    return state.pos;
}

_PARSEC_END
