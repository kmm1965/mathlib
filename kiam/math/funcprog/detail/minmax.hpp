#pragma once

_FUNCPROG_BEGIN

template<typename T>
auto max(){
    return _([](T const& l, T const& r){
        return std::max(l, r);
    });
}

template<typename T>
auto min(){
    return _([](T const& l, T const& r){
        return std::min(l, r);
    });
}

_FUNCPROG_END
