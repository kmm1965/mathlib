#pragma once

_FUNCPROG2_BEGIN

template<typename T>
__DEVICE auto max(){
    return _([](T const& l, T const& r){
        return std::max(l, r);
    });
}

template<typename T>
__DEVICE auto min(){
    return _([](T const& l, T const& r){
        return std::min(l, r);
    });
}

_FUNCPROG2_END
