#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<class T>
struct math_object_base
{
    typedef T final_type;

    __DEVICE __HOST
    final_type& self() {
        return static_cast<final_type&>(*this);
    }

    __DEVICE __HOST
    CONSTEXPR const final_type& self() const {
        return static_cast<const final_type&>(*this);
    }

    __DEVICE __HOST
    final_type& operator()(){
        return self();
    }

    __DEVICE __HOST
    CONSTEXPR const final_type& operator()() const {
        return self();
    }

protected: // protect from direct construction
    CONSTEXPR math_object_base() {}
};

template<class T, class _Proxy = T>
struct math_object : math_object_base<T>
{
    typedef _Proxy proxy_type;

    CONSTEXPR proxy_type get_proxy() const {
        return (*this)();
    }

protected: // protect from direct construction
    CONSTEXPR math_object() {}
};

_KIAM_MATH_END
