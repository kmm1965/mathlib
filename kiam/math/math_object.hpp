#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<class T>
struct math_object_base
{
    typedef T final_type;

    __DEVICE __HOST
    final_type& self(){
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
    CONSTEXPR __DEVICE __HOST math_object_base(){}
};

template<class T, class _Proxy = T>
struct math_object : math_object_base<T>
{
    typedef _Proxy proxy_type;

    proxy_type get_proxy(){
        return proxy_type((*this)());
    }

    CONSTEXPR proxy_type get_proxy() const {
        return proxy_type((*this)());
    }

protected: // Protect from direct construction
    CONSTEXPR __DEVICE __HOST math_object(){}
};

#define MOBJ(MO) math_object<MO, typename MO::proxy_type>

template<typename T>
struct proxy_type {
    using type = T;
};

template<typename T>
using proxy_type_t = typename proxy_type<T>::type;

template<typename MO>
struct proxy_type<MOBJ(MO)>
{
    using type = typename MO::proxy_type;
};

template<typename T>
struct math_vector;

template<typename T>
struct vector_proxy;

template<typename T>
struct proxy_type<math_vector<T> >
{
    using type = vector_proxy<T>;
};

template<typename T>
proxy_type_t<T> get_proxy(T const& t);

template<typename MO>
typename MO::proxy_type get_proxy(MOBJ(MO) const& mo){
    return mo.get_proxy();
}

template<typename T>
vector_proxy<T> get_proxy(math_vector<T> const& vec){
    return vector_proxy<T>(vec);
}

_KIAM_MATH_END
