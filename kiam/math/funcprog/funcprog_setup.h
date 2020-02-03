#pragma once

#include <list>
#include <string>

#ifndef FUNCPROG_USE_CPP17
#if __cplusplus >= 201700 || _HAS_CXX17
#define FUNCPROG_USE_CPP17 1
#endif
#endif

#include <functional>

#if FUNCPROG_USE_CPP17
#include <optional>
#include <variant>
#else
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#endif

#include "../math_def.h"

#define _FUNCPROG_BEGIN _KIAM_MATH_BEGIN namespace funcprog {
#define _FUNCPROG_END } _KIAM_MATH_END

#define _FUNCPROG _KIAM_MATH::funcprog

_FUNCPROG_BEGIN

template<typename FuncType> using function_t = std::function<FuncType>;

#if FUNCPROG_USE_CPP17
    template<typename T> using optional_t = std::optional<T>;
    template<typename... Types> using variant_base = std::variant<Types...>;
#else
    template<typename T> using optional_t = boost::optional<T>;
    template<typename... Types> using variant_base = boost::variant<Types...>;
#endif

template<typename... Types>
struct variant_t : variant_base<Types...>
{
    using super = variant_base<Types...>;

    variant_t(variant_t const& other) : super(other) {}
    template<typename T> variant_t(T const& value) : super(value) {}

    size_t index() const {
#if FUNCPROG_USE_CPP17
        return super::index();
#else
        return super::which();
#endif
    }

    template<typename T>
    T const& get() const {
#if FUNCPROG_USE_CPP17
        return std::get<T>(*this);
#else
        return boost::get<T>(*this);
#endif
    }

};

template<typename T>
using list_t = std::list<T>;

_FUNCPROG_END
