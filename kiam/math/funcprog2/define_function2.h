#pragma once

#include "funcprog2_setup.h"

#include <boost/preprocessor/enum_params.hpp>
#include <boost/utility/identity_type.hpp>

#define FUNCTION_TEMPLATE(ntempl) template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)>
#define FUNCTION_TEMPLATE_ARGS(ntempl) template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args>

// FUNCTION_2
#define DECLARE_FUNCTION_2(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr Ret name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1); }); }

// FUNCTION_2_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NC(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE Ret name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1); }); }

// FUNCTION_2_ARGS
#define DECLARE_FUNCTION_2_ARGS(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr Ret name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([=](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1); }); }

// FUNCTION_2_ARGS_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NC_ARGS(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE Ret name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1); }); }

// FUNCTION_2_NOTEMPL
#define DECLARE_FUNCTION_2_NOTEMPL(Ret, name, type0, type1) \
    inline __DEVICE constexpr Ret name(type0, type1); \
    inline __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1){ return name(p0, p1); }); }

// FUNCTION_2_NOTEMPL_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NOTEMPL_NC(Ret, name, type0, type1) \
    inline __DEVICE Ret name(type0, type1); \
    inline __DEVICE auto _##name(type0 p0){ return _([p0](type1 p1){ return name(p0, p1); });}

// FUNCTION_3
#define DECLARE_FUNCTION_3(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); }); }

// FUNCTION_3_NC (Non constexpr version)
#define DECLARE_FUNCTION_3_NC(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); }); }

// FUNCTION_3_ARGS
#define DECLARE_FUNCTION_3_ARGS(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2); }); }; \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2); }); }

// FUNCTION_3_ARGS_NC (Non constexpr version)
#define DECLARE_FUNCTION_3_ARGS_NC(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2); }); } \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2); }); }

// FUNCTION_3_NOTEMPL
#define DECLARE_FUNCTION_3_NOTEMPL(Ret, name, type0, type1, type2) \
    inline __DEVICE constexpr Ret name(type0, type1, type2); \
    inline __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([=](type2 p2){ return name(p0, p1, p2); };) } \
    inline __DEVICE constexpr auto _##name(type0 p0){ \
        return _([=](type1 p1, type2 p2){ return name(p0, p1, p2); }); }

// FUNCTION_3_NOTEMPL_NC (Non constexpr version)
#define DECLARE_FUNCTION_3_NOTEMPL_NC(Ret, name, type0, type1, type2) \
    inline __DEVICE Ret name(type0, type1, type2); \
    inline __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([=](type2 p2){ return name(p0, p1, p2); }); } \
    inline __DEVICE auto _##name(type0 p0){ \
        return _([=](type1 p1, type2 p2){ return name(p0, p1, p2); }); }

// FUNCTION_4
#define DECLARE_FUNCTION_4(ntempl, Ret, name, type0, type1, type2, type3) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr Ret name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); }

// FUNCTION_4_NC (Non constexpr version)
#define DECLARE_FUNCTION_4_NC(ntempl, Ret, name, type0, type1, type2, type3) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE Ret name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); }); }

// FUNCTION_4_ARGS
#define DECLARE_FUNCTION_4_ARGS(ntempl, Ret, name, type0, type1, type2, type3) \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr Ret name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2, p3); }); } \
    FUNCTION_TEMPLATE_ARGS(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2, p3); }); }

// FUNCTION_5
#define DECLARE_FUNCTION_5(ntempl, Ret, name, type0, type1, type2, type3, type4) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr Ret name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return _([p0, p1, p2, p3](type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); }

// FUNCTION_5_NC (Non constexpr version)
#define DECLARE_FUNCTION_5_NC(ntempl, Ret, name, type0, type1, type2, type3, type4) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE Ret name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return _([p0, p1, p2, p3](type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); }); }

// FUNCTION_6
#define DECLARE_FUNCTION_6(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr Ret name(type0, type1, type2, type3, type4, type5); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ \
        return _([p0, p1, p2, p3, p4](type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return _([p0, p1, p2, p3](type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE constexpr auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); }

// FUNCTION_6_NC (Non constexpr version)
#define DECLARE_FUNCTION_6_NC(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
    FUNCTION_TEMPLATE(ntempl) __DEVICE Ret name(type0, type1, type2, type3, type4, type5); \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ \
        return _([p0, p1, p2, p3, p4](type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return _([p0, p1, p2, p3](type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1, type2 p2){ \
        return _([p0, p1, p2](type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0, type1 p1){ \
        return _([p0, p1](type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); } \
    FUNCTION_TEMPLATE(ntempl) __DEVICE auto _##name(type0 p0){ \
        return _([p0](type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); }); }
