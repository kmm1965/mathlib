#pragma once

#include "funcprog_setup.h"

#include <boost/preprocessor/enum_params.hpp>
#include <boost/utility/identity_type.hpp>

#define FUNCTION_TEMPLATE(ntempl) template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)>
#define FUNCTION_TEMPLATE_ARGS(ntempl) template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args>

// FUNCTION_2
#define DECLARE_FUNCTION_2(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_2(ntempl, Ret, name, type0, p0, type1, p1, impl) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0 p0, type1 p1){ impl } \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1)> _##name(type0 p0){ \
        return [p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1);};}

// FUNCTION_2_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NC(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_NC_2(ntempl, Ret, name, type0, p0, type1, p1, impl) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0 p0, type1 p1){ impl } \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1)> _##name(type0 p0){ \
        return [p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1);};}

// FUNCTION_2_ARGS
#define DECLARE_FUNCTION_2_ARGS(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr Ret name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_2_ARGS(ntempl, Ret, name, type0, p0, type1, p1, impl) \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr Ret name(type0 p0, type1 p1){ impl } \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type1)> _##name(type0 p0){ \
        return [=](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1);};}

// FUNCTION_2_ARGS_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NC_ARGS(ntempl, Ret, name, type0, type1) \
    FUNCTION_TEMPLATE_ARGS(ntempl) Ret name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_2_NC_ARGS(ntempl, Ret, name, type0, p0, type1, p1, impl) \
    FUNCTION_TEMPLATE_ARGS(ntempl) Ret name(type0 p0, type1 p1){ impl } \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type1)> _##name(type0 p0){ \
        return [p0](type1 p1){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1);};}

// FUNCTION_2_NOTEMPL
#define DECLARE_FUNCTION_2_NOTEMPL(Ret, name, type0, type1) \
    constexpr Ret name(type0, type1); \
    constexpr function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_2_NOTEMPL(Ret, name, type0, p0, type1, p1, impl) \
    constexpr Ret name(type0 p0, type1 p1){ impl } \
    constexpr function_t<Ret(type1)> _##name(type0 p0){ \
        return [p0](type1 p1){ return name(p0, p1);};}

// FUNCTION_2_NOTEMPL_NC (Non constexpr version)
#define DECLARE_FUNCTION_2_NOTEMPL_NC(Ret, name, type0, type1) \
    Ret name(type0, type1); \
    function_t<Ret(type1)> _##name(type0)

#define DEFINE_FUNCTION_2_NOTEMPL_NC(Ret, name, type0, p0, type1, p1, impl) \
    Ret name(type0 p0, type1 p1){ impl } \
    function_t<Ret(type1)> _##name(type0 p0){ return [p0](type1 p1){ return name(p0, p1); };}

// FUNCTION_3
#define DECLARE_FUNCTION_3(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2);};}

// FUNCTION_3_NC (Non constexpr version)
#define DECLARE_FUNCTION_NC_3(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3_NC(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2);};}

// FUNCTION_3_ARGS
#define DECLARE_FUNCTION_3_ARGS(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type2)> _##name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3_ARGS(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2);};} \
    FUNCTION_TEMPLATE_ARGS(ntempl) constexpr function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2);};}

// FUNCTION_3_ARGS_NC (Non constexpr version)
#define DECLARE_FUNCTION_3_ARGS_NC(ntempl, Ret, name, type0, type1, type2) \
    FUNCTION_TEMPLATE_ARGS(ntempl) Ret name(type0, type1, type2); \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type2)> _##name(type0, type1); \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3_ARGS_NC(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    FUNCTION_TEMPLATE_ARGS(ntempl) Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2);};} \
    FUNCTION_TEMPLATE_ARGS(ntempl) function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T), Args...>(p0, p1, p2);};}

// FUNCTION_3_NOTEMPL
#define DECLARE_FUNCTION_3_NOTEMPL(Ret, name, type0, type1, type2) \
    constexpr Ret name(type0, type1, type2); \
    constexpr function_t<Ret(type2)> _##name(type0, type1); \
    constexpr function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3_NOTEMPL(Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    constexpr Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    constexpr function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [=](type2 p2){ return name(p0, p1, p2); };} \
    constexpr function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [=](type1 p1, type2 p2){ return name(p0, p1, p2); };}

// FUNCTION_3_NOTEMPL_NC (Non constexpr version)
#define DECLARE_FUNCTION_3_NOTEMPL_NC(Ret, name, type0, type1, type2) \
    Ret name(type0, type1, type2); \
    function_t<Ret(type2)> _##name(type0, type1); \
    function_t<Ret(type1, type2)> _##name(type0)

#define DEFINE_FUNCTION_3_NOTEMPL_NC(Ret, name, type0, p0, type1, p1, type2, p2, impl) \
    Ret name(type0 p0, type1 p1, type2 p2){ impl } \
    function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
        return [=](type2 p2){ return name(p0, p1, p2); };} \
    function_t<Ret(type1, type2)> _##name(type0 p0){ \
        return [=](type1 p1, type2 p2){ return name(p0, p1, p2); };}

// FUNCTION_4
#define DECLARE_FUNCTION_4(ntempl, Ret, name, type0, type1, type2, type3) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3)> _##name(type0)

#define DEFINE_FUNCTION_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, impl) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0 p0, type1 p1, type2 p2, type3 p3){ impl } \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};}

// FUNCTION_4_NC (Non constexpr version)
#define DECLARE_FUNCTION_4_NC(ntempl, Ret, name, type0, type1, type2, type3) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3)> _##name(type0)

#define DEFINE_FUNCTION_4_NC(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, impl) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0 p0, type1 p1, type2 p2, type3 p3){ impl } \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3);};}

// FUNCTION_5
#define DECLARE_FUNCTION_5(ntempl, Ret, name, type0, type1, type2, type3, type4) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type4)> _##name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3, type4)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3, type4)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3, type4)> _##name(type0)

#define DEFINE_FUNCTION_5(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, impl) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ impl } \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type4)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return [p0, p1, p2, p3](type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3, type4)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3, type4)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3, type4)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};}

// FUNCTION_5_NC (Non constexpr version)
#define DECLARE_FUNCTION_5_NC(ntempl, Ret, name, type0, type1, type2, type3, type4) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type4)> _##name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3, type4)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3, type4)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3, type4)> _##name(type0)

#define DEFINE_FUNCTION_NC_5(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, impl) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ impl } \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type4)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return [p0, p1, p2, p3](type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3, type4)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3, type4)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3, type4)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3, type4 p4){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4);};}

// FUNCTION_6
#define DECLARE_FUNCTION_6(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0, type1, type2, type3, type4, type5); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type5)> _##name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type4, type5)> _##name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3, type4, type5)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3, type4, type5)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0)

#define DEFINE_FUNCTION_6(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5, impl) \
    FUNCTION_TEMPLATE(ntempl) constexpr Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ impl } \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ \
        return [p0, p1, p2, p3, p4](type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type4, type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return [p0, p1, p2, p3](type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type3, type4, type5)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type2, type3, type4, type5)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) constexpr function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};}

// FUNCTION_6_NC (Non constexpr version)
#define DECLARE_FUNCTION_NC_6(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0, type1, type2, type3, type4, type5); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type5)> _##name(type0, type1, type2, type3, type4); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type4, type5)> _##name(type0, type1, type2, type3); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3, type4, type5)> _##name(type0, type1, type2); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3, type4, type5)> _##name(type0, type1); \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0)

#define DEFINE_FUNCTION_6_NC(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5, impl) \
    FUNCTION_TEMPLATE(ntempl) Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ impl } \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ \
        return [p0, p1, p2, p3, p4](type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type4, type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
        return [p0, p1, p2, p3](type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type3, type4, type5)> _##name(type0 p0, type1 p1, type2 p2){ \
        return [p0, p1, p2](type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type2, type3, type4, type5)> _##name(type0 p0, type1 p1){ \
        return [p0, p1](type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};} \
    FUNCTION_TEMPLATE(ntempl) function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0 p0){ \
        return [p0](type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5);};}
