#pragma once

#include "funcprog_setup.h"

#include <boost/preprocessor/enum_params.hpp>
#include <boost/utility/identity_type.hpp>

// FUNCTION_2
#define DECLARE_FUNCTION_2_IMPL(ntempl, Ret, name, type0, type1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0, type1)

#define DECLARE_FUNCTION_2_1(ntempl, Ret, name, type0, type1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1)> _##name(type0)

#define DECLARE_FUNCTION_2(ntempl, Ret, name, type0, type1) \
	DECLARE_FUNCTION_2_IMPL(ntempl, Ret, name, type0, type1); \
	DECLARE_FUNCTION_2_1(ntempl, Ret, name, type0, type1)

#define DEFINE_FUNCTION_2_IMPL(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0 p0, type1 p1){ impl }

#define DEFINE_FUNCTION_2_1(ntempl, Ret, name, type0, p0, type1, p1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1)> _##name(type0 p0){ \
		return [p0](type1 p1){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1); \
		}; \
	}

#define DEFINE_FUNCTION_2(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_IMPL(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_1(ntempl, Ret, name, type0, p0, type1, p1)

// FUNCTION_2_ARGS
#define DECLARE_FUNCTION_2_ARGS_IMPL(ntempl, Ret, name, type0, type1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	Ret name(type0, type1)

#define DECLARE_FUNCTION_2_ARGS_1(ntempl, Ret, name, type0, type1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type1)> _##name(type0)

#define DECLARE_FUNCTION_2_ARGS(ntempl, Ret, name, type0, type1) \
	DECLARE_FUNCTION_2_ARGS_IMPL(ntempl, Ret, name, type0, type1); \
	DECLARE_FUNCTION_2_ARGS_1(ntempl, Ret, name, type0, type1)

#define DEFINE_FUNCTION_2_ARGS_IMPL(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	Ret name(type0 p0, type1 p1){ impl }

#define DEFINE_FUNCTION_2_ARGS_1(ntempl, Ret, name, type0, p0, type1, p1) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type1)> _##name(type0 p0){ \
		return [p0](type1 p1){ \
			return name(p0, p1); \
		}; \
	}

#define DEFINE_FUNCTION_2_ARGS(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_ARGS_IMPL(ntempl, Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_ARGS_1(ntempl, Ret, name, type0, p0, type1, p1)

// FUNCTION_2_NOTEMPL
#define DECLARE_FUNCTION_2_NOTEMPL_IMPL(Ret, name, type0, type1) \
	inline Ret name(type0, type1)

#define DECLARE_FUNCTION_2_NOTEMPL_1(Ret, name, type0, type1) \
	inline function_t<Ret(type1)> _##name(type0)

#define DECLARE_FUNCTION_2_NOTEMPL(Ret, name, type0, type1) \
	DECLARE_FUNCTION_2_NOTEMPL_IMPL(Ret, name, type0, type1); \
	DECLARE_FUNCTION_2_NOTEMPL_1(Ret, name, type0, type1)

#define DEFINE_FUNCTION_2_NOTEMPL_IMPL(Ret, name, type0, p0, type1, p1, impl) \
	inline Ret name(type0 p0, type1 p1){ impl }

#define DEFINE_FUNCTION_2_NOTEMPL_1(Ret, name, type0, p0, type1, p1) \
	inline function_t<Ret(type1)> _##name(type0 p0){ \
		return [p0](type1 p1){ \
			return name(p0, p1); \
		}; \
	}

#define DEFINE_FUNCTION_2_NOTEMPL(Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_NOTEMPL_IMPL(Ret, name, type0, p0, type1, p1, impl) \
	DEFINE_FUNCTION_2_NOTEMPL_1(Ret, name, type0, p0, type1, p1)

// FUNCTION_3
#define DECLARE_FUNCTION_3_IMPL(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0, type1, type2)

#define DECLARE_FUNCTION_3_2(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2)> _##name(type0, type1)

#define DECLARE_FUNCTION_3_1(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2)> _##name(type0)

#define DECLARE_FUNCTION_3(ntempl, Ret, name, type0, type1, type2) \
	DECLARE_FUNCTION_3_IMPL(ntempl, Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_2(ntempl, Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_1(ntempl, Ret, name, type0, type1, type2)

#define DEFINE_FUNCTION_3_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0 p0, type1 p1, type2 p2){ impl }

#define DEFINE_FUNCTION_3_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	DEFINE_FUNCTION_3_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2)

// FUNCTION_3_ARGS
#define DECLARE_FUNCTION_3_ARGS_IMPL(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	Ret name(type0, type1, type2)

#define DECLARE_FUNCTION_3_ARGS_2(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type2)> _##name(type0, type1)

#define DECLARE_FUNCTION_3_ARGS_1(ntempl, Ret, name, type0, type1, type2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type1, type2)> _##name(type0)

#define DECLARE_FUNCTION_3_ARGS(ntempl, Ret, name, type0, type1, type2) \
	DECLARE_FUNCTION_3_ARGS_IMPL(ntempl, Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_ARGS_2(ntempl, Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_ARGS_1(ntempl, Ret, name, type0, type1, type2)

#define DEFINE_FUNCTION_3_ARGS_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	Ret name(type0 p0, type1 p1, type2 p2){ impl }

#define DEFINE_FUNCTION_3_ARGS_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3_ARGS_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T), typename... Args> \
	function_t<Ret(type1, type2)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3_ARGS(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_ARGS_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_ARGS_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2) \
	DEFINE_FUNCTION_3_ARGS_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2)

// FUNCTION_3_NOTEMPL
#define DECLARE_FUNCTION_3_NOTEMPL_IMPL(Ret, name, type0, type1, type2) \
	inline Ret name(type0, type1, type2)

#define DECLARE_FUNCTION_NOTEMPL_3_2(Ret, name, type0, type1, type2) \
	inline function_t<Ret(type2)> _##name(type0, type1)

#define DECLARE_FUNCTION_3_NOTEMPL_1(Ret, name, type0, type1, type2) \
	inline function_t<Ret(type1, type2)> _##name(type0)

#define DECLARE_FUNCTION_3_NOTEMPL(Ret, name, type0, type1, type2) \
	DECLARE_FUNCTION_3_NOTEMPL_IMPL(Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_NOTEMPL_2(Ret, name, type0, type1, type2); \
	DECLARE_FUNCTION_3_NOTEMPL_1(Ret, name, type0, type1, type2)

#define DEFINE_FUNCTION_3_NOTEMPL_IMPL(Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	inline Ret name(type0 p0, type1 p1, type2 p2){ impl }

#define DEFINE_FUNCTION_3_NOTEMPL_2(Ret, name, type0, p0, type1, p1, type2, p2) \
	inline function_t<Ret(type2)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2){ \
			return name(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3_NOTEMPL_1(Ret, name, type0, p0, type1, p1, type2, p2) \
	inline function_t<Ret(type1, type2)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2){ \
			return name(p0, p1, p2); \
		}; \
	}

#define DEFINE_FUNCTION_3_NOTEMPL(Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_NOTEMPL_IMPL(Ret, name, type0, p0, type1, p1, type2, p2, impl) \
	DEFINE_FUNCTION_3_NOTEMPL_2(Ret, name, type0, p0, type1, p1, type2, p2) \
	DEFINE_FUNCTION_3_NOTEMPL_1(Ret, name, type0, p0, type1, p1, type2, p2)

// FUNCTION_4
#define DECLARE_FUNCTION_4_IMPL(ntempl, Ret, name, type0, type1, type2, type3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0, type1, type2, type3)

#define DECLARE_FUNCTION_4_3(ntempl, Ret, name, type0, type1, type2, type3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3)> _##name(type0, type1, type2)

#define DECLARE_FUNCTION_4_2(ntempl, Ret, name, type0, type1, type2, type3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3)> _##name(type0, type1)

#define DECLARE_FUNCTION_4_1(ntempl, Ret, name, type0, type1, type2, type3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3)> _##name(type0)

#define DECLARE_FUNCTION_4(ntempl, Ret, name, type0, type1, type2, type3) \
	DECLARE_FUNCTION_4_IMPL(ntempl, Ret, name, type0, type1, type2, type3); \
	DECLARE_FUNCTION_4_3(ntempl, Ret, name, type0, type1, type2, type3); \
	DECLARE_FUNCTION_4_2(ntempl, Ret, name, type0, type1, type2, type3); \
	DECLARE_FUNCTION_4_1(ntempl, Ret, name, type0, type1, type2, type3)

#define DEFINE_FUNCTION_4_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0 p0, type1 p1, type2 p2, type3 p3){ impl	}

#define DEFINE_FUNCTION_4_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3)> _##name(type0 p0, type1 p1, type2 p2){ \
		return [=](type3 p3){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); \
		}; \
	}

#define DEFINE_FUNCTION_4_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2, type3 p3){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); \
		}; \
	}

#define DEFINE_FUNCTION_4_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2, type3 p3){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3); \
		}; \
	}

#define DEFINE_FUNCTION_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, impl) \
	DEFINE_FUNCTION_4_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, impl) \
	DEFINE_FUNCTION_4_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3) \
	DEFINE_FUNCTION_4_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3) \
	DEFINE_FUNCTION_4_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3)

// FUNCTION_5
#define DECLARE_FUNCTION_5_IMPL(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0, type1, type2, type3, type4)

#define DECLARE_FUNCTION_5_4(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type4)> _##name(type0, type1, type2, type3)

#define DECLARE_FUNCTION_5_3(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3, type4)> _##name(type0, type1, type2)

#define DECLARE_FUNCTION_5_2(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3, type4)> _##name(type0, type1)

#define DECLARE_FUNCTION_5_1(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3, type4)> _##name(type0)

#define DECLARE_FUNCTION_5(ntempl, Ret, name, type0, type1, type2, type3, type4) \
	DECLARE_FUNCTION_5_IMPL(ntempl, Ret, name, type0, type1, type2, type3, type4); \
	DECLARE_FUNCTION_5_4(ntempl, Ret, name, type0, type1, type2, type3, type4); \
	DECLARE_FUNCTION_5_3(ntempl, Ret, name, type0, type1, type2, type3, type4); \
	DECLARE_FUNCTION_5_2(ntempl, Ret, name, type0, type1, type2, type3, type4); \
	DECLARE_FUNCTION_5_1(ntempl, Ret, name, type0, type1, type2, type3, type4)

#define DEFINE_FUNCTION_5_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ impl	}

#define DEFINE_FUNCTION_5_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type4)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
		return [=](type4 p4){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); \
		}; \
	}

#define DEFINE_FUNCTION_5_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3, type4)> _##name(type0 p0, type1 p1, type2 p2){ \
		return [=](type3 p3, type4 p4){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); \
		}; \
	}

#define DEFINE_FUNCTION_5_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3, type4)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2, type3 p3, type4 p4){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); \
		}; \
	}

#define DEFINE_FUNCTION_5_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3, type4)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2, type3 p3, type4 p4){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4); \
		}; \
	}

#define DEFINE_FUNCTION_5(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, impl) \
	DEFINE_FUNCTION_5_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, impl) \
	DEFINE_FUNCTION_5_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	DEFINE_FUNCTION_5_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	DEFINE_FUNCTION_5_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4) \
	DEFINE_FUNCTION_5_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4)

// FUNCTION_6
#define DECLARE_FUNCTION_6_IMPL(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0, type1, type2, type3, type4, type5)

#define DECLARE_FUNCTION_6_5(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type5)> _##name(type0, type1, type2, type3, type4)

#define DECLARE_FUNCTION_6_4(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type4, type5)> _##name(type0, type1, type2, type3)

#define DECLARE_FUNCTION_6_3(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3, type4, type5)> _##name(type0, type1, type2)

#define DECLARE_FUNCTION_6_2(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3, type4, type5)> _##name(type0, type1)

#define DECLARE_FUNCTION_6_1(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0)

#define DECLARE_FUNCTION_6(ntempl, Ret, name, type0, type1, type2, type3, type4, type5) \
	DECLARE_FUNCTION_6_IMPL(ntempl, Ret, name, type0, type1, type2, type3, type4, type5); \
	DECLARE_FUNCTION_6_5(ntempl, Ret, name, type0, type1, type2, type3, type4, type5); \
	DECLARE_FUNCTION_6_4(ntempl, Ret, name, type0, type1, type2, type3, type4, type5); \
	DECLARE_FUNCTION_6_3(ntempl, Ret, name, type0, type1, type2, type3, type4, type5); \
	DECLARE_FUNCTION_6_2(ntempl, Ret, name, type0, type1, type2, type3, type4, type5); \
	DECLARE_FUNCTION_6_1(ntempl, Ret, name, type0, type1, type2, type3, type4, type5)

#define DEFINE_FUNCTION_6_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5, impl) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	Ret name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ impl	}

#define DEFINE_FUNCTION_6_5(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3, type4 p4){ \
		return [=](type5 p5){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); \
		}; \
	}

#define DEFINE_FUNCTION_6_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type4, type5)> _##name(type0 p0, type1 p1, type2 p2, type3 p3){ \
		return [=](type4 p4, type5 p5){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); \
		}; \
	}

#define DEFINE_FUNCTION_6_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type3, type4, type5)> _##name(type0 p0, type1 p1, type2 p2){ \
		return [=](type3 p3, type4 p4, type5 p5){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); \
		}; \
	}

#define DEFINE_FUNCTION_6_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type2, type3, type4, type5)> _##name(type0 p0, type1 p1){ \
		return [=](type2 p2, type3 p3, type4 p4, type5 p5){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); \
		}; \
	}

#define DEFINE_FUNCTION_6_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	template<BOOST_PP_ENUM_PARAMS(ntempl, typename T)> \
	function_t<Ret(type1, type2, type3, type4, type5)> _##name(type0 p0){ \
		return [=](type1 p1, type2 p2, type3 p3, type4 p4, type5 p5){ \
			return name<BOOST_PP_ENUM_PARAMS(ntempl, T)>(p0, p1, p2, p3, p4, p5); \
		}; \
	}

#define DEFINE_FUNCTION_6(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5, impl) \
	DEFINE_FUNCTION_6_IMPL(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5, impl) \
	DEFINE_FUNCTION_6_5(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	DEFINE_FUNCTION_6_4(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	DEFINE_FUNCTION_6_3(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	DEFINE_FUNCTION_6_2(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5) \
	DEFINE_FUNCTION_6_1(ntempl, Ret, name, type0, p0, type1, p1, type2, p2, type3, p3, type4, p4, type5, p5)
