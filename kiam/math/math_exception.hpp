#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

class math_exception : public std::runtime_error
{
public:
	math_exception(const char *msg) : std::runtime_error(msg){}
	math_exception(const std::string &msg) : std::runtime_error(msg){}

	template<typename T>
	math_exception(const char *msg, const T &arg) : std::runtime_error(get_string(msg, arg)){}

	template<typename T>
	math_exception(const char *msg, const T &arg, const char *msg2) : std::runtime_error(get_string(msg, arg, msg2)){}

	template<typename T1, typename T2>
	math_exception(const char *msg, const T1 &arg1, const char *msg2, const T2 &arg2) : std::runtime_error(get_string(msg, arg1, msg2, arg2)){}

private:
	template<typename T>
	static std::string get_string(const char *msg, const T &arg)
	{
		std::ostringstream s;
		return static_cast<std::ostringstream&>(s << msg << arg).str();
	}

	template<typename T>
	static std::string get_string(const char *msg, const T &arg, const char *msg2)
	{
		std::ostringstream s;
		return static_cast<std::ostringstream&>(s << msg << arg << msg2).str();
	}

	template<typename T1, typename T2>
	static std::string get_string(const char *msg, const T1 &arg1, const char *msg2, const T2 &arg2)
	{
		std::ostringstream s;
		return static_cast<std::ostringstream&>(s << msg << arg1 << msg2 << arg2).str();
	}
};

_KIAM_MATH_END
