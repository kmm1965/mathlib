#pragma once

#include "int_constant.hpp"

_SYMDIFF1_BEGIN

struct variable : expression<variable>
{
	typedef int_constant<1> diff_type;

	diff_type diff() const
	{
		return diff_type();
	}

	template<typename T>
	T operator()(const T &x) const {
		return x;
	}

    std::string to_string() const
    {
        return "x";
    }
};

_SYMDIFF1_END
