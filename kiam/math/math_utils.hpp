#pragma once

#if _HAS_CXX17
#include <execution>
#define EXECUTION_POLICY std::execution::par,
//#define EXECUTION_POLICY /* serial version */
#else
#define EXECUTION_POLICY
#endif

#if defined(__CUDACC__)
#include "cuda_math_utils.hpp"
#elif defined(__OPENCL__)
#include "opencl_math_utils.hpp"
#else
#include "cpu_math_utils.hpp"
#endif

_KIAM_MATH_BEGIN

template<typename T>
__DEVICE __HOST
T det2(T J00, T J01, T J10, T J11){
	return J00 * J11 - J10 * J01;
}

template<typename T>
__DEVICE __HOST
T det3(T J00, T J01, T J02, T J10, T J11, T J12, T J20, T J21, T J22){
	return
		J00 * det2(J11, J12, J21, J22) -
		J01 * det2(J10, J12, J20, J22) +
		J02 * det2(J10, J11, J20, J21);
}

template<typename It1, typename It2>
struct pair_iterator
{
    using value_type1 = typename std::iterator_traits<It1>::value_type;
    using value_type2 = typename std::iterator_traits<It2>::value_type;

    pair_iterator(It1 it1, It2 it2) : it1(it1), it2(it2) {}

    pair_iterator& operator++()
    {
        ++it1;
        ++it2;
        return *this;
    }

    pair_iterator operator++(int)
    {
        pair_iterator result(*this);
        ++(*this);
        return result;
    }

    math_pair<value_type1&, value_type2&> operator*() {
        return math_pair<value_type1&, value_type2&>(*it1, *it2);
    }

private:
    It1 it1;
    It2 it2;
};

template<typename It1, typename It2>
pair_iterator<It1, It2> make_pair_iterator(It1 it1, It2 it2) {
    return pair_iterator<It1, It2>(it1, it2);
}

_KIAM_MATH_END
