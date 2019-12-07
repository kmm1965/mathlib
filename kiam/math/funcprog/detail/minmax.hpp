#pragma once

_FUNCPROG_BEGIN

template<typename T>
function_t<T(T const&, T const&)> max(){
	return [](T const& l, T const& r){
		return std::max(l, r);
	};
}

template<typename T>
function_t<T(T const&, T const&)> min(){
	return [](T const& l, T const& r){
		return std::min(l, r);
	};
}

template<typename T>
T maximum(List<T> const& l) {
	return foldl1(max<T>(), l);
}

template<typename T>
T minimum(List<T> const& l) {
	return foldl1(min<T>(), l);
}

_FUNCPROG_END
