#pragma once

struct tag_main;
struct tag_aux;

template<typename T>
struct other_tag;

template<>
struct other_tag<tag_main>
{
	typedef tag_aux type;
};

template<>
struct other_tag<tag_aux>
{
	typedef tag_main type;
};

template<typename T>
using other_tag_t = typename other_tag<T>::type;
