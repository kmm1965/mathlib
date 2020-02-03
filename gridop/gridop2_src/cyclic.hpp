#pragma once

struct tag_main {};
struct tag_aux {};

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

typedef std::tuple<tag_main, tag_main> tag_cntr;
typedef std::tuple<tag_aux, tag_aux> tag_node;
typedef std::tuple<tag_aux, tag_main> tag_edge_x;
typedef std::tuple<tag_main, tag_aux> tag_edge_y;
