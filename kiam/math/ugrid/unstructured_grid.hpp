#pragma once

#include "ugrid_def.h"
#include "../math_object.hpp"
#include "../math_vector.hpp"
#include "../math_array.hpp"
#include "ugrid_geometry.hpp"

_UGRID_MATH_BEGIN

#ifndef UGRID_CELL_H
#define UGRID_CELL_H 0
#endif

#ifndef UGRID_CELL_TYPE
#define UGRID_CELL_TYPE 0
#endif

#ifndef UGRID_CELL_B_CENTER
#define UGRID_CELL_B_CENTER 0
#endif

#ifndef UGRID_CELL_B_BC_CELL
#define UGRID_CELL_B_BC_CELL 0
#endif

#ifndef UGRID_CELL_B_PATCH
#define UGRID_CELL_B_PATCH 0
#endif

#ifndef UGRID_INTERFACE_AREA
#define UGRID_INTERFACE_AREA 0
#endif

#ifndef UGRID_INTERFACE_CENTER
#define UGRID_INTERFACE_CENTER 0
#endif

template<typename T, unsigned DIM>
struct unstructured_grid_proxy;

template<typename T, unsigned DIM>
struct unstructured_grid : _KIAM_MATH::math_object<unstructured_grid<T, DIM>, unstructured_grid_proxy<T, DIM> >
{
	static_assert(DIM == 2 || DIM == 3, "Dimention of the grid should be 2 or 3");
	typedef T value_type;
	static const unsigned dim = DIM;
	typedef unstructured_grid type;
	typedef _KIAM_MATH::math_object<type, unstructured_grid_proxy<value_type, dim> > super;

	using vector_t = _KIAM_MATH::vector_type_t<value_type, dim>;

	template<unsigned DIM_>
	static typename std::enable_if<DIM_ == 2, vector_t>::type get_max_value(){
		return vector_t(std::numeric_limits<value_type>::max(), std::numeric_limits<value_type>::max());
	}

	template<unsigned DIM_>
	static typename std::enable_if<DIM_ == 3, vector_t>::type get_max_value(){
		return vector_t(std::numeric_limits<value_type>::max(), std::numeric_limits<value_type>::max(), std::numeric_limits<value_type>::max());
	}

    unstructured_grid() : m_ptMin(), m_ptMax(), m_dxMin(get_max_value<dim>())
#if UGRID_CELL_H
		, m_hMin(std::numeric_limits<value_type>::max())
#endif
	{}

	struct cell_value_type
	{
		value_type volume;			// объЄм €чейки
		vector_t gcenter;		// геометрический центр €чейки
		vector_t center;			// центр масс €чейки
		vector_t dx;				// максимальные размеры €чейки
#if UGRID_CELL_H
		value_type h;				// характерный размер €чейки
#endif
#if UGRID_CELL_TYPE
		unsigned cell_type;			// тип €чейки
#endif
	};

	struct cell_b_value_type {
		__HOST __DEVICE
		cell_b_value_type() : cell((unsigned)-1), interface((unsigned)-1)
#if UGRID_CELL_B_BC_CELL
			, bc_cell((unsigned)-1)
#endif
#if UGRID_CELL_B_PATCH
			, patch_num(-1), patch_type(-1)
#endif
		{}
		unsigned cell;			// номер €чейки на границе
		unsigned interface;		// номер поверхности на границе
#if UGRID_CELL_B_CENTER
		vector_t center;		// центр фиктивной €чейки
#endif
#if UGRID_CELL_B_BC_CELL
		unsigned bc_cell;		// номер €чейки "за границей"
#endif
#if UGRID_CELL_B_PATCH
		// »нтерпретаци€ этих двух полей - дело приложени€.
		int patch_num;			// номер патча
		int patch_type;			// тип патча
#endif
	};

	struct interface_value_type {
		unsigned cells[2];		// €чейки, контактирующие через поверхность
		unsigned cell_loc_num[2]; // локальный номер поверхности в €чейке
		vector_t normal;		// нормаль к поверхности
#if UGRID_INTERFACE_AREA
		value_type area;		// площадь поверхности
#endif
#if UGRID_INTERFACE_CENTER
		vector_t center;		// центр грани
#endif
	};

	void calc_geometry(const mpi::communicator &comm);

	_KIAM_MATH::math_vector<vector_t> m_nodes;	// массив координат узлов
	_KIAM_MATH::math_vector<unsigned> m_nodes_l2g;
	_KIAM_MATH::math_vector<cell_value_type> m_cells;
	_KIAM_MATH::math_vector<unsigned> m_cells_l2g;
	_KIAM_MATH::math_vector<cell_b_value_type> m_cells_b;
	_KIAM_MATH::math_vector<interface_value_type> m_interfaces;
	_KIAM_MATH::math_vector<unsigned> m_cell_node_shift, m_cell_node_index, m_cell_interface_shift, m_cell_interface_index,
		m_interface_node_shift, m_interface_node_index;
	vector_t m_ptMin, m_ptMax;	// ћинимальные и максимальные координаты вершин.
	vector_t m_dxMin;	// ћинимальные размеры €чеек.
#if UGRID_CELL_H
	value_type m_hMin;		// ћинимальный характерный размер €чейки.
#endif
	unsigned m_local_cells, m_global_cells, m_local_nodes, m_global_nodes;
};

_UGRID_MATH_END

#include "unstructured_grid.inl"
#include "unstructured_grid_geometry.inl"
