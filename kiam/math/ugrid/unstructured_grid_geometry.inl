#pragma once

#include "../host_vector.hpp"

_UGRID_MATH_BEGIN

template<unsigned DIM, typename T>
typename std::enable_if<DIM == 2>::type get_pt_min(vector2_value<T> &ptMin, const vector2_value<T> &pt)
{
    if (ptMin.value_x() > pt.value_x())
        ptMin.value_x(pt.value_x());
    if (ptMin.value_y() > pt.value_y())
        ptMin.value_y(pt.value_y());
}

template<unsigned DIM, typename T>
typename std::enable_if<DIM == 3>::type get_pt_min(vector_value<T> &ptMin, const vector_value<T> &pt)
{
    if (ptMin.value_x() > pt.value_x())
        ptMin.value_x(pt.value_x());
    if (ptMin.value_y() > pt.value_y())
        ptMin.value_y(pt.value_y());
    if (ptMin.value_z() > pt.value_z())
        ptMin.value_z(pt.value_z());
}

template<unsigned DIM, typename T>
typename std::enable_if<DIM == 2>::type get_pt_max(vector2_value<T> &ptMax, const vector2_value<T> &pt)
{
    if (ptMax.value_x() < pt.value_x())
        ptMax.value_x(pt.value_x());
    if (ptMax.value_y() < pt.value_y())
        ptMax.value_y(pt.value_y());
}

template<unsigned DIM, typename T>
typename std::enable_if<DIM == 3>::type get_pt_max(vector_value<T> &ptMax, const vector_value<T> &pt)
{
    if (ptMax.value_x() < pt.value_x())
        ptMax.value_x(pt.value_x());
    if (ptMax.value_y() < pt.value_y())
        ptMax.value_y(pt.value_y());
    if (ptMax.value_z() < pt.value_z())
        ptMax.value_z(pt.value_z());
}

template<typename T, unsigned DIM>
void unstructured_grid<T, DIM>::calc_geometry(const mpi::communicator &comm)
{
    const int psize = comm.size(), rank = comm.rank();
    // Рассчитаем геометрию ячеек.
    if (rank == 0)
        std::cout << "Calculating cells geometry..." << std::flush;
    boost::timer::cpu_timer t_geometry;
    const HOST_VECTOR_T<vector_t> _REF nodes(m_nodes);
    HOST_VECTOR_T<cell_value_type> _REF cells(m_cells);
    HOST_VECTOR_T<interface_value_type> _REF interfaces(m_interfaces);
    HOST_VECTOR_T<unsigned>
        _REF cell_node_shift(m_cell_node_shift),
        _REF cell_node_index(m_cell_node_index),
        _REF cell_interface_shift(m_cell_interface_shift),
        _REF cell_interface_index(m_cell_interface_index),
        _REF interface_node_shift(m_interface_node_shift),
        _REF interface_node_index(m_interface_node_index);
    // Рассчитаем различные свойства ячеек
    for(unsigned icell = 0; icell < cells.size(); ++icell){
        cell_value_type &cell = cells[icell];
        vector_t nodeMin, nodeMax; // Минимальные и максимальные координаты ячейки.
        const unsigned
            cell_node_index_start = cell_node_shift[icell],
            cell_node_index_end = cell_node_shift[icell + 1],
            cell_node_count = cell_node_index_end - cell_node_index_start;
        for (unsigned i = cell_node_index_start; i < cell_node_index_end; ++i){
            const vector_t &node = nodes[cell_node_index[i]];
            if (i == cell_node_index_start){
                nodeMin = node;
                nodeMax = node;
            } else {
                get_pt_min<dim>(nodeMin, node);
                get_pt_max<dim>(nodeMax, node);
            }
        }
        cell.dx = nodeMax - nodeMin;        // Сохрняем максимальные размеры ячейки
        get_pt_min<dim>(m_dxMin, cell.dx);  // Минимальные размеры по всей сетке
        // Геометрический центр ячейки
        cell.gcenter = std::accumulate(cell_node_index.cbegin() + cell_node_index_start,
            cell_node_index.cbegin() + cell_node_index_end, vector_t(),
            [&nodes](const vector_t &vec, unsigned inode){ return vec + nodes[inode]; }) / value_type(cell_node_count);
        if(cell_node_count == 5) // Для пирамиды особый случай
            ((cell.gcenter *= 5.) += nodes[cell_node_index[cell_node_index_start]]) /= 6.;

        // Центр масс ячейки
        cell.mcenter = cell_mcenter<value_type, dim>(nodes.get_vector_proxy(), cell_node_index.get_vector_proxy(), cell_node_index_start, cell_node_count);
        // Объём ячейки
        cell.volume = cell_volume<value_type, dim>(nodes.get_vector_proxy(), cell_node_index.get_vector_proxy(), cell_node_index_start, cell_node_count);
        assert(cell.volume > 0);
    }
    if(rank == 0)
        std::cout << "done" << std::endl;
    // Рассчитаем геометрию граней.
    if (rank == 0)
        std::cout << "Calculating interfaces geometry..." << std::flush;
    // Теперь рассчитаем свойства граней (интерфейсов)
    for (unsigned iface = 0; iface < interfaces.size(); ++iface){
        interface_value_type &interface = interfaces[iface];
        const unsigned
            interface_node_index_start = interface_node_shift[iface],
            interface_node_index_end = interface_node_shift[iface + 1],
            interface_node_count = interface_node_index_end - interface_node_index_start;
        assert(interface_node_count == dim || dim == 3 && interface_node_count == dim + 1);
        interface.normal = calc_normal<value_type, dim>(nodes.get_vector_proxy(), interface_node_index.get_vector_proxy(), interface_node_index_start);
        // Сдалем эту нормаль внешней для первой ячейки.
        // Если у интерфейса есть вторая ячейка (интерфейс не граничный),
        // то для этой второй ячейки нормаль будет внутренней.
        const unsigned
            icell0 = interface.cells[0],
            cell0_node_index_start = cell_node_shift[icell0],
            cell0_node_index_end = cell_node_shift[icell0 + 1];
        // Найдём вершину ячейки, не принадлежащую интерфейсу.
        unsigned inode = (unsigned)-1;
        for (unsigned i = cell0_node_index_start; i < cell0_node_index_end; ++i){
            const unsigned cell0_node = cell_node_index[i];
            // Вершина ячейки является и вершиной интерфейса, такие вершины нам не интересны.
            if (std::find(interface_node_index.cbegin() + interface_node_index_start, interface_node_index.cbegin() + interface_node_index_end, cell0_node) ==
                interface_node_index.cbegin() + interface_node_index_end)
            { // Вершина не принадлежит грани.
                inode = cell0_node;
                break;
            }
        }
        assert(inode != (unsigned)-1);
        // Скалярное произведение этого вектора на вектор нормали должно быть отрицательным.
        // Если это не так, то меняем знак вектора нормали.
        const vector_t &interface_node0 = nodes[interface_node_index[interface_node_index_start]];
        if ((interface.normal & (nodes[inode] - interface_node0)) > 0)
            interface.normal = -interface.normal;
        interface.normal /= interface.normal.length(); // Делаем нормаль единичной.
#if UGRID_INTERFACE_AREA
        interface.area = calc_interface_area<value_type, dim>(nodes.get_vector_proxy(), interface_node_index.get_vector_proxy(), interface_node_index_start, interface_node_count);
#endif
#if UGRID_INTERFACE_CENTER
        // Геометрический центр грани
        interface.gcenter = std::accumulate(interface_node_index.cbegin() + interface_node_index_start,
            interface_node_index.cbegin() + interface_node_index_end, vector_t(),
            [&nodes](const vector_t& vec, unsigned inode) { return vec + nodes[inode]; }) / value_type(interface_node_count);
        // Вычислим центр масс грани.
        if(interface_node_count == dim){
            // Это простейший случай. В двумерном случае отрезок, в трёхмерном - треугольник.
            interface.mcenter = interface.gcenter;
        } else if(dim == 3){
            assert(interface_node_count == 4);
            interface.mcenter = quadrilateral_mcenter<value_type, dim>(
                nodes[interface_node_index[interface_node_index_start]],
                nodes[interface_node_index[interface_node_index_start + 1]],
                nodes[interface_node_index[interface_node_index_start + 2]],
                nodes[interface_node_index[interface_node_index_start + 3]]);
        } else { // dim == 2
            assert(false); // Этого не может быть
        }
#endif
    }
    if (rank == 0)
        std::cout << "done" << std::endl;
#if UGRID_CELL_H
    // Найдём минимальную высоту каждой ячейки
    if (rank == 0)
        std::cout << "Calculating minimal cell height..." << std::flush;
    value_type hMin = std::numeric_limits<value_type>::max();
    for(unsigned icell = 0; icell < cells.size(); ++icell){
        cell_value_type &cell = cells[icell];
        const unsigned
            cell_interface_index_start = cell_interface_shift[icell],
            cell_interface_index_end = cell_interface_shift[icell + 1],
            cell_interface_count = cell_interface_index_end - cell_interface_index_start;
        if(cell_interface_count == dim + 1){ // Тетраэдр или треугольник
            // Для тетраэдров и треугольников минимальная высота обратна максимальной площади грани.
            value_type max_interface_area = 0;
            for (unsigned i = cell_interface_index_start; i < cell_interface_index_end; ++i){
                const unsigned iface = cell_interface_index[i];
                const interface_value_type &interface = interfaces[iface];
#if UGRID_INTERFACE_AREA
                if (max_interface_area < interface.area)
                    max_interface_area = interface.area;
#else
                const unsigned
                    interface_node_index_start = interface_node_shift[iface],
                    interface_node_count = interface_node_shift[iface + 1] - interface_node_index_start;
                assert(interface_node_count == dim);
                const value_type area = calc_interface_area<dim>(nodes.get_vector_proxy(), interface_node_index.get_vector_proxy(), interface_node_index_start, interface_node_count);
                if (max_interface_area < area)
                    max_interface_area = area;
#endif
            }
            assert(max_interface_area > 0);
            cell.h = cell.volume * dim / max_interface_area;
        } else {
            const unsigned cell_node_index_start = cell_node_shift[icell];
            cell.h = calc_cell_height<value_type, dim>(nodes.get_vector_proxy(), cell_node_index.get_vector_proxy(), cell_node_index_start, cell_node_shift[icell + 1] - cell_node_index_start);
        }
        assert(cell.h > 0);
        if(hMin > cell.h)
            hMin = cell.h;
    }
    mpi::all_reduce(comm, hMin, m_hMin, mpi::minimum<value_type>());
    if (rank == 0)
        std::cout << "done, hMin=" << std::scientific << std::setprecision(3) << m_hMin << std::endl;
#endif // UGRID_CELL_H
    // Рассчитаем граничные (фиктивные) ячейки.
    if (rank == 0)
        std::cout << "Calculating boundary cells..." << std::flush;
    HOST_VECTOR_T<cell_b_value_type> _REF cells_b(m_cells_b);
#if UGRID_CELL_B_PATCH
    assert(cells_b.size() > 0);
    for (unsigned iface = unsigned(interfaces.size() - cells_b.size()), icell_b = 0; iface < (unsigned)interfaces.size(); ++iface, ++icell_b){
#else
    for (unsigned iface = 0; iface < interfaces.size(); ++iface){
#endif
        interface_value_type &interface = interfaces[iface];
#if UGRID_CELL_B_PATCH
        cell_b_value_type &cell_b = cells_b[icell_b];
        assert(cell_b.patch_type != 0);
        if(cell_b.patch_type == 1)  // exchange
            continue;
        assert(interface.cells[1] == (unsigned)-1);
#else
        if (interface.cells[1] != (unsigned)-1) // Внутренняя грань
            continue;
        cell_b_value_type cell_b;
#endif
        cell_b.cell = interface.cells[0];
        cell_b.interface = iface;
#if UGRID_CELL_B_CENTER
        // По умолчанию центр фиктивной ячейки совпадает с центром грани.
        cell_b.gcenter = interface.gcenter;
        cell_b.mcenter = interface.mcenter;
#endif
        interface.cells[1] = unsigned(cells.size() +
#if UGRID_CELL_B_PATCH
            icell_b
#else
            cells_b.size()
#endif
        );
        // Теперь надо ячейке, которой принадлежит эта грань, прописать в качестве соседа "фиктивную" ячейку.
        assert(interface.cell_loc_num[1] == (unsigned)-1);
        interface.cell_loc_num[1] = 0;
#if !UGRID_CELL_B_PATCH
        cells_b.push_back(cell_b);
#endif
    }
    // Проверим, что для внутренних граней есть обе соседние ячейки
    bool has_errors = false;
    for (unsigned iface = 0, iface_end = unsigned(interfaces.size() - cells_b.size()); iface < iface_end; ++iface){
        interface_value_type& interface = interfaces[iface];
        assert(interface.cells[0] < cells.size());
        if (interface.cells[1] >= cells.size()) {
            std::cerr << "Error in the grid: for the inner interface " << iface << " the second inner cell is not defined" << std::endl;
            has_errors = true;
        }
    }
    if (has_errors)
        throw std::runtime_error("Errors were detected!");
#if !defined(__CUDACC__) && !defined(NDEBUG)
    for (unsigned iface = 0; iface < interfaces.size(); ++iface){
        assert(interfaces[iface].cells[1] != (unsigned)-1);
    }
#endif
    size_t total_cells_b_count, total_cells_count;
    mpi::reduce(comm, cells_b.size(), total_cells_b_count, std::plus<size_t>(), 0);
    mpi::reduce(comm, cells.size() + cells_b.size(), total_cells_count, std::plus<size_t>(), 0);
    if (rank == 0) std::cout << "done" << std::endl
        << " Total number of boundary cells=" << total_cells_b_count << std::endl
        << " Total number of cells=" << total_cells_count << std::endl;
    // Рассчитаем минимальные и максимальные координаты вершин.
    m_ptMin = m_ptMax = nodes[0];
    for(unsigned inode = 1; inode < nodes.size(); ++inode){
        const vector_t &node = nodes[inode];
        get_pt_min<dim>(m_ptMin, node);
        get_pt_max<dim>(m_ptMax, node);
    }
#ifdef __CUDACC__
    m_cells = cells;
    m_interfaces = interfaces;
    m_cells_b = cells_b;
#endif
    t_geometry.stop();
    if (rank == 0)
        std::cout << t_geometry.format(2, "Total geometry time=%w seconds") << std::endl;
}

_UGRID_MATH_END
