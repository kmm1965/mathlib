#pragma once

namespace hpc {

class process_grid
{
public:
	process_grid();
	process_grid(size_t my_node, size_t size_x, size_t size_y = 1, size_t size_z = 1);

	void resize(size_t my_node, size_t size_x, size_t size_y = 1, size_t size_z = 1);

	size_t my_node() const { return m_my_node; }
	size_t nodes() const { return m_size_x * m_size_y * m_size_z; }

	size_t size_x() const { return m_size_x; }
	size_t size_y() const { return m_size_y; }
	size_t size_z() const { return m_size_z; }

	size_t get_x() const { return m_x; }
	size_t get_y() const { return m_y; }
	size_t get_z() const { return m_z; }

	size_t get_node(size_t x, size_t y, size_t z) const;

private:
	size_t m_my_node;
	size_t m_size_x, m_size_y, m_size_z;
	size_t m_x, m_y, m_z;
};

}	// namespace hpc
