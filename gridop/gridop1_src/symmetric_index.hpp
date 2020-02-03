#pragma once

struct symmetric_index : index_base<symmetric_index>
{
    typedef isize_t value_type;

    constexpr symmetric_index(size_t size) : m_size(size){}

    constexpr size_t size() const {
        return m_size;
    }

    constexpr size_t operator[](value_type i) const
	{
		const value_type N = (value_type)m_size;
		assert(i > -N && i < 2 * N - 1);
		if (i < 0) i = -i;
		else if (i >= N)
			i = N - (i - N + 2);
		assert(i >= 0 && i < N);
		return i;
	}

    constexpr value_type value(size_t i) const
    {
        assert(i < m_size);
        return (value_type)i;
    }

private:
    const size_t m_size;
};
