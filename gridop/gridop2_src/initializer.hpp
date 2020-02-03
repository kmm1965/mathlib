#pragma once

template<class TAG, typename T, typename L, typename B>
struct initializer : evaluable_object<TAG, initializer<TAG, T, L, B> >
{
    typedef T value_type;
    typedef L length_type;
    typedef dim2_index<default_index, default_index> index_type;

    constexpr initializer(size_t size_x, size_t size_y, length_type x0, length_type x1, length_type x2,
        length_type y0, length_type y1, length_type y2, length_type h, B b, length_type R) :
        m_index(default_index(size_x), default_index(size_y)),
        x0(x0), x1(x1), x2(x2), y0(y0), y1(y1), y2(y2), h(h), b(b), R(R){}

    __DEVICE
    value_type operator[](size_t i) const
    {
        const typename index_type::value_type ind = m_index.value(i);
        const length_type
            x = (ind.first + 0.5) * h,
            y = (ind.second + 0.5) * h,
            r0 = func::sqrt(func::sqr(x - x0) + func::sqr(y - y0)),
            r1 = func::sqrt(func::sqr(x - x1) + func::sqr(y - y1)),
            r2 = func::sqrt(func::sqr(x - x2) + func::sqr(y - y2));

        return 0.5 * (1 + func::tanh(b * 0.5 * (R - r0)))
            + 0.5 * (1 + func::tanh(b * 0.5 * (R - r1)))
            + 0.5 * (1 + func::tanh(b * 0.5 * (R - r2)));
    }

private:
    const index_type m_index;
    const length_type x0, x1, x2, y0, y1, y2, h, R;
    const B b;
};
