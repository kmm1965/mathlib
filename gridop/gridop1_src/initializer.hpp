#pragma once

template<class TAG, typename T>
struct initializer : evaluable_object<TAG, initializer<TAG, T> >
{
    typedef T value_type;

    constexpr initializer(size_t size) : m_size(size){}

    constexpr value_type operator[](size_t i) const {
        //x = h*(i+0.5);
        //C[i] = 0.5 + 0.5*std::tanh(beta*(x-L*0.25)*0.5) + 0.5*std::tanh(-beta*(x-L*0.75)*0.5);//(i < N/2)?eps:(1-eps);
       // C[i] = 0.5 + 0.5*std::tanh(beta*(x-L*0.5)*0.5);
        return i > m_size / 2 ? eps : 1 - eps;
    }

private:
    const size_t m_size;
    const value_type eps = 0;
};
