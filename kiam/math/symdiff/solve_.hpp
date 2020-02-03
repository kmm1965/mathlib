#pragma once

#include <functional>

#include "square_matrix.hpp"
#include "val_array.hpp"

_SYMDIFF_BEGIN

template<typename T, size_t N>
val_array<T, N> solve(square_matrix<T, N> &&A, val_array<T, N> x)
{
    using namespace std::placeholders;
    const T eps = 1e-20;
    for(size_t i = 0; i < N; ++i){
        T& Aii = A(i, i);
        if(std::abs(Aii) < eps){
            Aii = 0;
            for(size_t i2 = i + 1; i2 < N; ++i2){
                T& Ai2i = A(i2, i);
                if(std::abs(Ai2i) < eps)
                    Ai2i = 0;
                else {
                    std::swap(x[i], x[i2]);
                    std::swap_ranges(&Aii, &Aii + (N - i), &Ai2i);
                    break;
                }
            }
        }
        if(Aii != 0){
            x[i] /= Aii;
            std::transform(&Aii, &Aii + (N - i), &Aii, [&Aii](const T& v) { return v / Aii; }/*std::bind(std::divides<T>(), _1, Aii)*/);
            assert(Aii == 1);
            for(size_t i2 = i + 1; i2 < N; ++i2){
                T& Ai2i = A(i2, i);
                if(std::abs(Ai2i) < eps)
                    Ai2i = 0;
                else {
                    x[i2] /= Ai2i;
                    std::transform(&Ai2i, &Ai2i + (N - i), &Ai2i, [&Ai2i](const T &v) { return v / Ai2i; }/*std::bind2nd(std::divides<T>(), Ai2i)*/);
                    assert(Ai2i == 1);
                    x[i2] -= x[i];
                    std::transform(&Ai2i, &Ai2i + (N - i), &Aii, &Ai2i, std::minus<T>());
                    assert(Ai2i == 0);
                }
            }
        }
    }
    for(int i = N - 1; i >= 0; --i){
        const T &Aii = A(i, i);
        T &xi = x[i];
        assert(Aii == 0 || Aii == 1);
        xi -= std::inner_product(&Aii + 1, &Aii + (N - i), &xi + 1, T());
        if(Aii == 0){
            assert(std::abs(xi) < eps);
            xi = 0;
        }
    }
    return x;
}

template<typename T, size_t N>
val_array<T, N> solveA(square_matrix<T, N> A, val_array<T, N> x){
    return solve(std::move(A), x);
}

_SYMDIFF_END
