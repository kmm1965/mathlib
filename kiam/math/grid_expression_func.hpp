#pragma once

#include "grid_expression.hpp"
#include "executor.hpp"

#include "funcprog2/Functor.hpp"
#include "funcprog2/Applicative.hpp"
#include "funcprog2/Monad.hpp"

using namespace _FUNCPROG2;

_KIAM_MATH_BEGIN

// Functor
template<class GEXP, typename FuncImpl, typename Ret, typename Arg0, typename... Args>
struct fmap_grid_expression : grid_expression<typename GEXP::tag_type, fmap_grid_expression<GEXP, FuncImpl, Ret, Arg0, Args...> >
{
    static_assert(funcprog2::is_same_as<funcprog2::value_type_t<GEXP>, Arg0>::value, "Should be the same");

    using function_type = function2<Ret(Arg0, Args...), FuncImpl>;
    using gexp_type = GRID_EXPR(GEXP);

    fmap_grid_expression(function_type const& f, gexp_type const& gexp) : f(f), gexp_proxy(gexp.get_proxy()){}

    __DEVICE CONSTEXPR auto operator[](size_t i) const {
        return gexp_proxy.push_args(f, i);
    }

private:
    function_type const f;
    typename GEXP::proxy_type const gexp_proxy;
};

namespace funcprog2 {

    template<typename TAG>
    struct _is_functor<_KIAM_MATH::_grid_expression<TAG> > : std::true_type {};

    template<typename TAG>
    struct Functor<_KIAM_MATH::_grid_expression<TAG> >
    {
        // <$> fmap :: Functor f => (a -> b) -> f a -> f b
        template<typename GEXP, typename FuncImpl, typename Ret, typename Arg0, typename... Args>
        static constexpr auto fmap(function2<Ret(Arg0, Args...), FuncImpl> const& f, GRID_EXPR(GEXP) const& gexp){
            return _KIAM_MATH::fmap_grid_expression<GEXP, FuncImpl, Ret, Arg0, Args...>(f, gexp);
        }
    };

    template<typename GEXP, typename FuncImpl, typename Ret, typename Arg0, typename... Args>
    constexpr auto fmap(function2<Ret(Arg0, Args...), FuncImpl> const& f, GRID_EXPR(GEXP) const& gexp){
        return Functor<_grid_expression<typename GEXP::tag_type> >::fmap(f, gexp);
    }

} // namespace funcprog2

// Applicative
template<typename TAG, typename T>
struct pure_grid_expression : grid_expression<TAG, pure_grid_expression<TAG, T> >
{
    pure_grid_expression(T const& value) : value(value){}

    __DEVICE constexpr T operator[](size_t) const {
        return value;
    }

    template<typename F>
    __DEVICE __HOST auto push_args(F const& f, size_t /*i*/) const {
        return _FUNCPROG2::operator<< <first_argument_type_t<F> >(f, value);
    }

private:
    T const value;
};

template<class func_type, unsigned nargs>
struct remove_args {
    using type = typename remove_args<remove_first_arg_t<func_type>, nargs - 1>::type;
};

template<class func_type>
struct remove_args<func_type, 0> {
    using type = func_type;
};

template<class func_type, unsigned nargs>
using remove_args_t = typename remove_args<func_type, nargs>::type;

template<class GEXP_F, class GEXP>
struct apply_grid_expression : grid_expression<typename GEXP::tag_type, apply_grid_expression<GEXP_F, GEXP> >
{
    using function_type = remove_args_t<typename GEXP_F::function_type, GEXP::num_args>;
    using gexp_type = GRID_EXPR(GEXP);

    apply_grid_expression(GRID_EXPR(GEXP_F) const& gexp_f, gexp_type const& gexp) :
        gexp_f_proxy(gexp_f.get_proxy()), gexp_proxy(const_cast<gexp_type&>(gexp).get_proxy()){}

    __DEVICE CONSTEXPR auto operator[](size_t i) const {
        return gexp_proxy.push_args(gexp_f_proxy[i], i);
    }

private:
    typename GEXP_F::proxy_type const gexp_f_proxy;
    typename GEXP::proxy_type const gexp_proxy;
};

namespace funcprog2 {

    template<typename TAG>
    struct _is_applicative<_KIAM_MATH::_grid_expression<TAG> > : std::true_type {};

    template<typename TAG, class GEXP, class _Proxy>
    struct is_applicative<_KIAM_MATH::grid_expression<TAG, GEXP, _Proxy> > : std::true_type {};

    template<typename TAG>
    struct Applicative<_KIAM_MATH::_grid_expression<TAG> > : Functor<_KIAM_MATH::_grid_expression<TAG> >
    {
        using super = Functor<_KIAM_MATH::_grid_expression<TAG> >;

        template<typename T>
        static constexpr auto pure(T const& x){
            return pure_grid_expression<TAG, T>(x);
        }

        template<class GEXP_F, class GEXP>
        static constexpr auto apply(GRID_EXPR(GEXP_F) const& gexp_f, GRID_EXPR(GEXP) const& gexp){
            return _KIAM_MATH::apply_grid_expression<GEXP_F, GEXP>(gexp_f, gexp);
        }
    };

    template<typename TAG, typename T>
    constexpr auto pure(T const& x){
        return Applicative<_grid_expression<TAG> >::pure(x);
    }

    template<class GEXP_F, class GEXP>
    constexpr auto apply(GRID_EXPR(GEXP_F) const& gexp_f, GRID_EXPR(GEXP) const& gexp){
        return Applicative<_grid_expression<typename GEXP::tag_type> >::apply(gexp_f, gexp);
    }

    template<class GEXP_F, class GEXP>
    constexpr auto operator*(GRID_EXPR(GEXP_F) const& gexp_f, GRID_EXPR(GEXP) const& gexp){
        return apply(gexp_f, gexp);
    }
} // namespace funcprog2

template<typename T>
constexpr auto p(T const& x){
    return Applicative<_grid_expression>::pure(x);
}

// Monad
template<class GEXP1, class GEXP2>
struct mbind0_grid_expression : grid_expression<mbind0_grid_expression<GEXP1, GEXP2> >
{
    using grid_expr1_type = GRID_EXPR(GEXP1);
    using grid_expr2_type = GRID_EXPR(GEXP2);

    mbind0_grid_expression(grid_expr1_type const& grid_expr1, grid_expr2_type const& grid_expr2) :
        grid_expr1_proxy(grid_expr1.get_proxy()), grid_expr2_proxy(grid_expr2.get_proxy()){}

    __DEVICE __HOST constexpr auto operator[](size_t /*i*/) const {
        return _(*this);
    }

    __DEVICE __HOST constexpr auto operator()(size_t i) const {
        grid_expr1_proxy[i](i);
        grid_expr2_proxy[i](i);
    }

private:
    typename GEXP1::proxy_type const grid_expr1_proxy;
    typename GEXP2::proxy_type const grid_expr2_proxy;
};

namespace funcprog2 {

    template<typename TAG>
    struct _is_monad<_KIAM_MATH::_grid_expression<TAG> > : std::true_type {};

    template<typename TAG, class GEXP, class _Proxy>
    struct is_monad<_KIAM_MATH::grid_expression<TAG, GEXP, _Proxy> > : std::true_type {};

    template<typename TAG>
    struct Monad<_KIAM_MATH::_grid_expression<TAG> > :
        Applicative<_KIAM_MATH::_grid_expression<TAG> >,
        _Monad<_KIAM_MATH::_grid_expression<TAG> >
    {
        using super = Applicative<_KIAM_MATH::_grid_expression<TAG> >;

        template<typename T>
        static constexpr auto mreturn(T const& x){
            return super::pure(x);
        }

        template<class GEXP, class GEXP_F>
        static constexpr auto mbind(GRID_EXPR(GEXP) const& grid_expr, GRID_EXPR(GEXP_F) const& grid_f){
            return grid_f * grid_expr;
        }

        template<class GEXP1, class GEXP2>
        static constexpr auto mbind0(GRID_EXPR(GEXP1) const& grid_expr1, GRID_EXPR(GEXP2) const& grid_expr2){
            return mbind0_grid_expression<GEXP1, GEXP2>(grid_expr1, grid_expr2);
        }
    };

    template<class GEXP, class GEXP_F>
    inline constexpr auto operator>>=(GRID_EXPR(GEXP) const& grid_expr, GRID_EXPR(GEXP_F) const& grid_f){
        return Monad<_grid_expression>::mbind(grid_expr, grid_f);
    }

    template<class GEXP1, class GEXP2>
    inline constexpr auto operator>>(GRID_EXPR(GEXP1) const& grid_expr1, GRID_EXPR(GEXP2) const& grid_expr2){
        return Monad<_grid_expression>::mbind0(grid_expr1, grid_expr2);
    }

} // namespace funcprog2

template<class GEXP>
void parallel_for(size_t size, GRID_EXPR(GEXP) const& gexp) {
    typename GEXP::proxy_type const gexp_proxy = gexp.get_proxy();
    default_executor<void>()(size, [gexp_proxy] __DEVICE (size_t i){ gexp_proxy[i](i); });
}

template<typename TAG, class T>
template<class GEXP>
void _KIAM_MATH::vector_grid_function<TAG, T>::operator&=(GRID_EXPR(GEXP) const& gexp){
    parallel_for(vector_type::size(), gexp * (*this));
}

_KIAM_MATH_END
