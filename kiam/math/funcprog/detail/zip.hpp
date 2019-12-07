#pragma once

_FUNCPROG_BEGIN

/*
foldr2 :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
foldr2 k z = go
  where
        go []    _ys     = z
        go _xs   []      = z
        go (x:xs) (y:ys) = k x y (go xs ys)
{-# INLINE [0] foldr2 #-}
*/
template<typename A, typename B, typename C>
C foldr2_(
	function_t<C(A, B, C)> const& k,
	C const& z,
	List<fdecay<A> > const& xs,
	List<fdecay<B> > const& ys)
{
	const function_t<C(List<fdecay<A> > const&, List<fdecay<B> > const&)> go =
		[&k, &z, &go](List<fdecay<A> > const& xs, List<fdecay<B> > const& ys)
	{
		return null(xs) || null(ys) ? z : k(head(xs), head(ys), go(tail(xs), tail(ys)));
	};
	return go(xs, ys);
}

template<typename A, typename B, typename C>
C foldr2_(
	function_t<C(A, B, const C&)> const& k,
	C const& z,
	List<fdecay<A> > const& xs,
	List<fdecay<B> > const& ys)
{
	const function_t<C(List<fdecay<A> > const&, List<fdecay<B> > const&)> go =
		[&k, &z, &go](List<fdecay<A> > const& xs, List<fdecay<B> > const& ys)
	{
		return null(xs) || null(ys) ? z : k(head(xs), head(ys), go(tail(xs), tail(ys)));
	};
	return go(xs, ys);
}

/*
foldr2_left :: (a -> b -> c -> d) -> d -> a -> ([b] -> c) -> [b] -> d
foldr2_left _k  z _x _r []     = z
foldr2_left  k _z  x  r (y:ys) = k x y (r ys)
*/
DEFINE_FUNCTION_4(4, function_t<T3(List<fdecay<T1> > const&)>, foldr2_left, function_t<T3(T0, T1, T2)> const&, k,
	T3 const&, z, fdecay<T0> const&, x, function_t<fdecay<T2>(List<fdecay<T1> > const&)> const&, r,
	return [=](List<fdecay<T1> > const& ys){
		return null(ys) ? z : k(x, head(ys), r(tail(ys)));
	};)

// foldr2 :: (a->b->c->c)->c ->[a] ->[b]->c
// foldr2 k z xs ys = foldr (foldr2_left k z) (\_ -> z) xs ys
DEFINE_FUNCTION_4(4, T2, foldr2, function_t<T2(T0, T1, T3)> const&, k, T2 const&, z,
	List<fdecay<T0> > const&, xs, List<fdecay<T1> > const&, ys,
	static_assert(is_same_as<T2, T3>::value, "Should be the same");
	return foldr(_foldr2_left(k, z), _([&z](List<fdecay<T0> > const&){ return z; }), xs)(ys);)

/*
----------------------------------------------
-- | 'zip' takes two lists and returns a list of corresponding pairs.
--
-- > zip [1, 2] ['a', 'b'] = [(1, 'a'), (2, 'b')]
--
-- If one input list is short, excess elements of the longer list are
-- discarded:
--
-- > zip [1] ['a', 'b'] = [(1, 'a')]
-- > zip [1, 2] ['a'] = [(1, 'a')]
--
-- 'zip' is right-lazy:
--
-- > zip [] _|_ = []
-- > zip _|_ [] = _|_
*/
// zip
DEFINE_FUNCTION_2(2, List<PAIR_T(T0, T1)>, zip, List<T0> const&, l1, List<T1> const&, l2,
	List<PAIR_T(T0, T1)> result;
    typename List<T0>::const_iterator it1;
	typename List<T1>::const_iterator it2;
    for (it1 = l1.cbegin(), it2 = l2.cbegin(); it1 != l1.cend() && it2 != l2.cend(); ++it1, ++it2)
        result.push_back(std::make_pair(*it1, *it2));
    return result;)

// zip3
DEFINE_FUNCTION_3(3, List<TUPLE3(T0, T1, T2)>, zip3, List<T0> const&, l1, List<T1> const&, l2, List<T2> const&, l3,
	List<TUPLE3(T0, T1, T2)> result;
	typename List<T0>::const_iterator it1;
	typename List<T1>::const_iterator it2;
	typename List<T2>::const_iterator it3;
    for (it1 = l1.cbegin(), it2 = l2.cbegin(), it3 = l3.cbegin(); it1 != l1.cend() && it2 != l2.cend() && it3 != l3.cend(); ++it1, ++it2, ++it3)
        result.push_back(std::make_tuple(*it1, *it2, *it3));
    return result;)

// zipWith
DEFINE_FUNCTION_3(3, List<T0>, zipWith, function_t<T0(T1, T2)> const&, f, List<fdecay<T1> > const&, l1, List<fdecay<T2> > const&, l2,
	List<T0> result;
	typename List<fdecay<T1> >::const_iterator it1;
	typename List<fdecay<T2> >::const_iterator it2;
    for (it1 = l1.cbegin(), it2 = l2.cbegin(); it1 != l1.cend() && it2 != l2.cend(); ++it1, ++it2)
        result.push_back(f(*it1, *it2));
    return result;)

// zipWith3
DEFINE_FUNCTION_4(4, List<T0>, zipWith3, function_t<T0(T1, T2, T3)> const&, f, List<fdecay<T1> > const&, l1,
	List<fdecay<T2> > const&, l2, List<fdecay<T3> > const&, l3,
	List<T0> result;
	typename List<T1>::const_iterator it1;
	typename List<T2>::const_iterator it2;
	typename List<T3>::const_iterator it3;
    for (it1 = l1.cbegin(), it2 = l2.cbegin(), it3 = l3.cbegin(); it1 != l1.cend() && it2 != l2.cend() && it3 != l3.cend(); ++it1, ++it2, ++it3)
        result.push_back(f(*it1, *it2, *it3));
	return result;)

// unzip
template<typename T0, typename T1>
pair_t<List<T0>, List<T1> > unzip(List<pair_t<T0, T1> > const& l)
{
	pair_t<List<T0>, List<T1> > result;
    std::for_each(l.cbegin(), l.cend(),
        [&result](pair_t<T0, T1> const& p)
        {
            result.first.push_back(p.first);
            result.second.push_back(p.second);
        });
    return result;
}

template<typename T0, typename T1>
function_t<pair_t<List<T0>, List<T1> >(List<pair_t<T0, T1> > const&)> unzip(){
	return [](List<pair_t<T0, T1> > const& l){
		return unzip(l);
	};
}

// unzip3
template<typename T0, typename T1, typename T2>
tuple3_t<List<T0>, List<T1>, List<T2> > unzip3(List<tuple3_t<T0, T1, T2> > const& l)
{
	tuple3_t<List<T0>, List<T1>, List<T2> > result;
	std::for_each(l.cbegin(), l.cend(),
		[&result](tuple3_t<T0, T1, T2> const& t)
		{
			std::get<0>(result).push_back(std::get<0>(t));
			std::get<1>(result).push_back(std::get<1>(t));
			std::get<2>(result).push_back(std::get<2>(t));
		});
	return result;
}

template<typename T0, typename T1, typename T2>
function_t<tuple3_t<List<T0>, List<T1>, List<T2> >(List<tuple3_t<T0, T1, T2> > const&)> unzip3(){
	return [](List<tuple3_t<T0, T1, T2> > const& l){
		return unzip3(l);
	};
}

_FUNCPROG_END
