#pragma once

#include "../Maybe.hpp"

_FUNCPROG_BEGIN

/*
-- | Decompose a list into its head and tail. If the list is empty,
-- returns 'Nothing'. If the list is non-empty, returns @'Just' (x, xs)@,
-- where @x@ is the head of the list and @xs@ its tail.
--
-- @since 4.8.0.0
uncons                  :: [a] -> Maybe (a, [a])
uncons []               = Nothing
uncons (x:xs)           = Just (x, xs)
*/

#define UNCONS_TYPE_(T) BOOST_IDENTITY_TYPE((pair_t<T, List<T> >))
#define UNCONS_TYPE(T) typename UNCONS_TYPE_(T)

template<typename T>
Maybe<PAIR_T(T, List<T>)> uncons(List<T> const& l) {
	return null(l) ? Nothing<PAIR_T(T, List<T>)>() : Just(std::make_pair(head(l), tail(l)));
}

DEFINE_FUNCTION_2(1, List<T0>, replicate, size_t, n, T0 const&, value,
	return List<T0>(n, value);)

/*
-- | 'takeWhile', applied to a predicate @p@ and a list @xs@, returns the
-- longest prefix (possibly empty) of @xs@ of elements that satisfy @p@:
--
-- > takeWhile (< 3) [1,2,3,4,1,2,3,4] == [1,2]
-- > takeWhile (< 9) [1,2,3] == [1,2,3]
-- > takeWhile (< 0) [1,2,3] == []
--

{-# NOINLINE [1] takeWhile #-}
takeWhile               :: (a -> Bool) -> [a] -> [a]
takeWhile _ []          =  []
takeWhile p (x:xs)
            | p x       =  x : takeWhile p xs
            | otherwise =  []
*/
DEFINE_FUNCTION_2(2, List<fdecay<T0> >, takeWhile, function_t<bool(T1)> const&, p, List<T0> const&, l,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	if (null(l))
		return l;
	else {
		T0 const& x = head(l);
		return p(x) ? x >> takeWhile(p, tail(l)) : List<fdecay<T0> >();
	}
)

/*
-- | 'dropWhile' @p xs@ returns the suffix remaining after 'takeWhile' @p xs@:
--
-- > dropWhile (< 3) [1,2,3,4,5,1,2,3] == [3,4,5,1,2,3]
-- > dropWhile (< 9) [1,2,3] == []
-- > dropWhile (< 0) [1,2,3] == [1,2,3]
--

dropWhile               :: (a -> Bool) -> [a] -> [a]
dropWhile _ []          =  []
dropWhile p xs@(x:xs')
            | p x       =  dropWhile p xs'
            | otherwise =  xs
*/
DEFINE_FUNCTION_2(2, List<fdecay<T0> >, dropWhile, function_t<bool(T1)> const&, p, List<T0> const&, l,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	if (null(l))
		return l;
	else {
		T0 const& x = head(l);
		return p(x) ? dropWhile(p, tail(l)) : l;
	})

// take
DEFINE_FUNCTION_2(1, List<T0>, take, int, n, List<T0> const&, l,
	typename List<T0>::const_iterator it = l.cbegin();
	const typename List<T0>::const_iterator iend = l.cend();
	while (n > 0 && it != iend){
		--n;
		++it;
	}
	return List<T0>(l.cbegin(), it);)

// drop
DEFINE_FUNCTION_2(1, List<T0>, drop, int, n, List<T0> const&, l,
	typename List<T0>::const_iterator it = l.cbegin();
	const typename List<T0>::const_iterator iend = l.cend();
	while (n > 0 && it != iend){
		--n;
		++it;
	}
	return List<T0>(it, iend);)

// splitAt
DEFINE_FUNCTION_2(1, PAIR_T(List<T0>, List<T0 >), splitAt, int, n, List<T0> const&, l,
	typename List<T0>::const_iterator it = l.cbegin();
	const typename List<T0>::const_iterator iend = l.cend();
	while (n > 0 && it != iend){
		--n;
		++it;
	}
	return std::make_pair(List<T0>(l.cbegin(), it), List<T0>(it, iend));)

/*
-- | 'span', applied to a predicate @p@ and a list @xs@, returns a tuple where
-- first element is longest prefix (possibly empty) of @xs@ of elements that
-- satisfy @p@ and second element is the remainder of the list:
--
-- > span (< 3) [1,2,3,4,1,2,3,4] == ([1,2],[3,4,1,2,3,4])
-- > span (< 9) [1,2,3] == ([1,2,3],[])
-- > span (< 0) [1,2,3] == ([],[1,2,3])
--
-- 'span' @p xs@ is equivalent to @('takeWhile' p xs, 'dropWhile' p xs)@

span                    :: (a -> Bool) -> [a] -> ([a],[a])
span _ xs@[]            =  (xs, xs)
span p xs@(x:xs')
         | p x          =  let (ys,zs) = span p xs' in (x:ys,zs)
         | otherwise    =  ([],xs)

*/

template<typename T>
using span_type = PAIR_T(List<fdecay<T> >, List<fdecay<T> >);

DEFINE_FUNCTION_2(1, span_type<T0>, span, function_t<bool(T0)> const&, p, List<fdecay<T0> > const&, l,
	if (null(l))
		return std::make_pair(l, l);
	else {
		T0 const& x = head(l);
		if (p(x)){
			const span_type<T0> yz = span(p, tail(l));
			return std::make_pair(x >> yz.first, yz.second);
		} else return std::make_pair(List<fdecay<T0> >(), l);
	})

/*
-- | 'break', applied to a predicate @p@ and a list @xs@, returns a tuple where
-- first element is longest prefix (possibly empty) of @xs@ of elements that
-- /do not satisfy/ @p@ and second element is the remainder of the list:
--
-- > break (> 3) [1,2,3,4,1,2,3,4] == ([1,2,3],[4,1,2,3,4])
-- > break (< 9) [1,2,3] == ([],[1,2,3])
-- > break (> 9) [1,2,3] == ([1,2,3],[])
--
-- 'break' @p@ is equivalent to @'span' ('not' . p)@.

break                   :: (a -> Bool) -> [a] -> ([a],[a])
#if defined(USE_REPORT_PRELUDE)
break p                 =  span (not . p)
#else
-- HBC version (stolen)
break _ xs@[]           =  (xs, xs)
break p xs@(x:xs')
           | p x        =  ([],xs)
           | otherwise  =  let (ys,zs) = break p xs' in (x:ys,zs)
#endif
*/
DEFINE_FUNCTION_2(1, span_type<T0>, _break, function_t<bool(T0)> const&, p, List<fdecay<T0> > const&, l,
	return span(_([p](T0 const& v){ return !p(v); }), l);)

/*
-- | 'reverse' @xs@ returns the elements of @xs@ in reverse order.
-- @xs@ must be finite.
reverse                 :: [a] -> [a]
#if defined(USE_REPORT_PRELUDE)
reverse                 =  foldl (flip (:)) []
#else
reverse l =  rev l []
  where
    rev []     a = a
    rev (x:xs) a = rev xs (x:a)
#endif
*/
// reverse = foldl (flip (:)) []
template<typename T>
List<T> reverse(List<T> const& l){
	return foldl(_flip(_(cons<T>)), List<T>(), l);
}

/*
-- | 'lookup' @key assocs@ looks up a key in an association list.
lookup                  :: (Eq a) => a -> [(a,b)] -> Maybe b
lookup _key []          =  Nothing
lookup  key ((x,y):xys)
| key == x          =  Just y
| otherwise         =  lookup key xys
*/

template<typename B, typename A>
Maybe<B> lookup(A const& a, List<pair_t<A, B> > const& l)
{
	if (null(l)) return Nothing<B>();
	const pair_t<A, B> &h = head(l);
	return h.first == a ? Just(h.second) : lookup(a, tail(l));
}

template<typename B, typename A>
function_t<Maybe<B>(List<pair_t<A, B> > const&)> lookup(A const& a){
	return [a](List<pair_t<A, B> > const& l){
		return lookup(a, l);
	};
}

template<typename B, typename A>
function_t<Maybe<B>(A const&, List<pair_t<A, B> > const&)> lookup(){
	return [](A const& a, List<pair_t<A, B> > const& l){
		return lookup(a, l);
	};
}

/*
-- | Map a function over a list and concatenate the results.
concatMap               :: (a -> [b]) -> [a] -> [b]
concatMap f             =  foldr ((++) . f) []
*/
DEFINE_FUNCTION_2(2, List<T1>, concatMap, function_t<List<T1>(T0 const&)> const&, f, List<T0> const&, l,
	return foldr(_(concat2<T1>) & f, List<T1>(), l);)

template<typename T>
List<T> sort(List<T> const& l)
{
	std::vector<T> v(l.cbegin(), l.cend());
	std::sort(v.begin(), v.end());
	return List<T>(v.cbegin(), v.cend());
}

// nub :: (Eq a) => [a] -> [a]
// nub =  nubBy (==)
template<typename T>
List<T> nub(List<T> const& l) {
	return nubBy(_(eq<T>), l);
}

// The concatenation of all the elements of a container of lists.
//concat::Foldable t = > t[a] ->[a]
//concat xs = build(\c n->foldr(\x y->foldr c y x) n xs)
template<typename T>
List<T> concat(List<List<T> > const& l) {
	return foldr(_(concat2<T>), List<T>(), l);
}

inline String strconcat(List<String> const& l){
    return foldr(_(concat2<char>), String(), l);
}

// nubBy           :: (a -> a -> Bool) -> [a] -> [a]
// nubBy eq []     =  []
// nubBy eq (x:xs) =  x : nubBy eq (filter (\ y -> not (eq x y)) xs)
DEFINE_FUNCTION_2(3, List<T0>, nubBy, function_t<bool(T1, T2)> const&, pred, List<T0> const&, l,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	static_assert(is_same_as<T0, T2>::value, "Should be the same");
	if (null(l))
        return l;
    T0 const& x = head(l);
    return x >> nubBy(pred, filter(_([&pred, &x](T0 const& y) { return !pred(x, y); }), tail(l)));)

// delete :: (Eq a) => a -> [a] -> [a]
// delete =  deleteBy (==)
DEFINE_FUNCTION_2(1, List<T0>, _delete, T0 const&, x, List<T0> const&, l,
    return deleteBy(_(eq<T0>), x, l);)

// deleteBy             :: (a -> a -> Bool) -> a -> [a] -> [a]
// deleteBy _  _ []     = []
// deleteBy eq x (y:ys) = if x `eq` y then ys else y : deleteBy eq x ys
DEFINE_FUNCTION_3(3, List<T0>, deleteBy, function_t<bool(T1, T2)> const&, pred, T0 const&, x, List<T0> const&, l,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	static_assert(is_same_as<T0, T2>::value, "Should be the same");
	if (null(l))
        return l;
    T0 const& y = head(l);
    const List<T0> ys = tail(l);
    return pred(x, y) ? ys : y >> deleteBy(pred, x, ys);)

// union                   :: (Eq a) => [a] -> [a] -> [a]
// union                   = unionBy (==)
DEFINE_FUNCTION_2(1, List<T0>, _union, List<T0> const&, xs, List<T0> const&, ys,
    return unionBy(_(eq<T0>), xs, ys);)

// unionBy                 :: (a -> a -> Bool) -> [a] -> [a] -> [a]
// unionBy eq xs ys        =  xs ++ foldl (flip (deleteBy eq)) (nubBy eq ys) xs
DEFINE_FUNCTION_3(3, List<T0>, unionBy, function_t<bool(T1, T2)> const&, pred, List<T0> const&, xs, List<T0> const&, ys,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	static_assert(is_same_as<T0, T2>::value, "Should be the same");
	return xs + foldl(_flip(_deleteBy<T0>(pred)), nubBy(pred, ys), xs);)

// intersect               :: (Eq a) => [a] -> [a] -> [a]
// intersect               =  intersectBy (==)
DEFINE_FUNCTION_2(1, List<T0>, intersect, List<T0> const&, xs, List<T0> const&, ys,
    return intersectBy(_(eq<T0>), xs, ys);)

// intersectBy             :: (a -> a -> Bool) -> [a] -> [a] -> [a]
// intersectBy _  [] _     =  []
// intersectBy _  _  []    =  []
// intersectBy eq xs ys    =  [x | x <- xs, any (eq x) ys]
DEFINE_FUNCTION_3(3, List<T0>, intersectBy, function_t<bool(T1, T2)> const&, pred, List<T0> const&, xs, List<T0> const&, ys,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	static_assert(is_same_as<T0, T2>::value, "Should be the same");
	if(null(xs) || null(ys))
        return List<T0>();
    List<T0> result;
    std::copy_if(xs.cbegin(), xs.cend(), std::back_inserter(result),
        [&pred, &ys](T0 const& x) { return any(pred << x, ys); });
    return result;)

// prependToAll            :: a -> [a] -> [a]
// prependToAll _   []     = []
// prependToAll sep (x:xs) = sep : x : prependToAll sep xs
DEFINE_FUNCTION_2(1, List<T0>, prependToAll, T0 const&, sep, List<T0> const&, l,
    return null(l) ? l : sep >> (head(l) >> prependToAll(sep, tail(l)));)

// intersperse             :: a -> [a] -> [a]
// intersperse _   []      = []
// intersperse sep (x:xs)  = x : prependToAll sep xs
DEFINE_FUNCTION_2(1, List<T0>, intersperse, T0 const&, sep, List<T0> const&, l,
    return null(l) ? l : head(l) >> prependToAll(sep, tail(l));)

// intercalate :: [a] -> [[a]] -> [a]
// intercalate xs xss = concat (intersperse xs xss)
DEFINE_FUNCTION_2(1, List<T0>, intercalate, List<T0> const&, xs, List<List<T0> > const&, xss,
    return concat(intersperse(xs, xss));)

// transpose               :: [[a]] -> [[a]]
// transpose []             = []
// transpose ([]   : xss)   = transpose xss
// transpose ((x:xs) : xss) = (x : [h | (h:_) <- xss]) : transpose (xs : [ t | (_:t) <- xss])
template<typename T>
List<List<T> > transpose(List<List<T> > const& l)
{
	if (null(l))
		return l;
	List<T> const& hl = head(l);
	const List<List<T> > xss = tail(l);
	return null(hl) ? transpose(xss) :
		foldl(_([](List<T> const& z, List<T> const& l_) { return z << head(l_); }), List<T>({ head(hl) }), xss) >>
		transpose(foldl(_([](List<List<T> > const& z, List<T> const& l_) {
			return z << tail(l_);
		}), List<List<T> >({ tail(hl) }), xss));
}

// partition      :: (a -> Bool) -> [a] -> ([a],[a])
// partition p xs = foldr (select p) ([],[]) xs
DEFINE_FUNCTION_2(2, PAIR_T(List<T0>, List<T0>), partition, function_t<bool(T1)> const&, p, List<T0> const&, xs,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	return foldr(_select<T0>(p), PAIR_T(List<T0>, List<T0>)(), xs);)

// select :: (a -> Bool) -> a -> ([a], [a]) -> ([a], [a])
// select p x ~(ts,fs) | p x       = (x:ts,fs)
//                     | otherwise = (ts, x:fs)
DEFINE_FUNCTION_3(2, PAIR_T(List<T0>, List<T0>), select, function_t<bool(T1)> const&, p, T0 const&, x, const PAIR_T(List<T0>, List<T0>)&, s,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	return p(x) ? std::make_pair(x >> s.first, s.second) : std::make_pair(s.first, x >> s.second);)

// insert :: Ord a => a -> [a] -> [a]
// insert e ls = insertBy (compare) e ls
DEFINE_FUNCTION_2(1, List<T0>, insert, T0 const&, e, List<T0> const&, ls,
    return insertBy(_(compare<T0>), e, ls);)

/*
insertBy :: (a -> a -> Ordering) -> a -> [a] -> [a]
insertBy _   x [] = [x]
insertBy cmp x ys@(y:ys')
 = case cmp x y of
     GT -> y : insertBy cmp x ys'
     _  -> x : ys
*/
DEFINE_FUNCTION_3(3, List<T0>, insertBy, function_t<Ordering(T1, T2)> const&, cmp, T0 const&, x, List<T0> const&, ls,
	static_assert(is_same_as<T0, T1>::value, "Should be the same");
	static_assert(is_same_as<T0, T2>::value, "Should be the same");
	if (null(ls))
        return List<T0>({ x });
    T0 const& y = head(ls);
    return cmp(x, y) == GT ? y >> insertBy(cmp, x, tail(ls)) : x >> ls;)

_FUNCPROG_END
