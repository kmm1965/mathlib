#pragma once

#include "../funcprog2_common.hpp"

_FUNCPROG2_BEGIN

template<typename T>
constexpr List<T> operator+(List<T> const& l1, List<T> const& l2);

template<typename T> constexpr T const& head(List<T> const&);
template<typename T> constexpr List<T> tail(List<T> const&);
template<typename T> constexpr T const& last(List<T> const&);
template<typename T> constexpr List<T> init(List<T> const&);
template<typename T> constexpr List<T> sort(List<T> const& l);

// The concatenation of all the elements of a container of lists.
//concat::Foldable t = > t[a] ->[a]
template<typename T>
constexpr List<T> concat(List<List<T> > const& l);

inline String strconcat(List<String> const&);

DECLARE_FUNCTION_2(2, List<fdecay<T0> >, filter, FUNCTION2(bool(T0), T1) const&, List<fdecay<T0> > const&)
DECLARE_FUNCTION_2(1, List<T0>, cons, T0 const&, List<T0> const&)
DECLARE_FUNCTION_2(1, List<T0>, concat2, List<T0> const&, List<T0> const&)

/*
-- | /O(n^2)/. The 'nub' function removes duplicate elements from a list.
-- In particular, it keeps only the first occurrence of each element.
-- (The name 'nub' means \`essence\'.)
-- It is a special case of 'nubBy', which allows the programmer to supply
-- their own equality test.
--
-- >>> nub [1,2,3,4,3,2,1,2,4,3,5]
-- [1,2,3,4,5]
*/
template<typename T> constexpr List<T> nub(List<T> const&);

/*
-- | The 'nubBy' function behaves just like 'nub', except it uses a
-- user-supplied equality predicate instead of the overloaded '=='
-- function.
--
-- >>> nubBy (\x y -> mod x 3 == mod y 3) [1,2,4,5,6]
-- [1,2,6]
*/
// nubBy           :: (a -> a -> Bool) -> [a] -> [a]
DECLARE_FUNCTION_2(4, List<T0>, nubBy, FUNCTION2(bool(T1, T2), T3) const&, List<T0> const&);

/*
-- | 'delete' @x@ removes the first occurrence of @x@ from its list argument.
-- For example,
--
-- >>> delete 'a' "banana"
-- "bnana"
--
-- It is a special case of 'deleteBy', which allows the programmer to
-- supply their own equality test.
*/
// delete :: (Eq a) => a -> [a] -> [a]
// delete =  deleteBy (==)
DECLARE_FUNCTION_2(1, List<T0>, delete_, T0 const&, List<T0> const&);

/*
-- | The 'deleteBy' function behaves like 'delete', but takes a
-- user-supplied equality predicate.
--
-- >>> deleteBy (<=) 4 [1..10]
-- [1,2,3,5,6,7,8,9,10]
deleteBy                :: (a -> a -> Bool) -> a -> [a] -> [a]
deleteBy _  _ []        = []
deleteBy eq x (y:ys)    = if x `eq` y then ys else y : deleteBy eq x ys
*/
/*
-- | The 'deleteBy' function behaves like 'delete', but takes a
-- user-supplied equality predicate.
--
-- >>> deleteBy (<=) 4 [1..10]
-- [1,2,3,5,6,7,8,9,10]
*/
// deleteBy             :: (a -> a -> Bool) -> a -> [a] -> [a]
DECLARE_FUNCTION_3(4, List<T0>, deleteBy, FUNCTION2(bool(T1, T2), T3) const&, T0 const&, List<T0> const&);

/*
-- | The 'union' function returns the list union of the two lists.
-- For example,
--
-- >>> "dog" `union` "cow"
-- "dogcw"
--
-- Duplicates, and elements of the first list, are removed from the
-- the second list, but if the first list contains duplicates, so will
-- the result.
-- It is a special case of 'unionBy', which allows the programmer to supply
-- their own equality test.

*/
// union                   :: (Eq a) => [a] -> [a] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, union_, List<T0> const&, List<T0> const&);

// The 'unionBy' function is the non-overloaded version of 'union'.
// unionBy                 :: (a -> a -> Bool) -> [a] -> [a] -> [a]
// unionBy eq xs ys        =  xs ++ foldl (flip (deleteBy eq)) (nubBy eq ys) xs
DECLARE_FUNCTION_3(4, List<T0>, unionBy, FUNCTION2(bool(T1, T2), T3) const&, List<T0> const&, List<T0> const&);

/*
-- | The 'intersect' function takes the list intersection of two lists.
-- For example,
--
-- >>> [1,2,3,4] `intersect` [2,4,6,8]
-- [2,4]
--
-- If the first list contains duplicates, so will the result.
--
-- >>> [1,2,2,3,4] `intersect` [6,4,4,2]
-- [2,2,4]
--
-- It is a special case of 'intersectBy', which allows the programmer to
-- supply their own equality test. If the element is found in both the first
-- and the second list, the element from the first list will be used.
*/
// intersect               :: (Eq a) => [a] -> [a] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, intersect, List<T0> const&, List<T0> const&);

// The 'intersectBy' function is the non-overloaded version of 'intersect'.
// intersectBy             :: (a -> a -> Bool) -> [a] -> [a] -> [a]
DECLARE_FUNCTION_3(4, List<T0>, intersectBy, FUNCTION2(bool(T1, T2), T3) const&, List<T0> const&, List<T0> const&);

/*
-- Not exported:
-- We want to make every element in the 'intersperse'd list available
-- as soon as possible to avoid space leaks. Experiments suggested that
-- a separate top-level helper is more efficient than a local worker.
*/
// prependToAll            :: a -> [a] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, prependToAll, T0 const&, List<T0> const&);

/*
-- | The 'intersperse' function takes an element and a list and
-- \`intersperses\' that element between the elements of the list.
-- For example,
--
-- >>> intersperse ',' "abcde"
-- "a,b,c,d,e"
*/
// intersperse             :: a -> [a] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, intersperse, T0 const&, List<T0> const&);

/*
-- | 'intercalate' @xs xss@ is equivalent to @('concat' ('intersperse' xs xss))@.
-- It inserts the list @xs@ in between the lists in @xss@ and concatenates the
-- result.
--
-- >>> intercalate ", " ["Lorem", "ipsum", "dolor"]
-- "Lorem, ipsum, dolor"
*/
// intercalate :: [a] -> [[a]] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, intercalate, List<T0> const&, List<List<T0> > const&);

/*
-- | The 'transpose' function transposes the rows and columns of its argument.
-- For example,
--
-- >>> transpose [[1,2,3],[4,5,6]]
-- [[1,4],[2,5],[3,6]]
--
-- If some of the rows are shorter than the following rows, their elements are skipped:
--
-- >>> transpose [[10,11],[20],[],[30,31,32]]
-- [[10,20,30],[11,31],[32]]
*/
template<typename T> constexpr List<List<T> > transpose(List<List<T> > const&);

/*
-- | The 'partition' function takes a predicate a list and returns
-- the pair of lists of elements which do and do not satisfy the
-- predicate, respectively; i.e.,
--
-- > partition p xs == (filter p xs, filter (not . p) xs)
--
-- >>> partition (`elem` "aeiou") "Hello World!"
-- ("eoo","Hll Wrld!")
*/
// partition      :: (a -> Bool) -> [a] -> ([a],[a])
DECLARE_FUNCTION_2(3, PAIR_T(List<T0>, List<T0>), partition, FUNCTION2(bool(T1), T2) const&, List<T0> const&);

// select :: (a -> Bool) -> a -> ([a], [a]) -> ([a], [a])
DECLARE_FUNCTION_3(3, PAIR_T(List<T0>, List<T0>), select, FUNCTION2(bool(T1), T2) const&, T0 const&, const PAIR_T(List<T0>, List<T0>)&);

/*
-- | The 'insert' function takes an element and a list and inserts the
-- element into the list at the first position where it is less
-- than or equal to the next element.  In particular, if the list
-- is sorted before the call, the result will also be sorted.
-- It is a special case of 'insertBy', which allows the programmer to
-- supply their own comparison function.
--
-- >>> insert 4 [1,2,3,5,6,7]
-- [1,2,3,4,5,6,7]
*/
// insert :: Ord a => a -> [a] -> [a]
DECLARE_FUNCTION_2(1, List<T0>, insert, T0 const&, List<T0> const&);

// The non-overloaded version of 'insert'.
// insertBy :: (a -> a -> Ordering) -> a -> [a] -> [a]
DECLARE_FUNCTION_3(4, List<T0>, insertBy, FUNCTION2(Ordering(T1, T2), T3) const&, T0 const&, List<T0> const&);

/*
-- | Decompose a list into its head and tail. If the list is empty,
-- returns 'Nothing'. If the list is non-empty, returns @'Just' (x, xs)@,
-- where @x@ is the head of the list and @xs@ its tail.
*/

//template<typename T>
//constexpr Maybe<PAIR_T(T, List<T>)> uncons(List<T> const& l);

DECLARE_FUNCTION_2(1, List<T0>, replicate, size_t, T0 const&);

/*
-- | 'takeWhile', applied to a predicate @p@ and a list @xs@, returns the
-- longest prefix (possibly empty) of @xs@ of elements that satisfy @p@:
--
-- > takeWhile (< 3) [1,2,3,4,1,2,3,4] == [1,2]
-- > takeWhile (< 9) [1,2,3] == [1,2,3]
-- > takeWhile (< 0) [1,2,3] == []
--
*/
DECLARE_FUNCTION_2(3, List<fdecay<T0> >, takeWhile, FUNCTION2(bool(T1), T2) const&, List<T0> const&);

/*
-- | 'dropWhile' @p xs@ returns the suffix remaining after 'takeWhile' @p xs@:
--
-- > dropWhile (< 3) [1,2,3,4,5,1,2,3] == [3,4,5,1,2,3]
-- > dropWhile (< 9) [1,2,3] == []
-- > dropWhile (< 0) [1,2,3] == [1,2,3]
*/
DECLARE_FUNCTION_2(3, List<fdecay<T0> >, dropWhile, FUNCTION2(bool(T1), T2) const&, List<T0> const&);

// take
DECLARE_FUNCTION_2(1, List<T0>, take, int, List<T0> const&);

// drop
DECLARE_FUNCTION_2(1, List<T0>, drop, int, List<T0> const&);

// splitAt
DECLARE_FUNCTION_2(1, PAIR_T(List<T0>, List<T0 >), splitAt, int, List<T0> const&);

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
*/

template<typename T>
using span_type = PAIR_T(List<fdecay<T> >, List<fdecay<T> >);

DECLARE_FUNCTION_2(2, span_type<T0>, span, FUNCTION2(bool(T0), T1) const&, List<fdecay<T0> > const&);

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
*/
DECLARE_FUNCTION_2(2, span_type<T0>, break_, FUNCTION2(bool(T0), T1) const&, List<fdecay<T0> > const&);

// | 'reverse' @xs@ returns the elements of @xs@ in reverse order.
// @xs@ must be finite.
// reverse :: [a] -> [a]
template<typename T>
constexpr List<T> reverse(List<T> const& l);

// 'lookup' @key assocs@ looks up a key in an association list.
// lookup :: (Eq a) => a -> [(a,b)] -> Maybe b
//DECLARE_FUNCTION_2(2, Maybe<T0>, lookup, T1 const&, List<PAIR_T(T1, T0)> const&);

// Map a function over a list and concatenate the results.
// concatMap               :: (a -> [b]) -> [a] -> [b]
DECLARE_FUNCTION_2(3, List<T1>, concatMap, FUNCTION2(List<T1>(T0 const&), T2) const&, List<T0> const&);

// nub :: (Eq a) => [a] -> [a]
// nub =  nubBy (==)
template<typename T>
constexpr List<T> nub(List<T> const& l);

// transpose               :: [[a]] -> [[a]]
template<typename T>
constexpr List<List<T> > transpose(List<List<T> > const& l);

_FUNCPROG2_END
