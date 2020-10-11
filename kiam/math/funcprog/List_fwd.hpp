#pragma once

#include "func.hpp"
#include "funcprog_common.hpp"

_FUNCPROG_BEGIN

class list_error;
class empty_list_error;

struct _List;

template<typename T>
struct List;

using String = List<char>;
using wString = List<wchar_t>;

// List
template<typename T>
constexpr List<T> operator>>(T const& value, List<T> const& l);

template<typename T>
constexpr List<T> operator<<(List<T> const& l, T const& value);

template<typename T>
constexpr List<T> operator+(List<T> const&l1, List<T> const&l2);

template<typename T> constexpr T const& head(List<T> const&);
template<typename T> constexpr List<T> tail(List<T> const&);
template<typename T> constexpr T const& last(List<T> const&);
template<typename T> constexpr List<T> init(List<T> const&);
template<typename T> constexpr bool null(List<T> const&);
template<typename T> constexpr int length(List<T> const&);
DECLARE_FUNCTION_2(1, List<fdecay<T0> >, filter, function_t<bool(T0)> const&, List<fdecay<T0> > const&);
DECLARE_FUNCTION_2(1, List<T0>, cons, T0 const& value, List<T0> const&);
DECLARE_FUNCTION_2(1, List<T0>, concat2, List<T0> const&, List<T0> const&);
template<typename T> constexpr List<T> sort(List<T> const&);

// The concatenation of all the elements of a container of lists.
template<typename T> constexpr List<T> concat(List<List<T> > const&);

inline constexpr std::string strconcat(List<std::string> const&);

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
DECLARE_FUNCTION_2(3, List<T0>, nubBy, function_t<bool(T1, T2)> const&, List<T0> const&);

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
DECLARE_FUNCTION_2(1, List<T0>, _delete, T0 const&, List<T0> const&);

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
DECLARE_FUNCTION_3(3, List<T0>, deleteBy, function_t<bool(T1, T2)> const&, T0 const&, List<T0> const&);

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
DECLARE_FUNCTION_2(1, List<T0>, _union, List<T0> const&, List<T0> const&);

// The 'unionBy' function is the non-overloaded version of 'union'.
DECLARE_FUNCTION_3(3, List<T0>, unionBy, function_t<bool(T1, T2)> const&, List<T0> const&, List<T0> const&);

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
DECLARE_FUNCTION_2(1, List<T0>, intersect, List<T0> const&, List<T0> const&);

// The 'intersectBy' function is the non-overloaded version of 'intersect'.
DECLARE_FUNCTION_3(3, List<T0>, intersectBy, function_t<bool(T1, T2)> const&, List<T0> const&, List<T0> const&);

/*
-- Not exported:
-- We want to make every element in the 'intersperse'd list available
-- as soon as possible to avoid space leaks. Experiments suggested that
-- a separate top-level helper is more efficient than a local worker.
*/
DECLARE_FUNCTION_2(1, List<T0>, prependToAll, T0 const&, List<T0> const&);

/*
-- | The 'intersperse' function takes an element and a list and
-- \`intersperses\' that element between the elements of the list.
-- For example,
--
-- >>> intersperse ',' "abcde"
-- "a,b,c,d,e"
*/
DECLARE_FUNCTION_2(1, List<T0>, intersperse, T0 const&, List<T0> const&);

/*
-- | 'intercalate' @xs xss@ is equivalent to @('concat' ('intersperse' xs xss))@.
-- It inserts the list @xs@ in between the lists in @xss@ and concatenates the
-- result.
--
-- >>> intercalate ", " ["Lorem", "ipsum", "dolor"]
-- "Lorem, ipsum, dolor"
*/
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
DECLARE_FUNCTION_2(2, PAIR_T(List<T0>, List<T0>), partition, function_t<bool(T1)> const&, List<T0> const&);
DECLARE_FUNCTION_3(2, PAIR_T(List<T0>, List<T0>), select, function_t<bool(T1)> const&, T0 const&, const PAIR_T(List<T0>, List<T0>)&);

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
DECLARE_FUNCTION_2(1, List<T0>, insert, T0 const&, List<T0> const&);

// The non-overloaded version of 'insert'.
DECLARE_FUNCTION_3(3, List<T0>, insertBy, function_t<Ordering(T1, T2)> const&, T0 const&, List<T0> const&);

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::List<T> const& l);

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::List<T> const& l);

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::List<_FUNCPROG::f0<T> > const& l);

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::List<_FUNCPROG::f0<T> > const& l);

} // namespace std
