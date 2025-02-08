#pragma once

#include "../fwd/Maybe_fwd.hpp"
#include "../Semigroup.hpp"
#include "../Monoid.hpp"
#include "../../pow.hpp"

_FUNCPROG_BEGIN

// Boolean monoid under conjunction ('&&').
struct All;

template<typename T> struct All_type;

template<>
struct All_type<bool> {
    using type = All;
};

struct All {
    using base_class = All;
    using value_type = bool;

    template<typename A>
    using type = typename All_type<A>::type;

    constexpr All(bool value) : value(value){}

    constexpr bool get() const { return value; }

private:
    bool const value;
};

inline constexpr All All_(bool value){
    return value;
}

inline constexpr bool getAll(All const& all){
    return all.get();
}

// Monoid
template<>
struct _is_monoid<All> : std::true_type {};

template<>
struct Monoid<All> : _Monoid<All>
{
    template<typename A>
    static constexpr All mempty(){
        static_assert(std::is_same_v<A, bool>, "Should be bool");
        return true;
    }
};

// Semigroup
template<>
struct _is_semigroup<All> : std::true_type {};

template<>
struct Semigroup<All> : _Semigroup<All>
{
    static constexpr All sg_op(All const& x, All const& y){
        return x.get() && y.get();
    }

    static constexpr All stimes(int n, All const& x){
        return _Semigroup<All>::stimesIdempotentMonoid<bool>(n, x);
    }
};

// Boolean monoid under disjunction ('||').
struct Any;

template<typename T> struct Any_type;

template<>
struct Any_type<bool> {
    using type = Any;
};

struct Any {
    using base_class = Any;
    using value_type = bool;

    template<typename A>
    using type = typename Any_type<A>::type;

    constexpr Any(bool value) : value(value){}

    constexpr bool get() const { return value; }

private:
    bool const value;
};

inline constexpr Any Any_(bool value){
    return value;
}

inline constexpr bool getAny(Any const& any){
    return any.get();
}

// Monoid
template<>
struct _is_monoid<Any> : std::true_type {};

template<>
struct Monoid<Any> : _Monoid<Any>
{
    template<typename A>
    static constexpr Any mempty(){
        static_assert(std::is_same_v<A, bool>, "Should be bool");
        return false;
    }
};

// Semigroup
template<>
struct _is_semigroup<Any> : std::true_type {};

template<>
struct Semigroup<Any> : _Semigroup<Any>
{
    static constexpr Any sg_op(Any const& x, Any const& y){
        return x.get() || y.get();
    }

    static constexpr Any stimes(int n, Any const& x){
        return _Semigroup<Any>::stimesIdempotentMonoid<bool>(n, x);
    }
};

/*
-- | Maybe monoid returning the leftmost non-Nothing value.
--
-- @'First' a@ is isomorphic to @'Alt' 'Maybe' a@, but precedes it
-- historically.
--
-- >>> getFirst (First (Just "hello") <> First Nothing <> First (Just "world"))
-- Just "hello"
--
-- Use of this type is discouraged. Note the following equivalence:
--
-- > Data.Monoid.First x === Maybe (Data.Semigroup.First x)
--
-- In addition to being equivalent in the structural sense, the two
-- also have 'Monoid' instances that behave the same. This type will
-- be marked deprecated in GHC 8.8, and removed in GHC 8.10.
-- Users are advised to use the variant from "Data.Semigroup" and wrap
-- it in 'Maybe'.
newtype First a = First { getFirst :: Maybe a }
*/
template<typename A>
struct First;

struct _First
{
    using base_class = _First;

    template<typename A>
    using type = First<A>;
};

template<typename A>
struct First : _First
{
    using value_type = A;

    First(Maybe<A> const& value) : value(value){}

    Maybe<A> const& get() const { return value; }

private:
    Maybe<A> const value;
};

template<typename A>
constexpr Maybe<A> getFirst(First<A> const& first){
    return first.get();
}

// Monoid
template<>
struct _is_monoid<_First> : std::true_type {};

template<>
struct Monoid<_First> : _Monoid<_First>
{
    //instance Monoid (First a) where
    //  mempty = First Nothing
    template<typename A>
    static constexpr First<A> mempty(){
        return Nothing<A>();
    }
};

// Semigroup
template<>
struct _is_semigroup<_First> : std::true_type {};

//instance Semigroup (First a) where
//  First Nothing <> b = b
//  a             <> _ = a
//  stimes = stimesIdempotentMonoid
template<>
struct Semigroup<_First> : _Semigroup<_First>
{
    template<typename A>
    static constexpr First<A> sg_op(First<A> const& x, First<A> const& y){
        return x.get() ? x : y;
    }

    template<typename A>
    static constexpr First<A> stimes(int n, First<A> const& x){
        return _Semigroup<_First>::stimesIdempotentMonoid<First<A> >(n, x);
    }
};

/*
-- | Maybe monoid returning the rightmost non-Nothing value.
--
-- @'Last' a@ is isomorphic to @'Dual' ('First' a)@, and thus to
-- @'Dual' ('Alt' 'Maybe' a)@
--
-- >>> getLast (Last (Just "hello") <> Last Nothing <> Last (Just "world"))
-- Just "world"
--
-- Use of this type is discouraged. Note the following equivalence:
--
-- > Data.Monoid.Last x === Maybe (Data.Semigroup.Last x)
--
-- In addition to being equivalent in the structural sense, the two
-- also have 'Monoid' instances that behave the same. This type will
-- be marked deprecated in GHC 8.8, and removed in GHC 8.10.
-- Users are advised to use the variant from "Data.Semigroup" and wrap
-- it in 'Maybe'.
newtype Last a = Last { getLast :: Maybe a }
*/
template<typename A>
struct Last;

struct _Last
{
    using base_class = _Last;

    template<typename A>
    using type = Last<A>;
};

template<typename A>
struct Last : _Last
{
    using value_type = A;

    Last(Maybe<A> const& value) : value(value){}

    Maybe<A> const& get() const { return value; }

private:
    Maybe<A> const value;
};

template<typename A>
constexpr Maybe<A> getLast(Last<A> const& last){
    return last.get();
}

// Monoid
template<>
struct _is_monoid<_Last> : std::true_type {};

template<>
struct Monoid<_Last> : _Monoid<_Last>
{
    //instance Monoid (Last a) where
    //  mempty = Last Nothing
    template<typename A>
    static constexpr Last<A> mempty(){
        return Nothing<A>();
    }
};

// Semigroup
template<>
struct _is_semigroup<_Last> : std::true_type {};

//instance Semigroup (Last a) where
//  a <> Last Nothing = a
//  _ <> b                   = b
//  stimes = stimesIdempotentMonoid
template<>
struct Semigroup<_Last> : _Semigroup<_Last>
{
    template<typename A>
    static constexpr Last<A> sg_op(Last<A> const& x, Last<A> const& y){
        return y.get() ? y : x;
    }

    template<typename A>
    static constexpr Last<A> stimes(int n, Last<A> const& x){
        return _Semigroup<_Last>::stimesIdempotentMonoid<Last<A> >(n, x);
    }
};

/*
-- | This data type witnesses the lifting of a 'Monoid' into an
-- 'Applicative' pointwise.
--
newtype Ap f a = Ap { getAp :: f a }

instance (Applicative f, Semigroup a) => Semigroup (Ap f a) where
        (Ap x) <> (Ap y) = Ap $ liftA2 (<>) x y

instance (Applicative f, Monoid a) => Monoid (Ap f a) where
        mempty = Ap $ pure mempty

instance (Applicative f, Bounded a) => Bounded (Ap f a) where
  minBound = pure minBound
  maxBound = pure maxBound

instance (Applicative f, Num a) => Num (Ap f a) where
  (+)         = liftA2 (+)
  (*)         = liftA2 (*)
  negate      = fmap negate
  fromInteger = pure . fromInteger
  abs         = fmap abs
  signum      = fmap signum
*/

//newtype Max a = Max {getMax :: Maybe a}
template<typename A>
struct Max;

struct _Max
{
    using base_class = _Max;

    template<typename A>
    using type = Max<A>;
};

template<typename A>
struct Max : _Max
{
    using value_type = A;

    Max(Maybe<A> const& value) : value(value){}

    Maybe<A> const& get() const { return value; }

private:
    Maybe<A> const value;
};

template<typename A>
Max<A> Max_(Maybe<A> const& x){
    return x;
}

template<typename A>
constexpr Maybe<A> getMax(Max<A> const& max_){
    return max_.get();
}

// Monoid
template<>
struct _is_monoid<_Max> : std::true_type {};

template<>
struct Monoid<_Max> : _Monoid<_Max>
{
    //instance Ord a => Monoid (Max a) where
    //  mempty = Max Nothing
    template<typename A>
    static constexpr Max<A> mempty(){
        return Nothing<A>();
    }
};

// Semigroup
template<>
struct _is_semigroup<_Max> : std::true_type {};

//instance Ord a => Semigroup (Max a) where
//    m <> Max Nothing = m
//    Max Nothing <> n = n
//    (Max m@(Just x)) <> (Max n@(Just y))
//      | x >= y    = Max m
//      | otherwise = Max n
template<>
struct Semigroup<_Max> : _Semigroup<_Max>
{
    template<typename A>
    static constexpr Max<A> sg_op(Max<A> const& mx, Max<A> const& my){
        const Maybe<A>& m = mx.get(), & n = my.get();
        if (!m) return n;
        else if (!n) return m;
        const A& x = m.value(), & y = n.value();
        return x >= y ? m : n;
    }
};

//newtype Min a = Min{ getMin::Maybe a }
template<typename A>
struct Min;

struct _Min
{
    using base_class = _Min;

    template<typename A>
    using type = Min<A>;
};

template<typename A>
struct Min : _Min
{
    using value_type = A;

    Min(Maybe<A> const& value) : value(value){}

    Maybe<A> const& get() const { return value; }

private:
    Maybe<A> const value;
};

template<typename A>
Min<A> Min_(Maybe<A> const& x){
    return x;
}

template<typename A>
constexpr Maybe<A> getMin(Min<A> const& min_){
    return min_.get();
}

// Monoid
template<>
struct _is_monoid<_Min> : std::true_type {};

template<>
struct Monoid<_Min> : _Monoid<_Min>
{
    //instance Ord a => Monoid (Min a) where
    //  mempty = Min Nothing
    template<typename A>
    static constexpr Min<A> mempty(){
        return Nothing<A>();
    }
};

// Semigroup
template<>
struct _is_semigroup<_Min> : std::true_type {};

//instance Ord a => Semigroup (Min a) where
//    m <> Min Nothing = m
//    Min Nothing <> n = n
//    (Min m@(Just x)) <> (Min n@(Just y))
//      | x <= y    = Min m
//      | otherwise = Min n
template<>
struct Semigroup<_Min> : _Semigroup<_Min>
{
    template<typename A>
    static constexpr Min<A> sg_op(Min<A> const& mx, Min<A> const& my){
        const Maybe<A>& m = mx.get(), & n = my.get();
        if (!m) return n;
        else if (!n) return m;
        const A& x = m.value(), & y = n.value();
        return x <= y ? m : n;
    }
};

/*
-- | Monoid under addition.
--
-- >>> getSum (Sum 1 <> Sum 2 <> mempty)
-- 3
newtype Sum a = Sum { getSum :: a }

instance Num a => Semigroup (Sum a) where
        (<>) = coerce ((+) :: a -> a -> a)
        stimes n (Sum a) = Sum (fromIntegral n * a)

instance Num a => Monoid (Sum a) where
        mempty = Sum 0

instance Functor Sum where
    fmap     = coerce

instance Applicative Sum where
    pure     = Sum
    (<*>)    = coerce

instance Monad Sum where
    m >>= k  = k (getSum m)
*/
template<typename A>
struct Sum;

struct _Sum
{
    using base_class = _Sum;

    template<typename A>
    using type = Sum<A>;
};

template<typename A>
struct Sum : _Sum
{
    using value_type = A;

    Sum(A const& value) : value(value){}

    A const& get() const { return value; }

private:
    A const value;
};

template<typename A>
Sum<A> Sum_(A const& x){
    return x;
}

template<typename A>
constexpr A getSum(Sum<A> const& sum){
    return sum.get();
}

// Monoid
template<>
struct _is_monoid<_Sum> : std::true_type {};

template<>
struct Monoid<_Sum> : _Monoid<_Sum>
{
    //instance Num a => Monoid (Sum a) where
    //  mempty = Sum 0
    template<typename A>
    static constexpr Sum<A> mempty(){
        return A();
    }
};

// Semigroup
template<>
struct _is_semigroup<_Sum> : std::true_type {};

//instance Num a => Semigroup (Sum a) where
template<>
struct Semigroup<_Sum> : _Semigroup<_Sum>
{
    //  (<>) = coerce ((+) :: a -> a -> a)
    template<typename A>
    static constexpr Sum<A> sg_op(Sum<A> const& x, Sum<A> const& y){
        return x.get() + y.get();
    }

    //  stimes n (Sum a) = Sum (fromIntegral n * a)
    template<typename A>
    static constexpr Sum<A> stimes(int n, Sum<A> const& x){
        return n * x.get();
    }
};

/*
-- | Monoid under multiplication.
--
-- >>> getProduct (Product 3 <> Product 4 <> mempty)
-- 12
newtype Product a = Product { getProduct :: a }

instance Num a => Semigroup (Product a) where
        (<>) = coerce ((*) :: a -> a -> a)
        stimes n (Product a) = Product (a ^ n)

instance Num a => Monoid (Product a) where
        mempty = Product 1

instance Functor Product where
    fmap     = coerce

instance Applicative Product where
    pure     = Product
    (<*>)    = coerce

instance Monad Product where
    m >>= k  = k (getProduct m)
*/

template<typename A>
struct Product;

struct _Product
{
    using base_class = _Product;

    template<typename A>
    using type = Product<A>;
};

template<typename A>
struct Product : _Product
{
    using value_type = A;

    Product(A const& value) : value(value){}

    A const& get() const { return value; }

private:
    A const value;
};

template<typename A>
Product<A> Product_(A const& x){
    return x;
}

template<typename A>
constexpr A getProduct(Product<A> const& product){
    return product.get();
}

// Monoid
template<>
struct _is_monoid<_Product> : std::true_type {};

template<>
struct Monoid<_Product> : _Monoid<_Product>
{
    //instance Num a => Monoid (Product a) where
    //  mempty = Product 1
    template<typename A>
    static constexpr Product<A> mempty(){
        return 1;
    }
};

// Semigroup
template<>
struct _is_semigroup<_Product> : std::true_type {};

//instance Num a => Semigroup (Product a) where
template<>
struct Semigroup<_Product> : _Semigroup<_Product>
{
    //  (<>) = coerce ((*) :: a -> a -> a)
    template<typename A>
    static constexpr Product<A> sg_op(Product<A> const& x, Product<A> const& y){
        return x.get() * y.get();
    }

    //  stimes n (Product a) = Product (a ^ n)
    template<typename A>
    static constexpr Product<A> stimes(int n, Product<A> const& x){
        return _KIAM_MATH::pown(x.get(), n);
    }
};

_FUNCPROG_END
