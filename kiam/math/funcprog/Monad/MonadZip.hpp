#pragma once

#include "../detail/tuples.hpp"

_FUNCPROG_BEGIN

/*
-- | `MonadZip` type class. Minimal definition: `mzip` or `mzipWith`
--
-- Instances should satisfy the laws:
--
-- * Naturality :
--
--   > liftM (f *** g) (mzip ma mb) = mzip (liftM f ma) (liftM g mb)
--
-- * Information Preservation:
--
--   > liftM (const ()) ma = liftM (const ()) mb
--   > ==>
--   > munzip (mzip ma mb) = (ma, mb)
--
class Monad m => MonadZip m where
    {-# MINIMAL mzip | mzipWith #-}

    mzip :: m a -> m b -> m (a,b)
    mzip = mzipWith (,)

    mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    mzipWith f ma mb = liftM (uncurry f) (mzip ma mb)

    munzip :: m (a,b) -> (m a, m b)
    munzip mab = (liftM fst mab, liftM snd mab)
    -- munzip is a member of the class because sometimes
    -- you can implement it more efficiently than the
    -- above default code.  See Trac #4370 comment by giorgidze
*/
template<typename T>
struct MonadZip;

template<typename T>
using MonadZip_t = MonadZip<base_class_t<T> >;

template<typename MZ>
struct _MonadZip
{
    // mzip :: m a -> m b -> m (a,b)
    // mzip = mzipWith (,)
    template<typename MA, typename MB>
    static typeof_t<MA, pair_t<value_type_t<MA>, value_type_t<MB> > >
    mzip(MA const& ma, MB const& mb)
    {
        static_assert(is_same_monad< base_class_t<MA>, base_class_t<MB> >::value, "Should be the same monad");
        return MZ::mzipWith(_(comma<value_type_t<MA>, value_type_t<MB> >), ma, mb);
    }

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f ma mb = liftM (uncurry f) (mzip ma mb)
    template<typename MA, typename MB, typename C>
    static typeof_t<MA, C> mzipWith(function_t<C(value_type_t<MA> const&, value_type_t<MB> const&)> const& f, MA const& ma, MB const& mb)
    {
        static_assert(is_same_monad< base_class_t<MA>, base_class_t<MB> >::value, "Should be the same monad");
        return liftM<MA>(_uncurry(f), MZ::mzip(ma, mb));
    }

    /*
    munzip :: m (a,b) -> (m a, m b)
    munzip mab = (liftM fst mab, liftM snd mab)
    -- munzip is a member of the class because sometimes
    -- you can implement it more efficiently than the
    -- above default code.  See Trac #4370 comment by giorgidze
    */
    template<typename MAB>
    static pair_t<typeof_t<MAB, fst_type_t<value_type_t<MAB> > >, typeof_t<MAB, snd_type_t<value_type_t<MAB> > > >
    munzip(MAB const& mab)
    {
        static_assert(is_monad_t<MAB>::value, "Should be a monad");
        static_assert(is_pair<value_type_t<MAB> >::value, "Should be a pair");
        using FST = fst_type_t<value_type_t<MAB> >;
        using SND = snd_type_t<value_type_t<MAB> >;
        return std::make_pair(liftM<MAB>(_(fst<FST, SND>), mab), liftM<MAB>(_(snd<FST, SND>), mab));
    }
};

template<typename MA, typename MB>
using mzip_type =
    typename std::enable_if<is_same_monad_t<MA, MB>::value,
        typeof_t<MA, pair_t<value_type_t<MA>, value_type_t<MB> > >
    >::type;


#define MZIP_TYPE_(MA, MB) BOOST_IDENTITY_TYPE((mzip_type<MA, MB>))
#define MZIP_TYPE(MA, MB) typename MZIP_TYPE_(MA, MB)

DEFINE_FUNCTION_2(2, MZIP_TYPE(T0, T1), mzip, T0 const&, ma, T1 const&, mb, return MonadZip_t<T0>::mzip(ma, mb);)

template<typename MA, typename MB, typename C, typename A, typename B>
using mzipWith_type = typename std::enable_if<
    is_same_monad_t<MA, MB>::value &&
    is_same_as<A, value_type_t<MA> >::value && is_same_as<B, value_type_t<MB> >::value,
    typeof_t<MA, C>
>::type;

#define MZIPWITH_TYPE_(MA, MB, C, A, B) BOOST_IDENTITY_TYPE((mzipWith_type<MA, MB, C, A, B>))
#define MZIPWITH_TYPE(MA, MB, C, A, B) typename MZIPWITH_TYPE_(MA, MB, C, A, B)

DEFINE_FUNCTION_3(5, MZIPWITH_TYPE(T1, T0, T2, T3, T4), mzipWith, function_t<T2(T3, T4)> const&, f,
    T1 const&, ma, T0 const&, mb, return MonadZip_t<T1>::mzipWith(f, ma, mb);)

template<typename MAB>
using munzip_type =
    typename std::enable_if<is_monad_t<MAB>::value && is_pair<value_type_t<MAB> >::value,
        pair_t<typeof_t<MAB, fst_type_t<value_type_t<MAB> > >, typeof_t<MAB, snd_type_t<value_type_t<MAB> > > >
    >::type;

template<typename T>
munzip_type<T> munzip(T const& mab) {
    return MonadZip_t<T>::munzip(mab);
}

_FUNCPROG_END
