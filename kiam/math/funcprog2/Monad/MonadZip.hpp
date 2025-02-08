#pragma once

#include "../fwd/MonadZip_fwd.hpp"
#include "../detail/tuples.hpp"

_FUNCPROG2_BEGIN

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
template<typename MZ>
struct _MonadZip
{
    // mzip :: m a -> m b -> m (a,b)
    // mzip = mzipWith (,)
    template<typename MA, typename MB>
    static constexpr typeof_t<MA, pair_t<value_type_t<MA>, value_type_t<MB> > >
    mzip(MA const& ma, MB const& mb)
    {
        static_assert(is_same_monad_v<MA, MB>, "Should be the same monad");
        return MZ::mzipWith(_(comma<value_type_t<MA>, value_type_t<MB> >), ma, mb);
    }

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f ma mb = liftM (uncurry f) (mzip ma mb)
    template<typename MA, typename MB, typename C, typename FuncImpl>
    static constexpr typeof_t<MA, C> mzipWith(function2<C(value_type_t<MA> const&, value_type_t<MB> const&), FuncImpl> const& f, MA const& ma, MB const& mb)
    {
        static_assert(is_same_monad_v<MA, MB>, "Should be the same monad");
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
    static constexpr pair_t<typeof_t<MAB, fst_type_t<value_type_t<MAB> > >, typeof_t<MAB, snd_type_t<value_type_t<MAB> > > >
    munzip(MAB const& mab)
    {
        static_assert(is_monad_v<MAB>, "Should be a monad");
        static_assert(is_pair_v<value_type_t<MAB> >, "Should be a pair");
        using FST = fst_type_t<value_type_t<MAB> >;
        using SND = snd_type_t<value_type_t<MAB> >;
        return std::make_pair(liftM<MAB>(_(fst<FST, SND>), mab), liftM<MAB>(_(snd<FST, SND>), mab));
    }
};

template<typename MA, typename MB>
using mzip_type = same_monad_type<MA, MB, typeof_t<MA, pair_t<value_type_t<MA>, value_type_t<MB> > > >;


#define MZIP_TYPE_(MA, MB) BOOST_IDENTITY_TYPE((mzip_type<MA, MB>))
#define MZIP_TYPE(MA, MB) typename MZIP_TYPE_(MA, MB)

DECLARE_FUNCTION_2(2, MZIP_TYPE(T0, T1), mzip, T0 const&, T1 const&)
FUNCTION_TEMPLATE(2) constexpr MZIP_TYPE(T0, T1) mzip(T0 const& ma, T1 const& mb) {
    return MonadZip_t<T0>::mzip(ma, mb);
}

template<typename MA, typename MB, typename C, typename A, typename B>
using mzipWith_type = std::enable_if_t<
    is_same_monad_v<MA, MB> &&
    is_same_as_v<A, value_type_t<MA> > && is_same_as_v<B, value_type_t<MB> >,
    typeof_t<MA, C>
>;

#define MZIPWITH_TYPE_(MA, MB, C, A, B) BOOST_IDENTITY_TYPE((mzipWith_type<MA, MB, C, A, B>))
#define MZIPWITH_TYPE(MA, MB, C, A, B) typename MZIPWITH_TYPE_(MA, MB, C, A, B)

DECLARE_FUNCTION_3(6, MZIPWITH_TYPE(T1, T0, T2, T3, T4), mzipWith, FUNCTION2(T2(T3, T4), T5) const&, T1 const&, T0 const&)
FUNCTION_TEMPLATE(6) constexpr MZIPWITH_TYPE(T1, T0, T2, T3, T4) mzipWith(FUNCTION2(T2(T3, T4), T5) const& f, T1 const& ma, T0 const& mb) {
    return MonadZip_t<T1>::mzipWith(f, ma, mb);
}

template<typename MAB>
using munzip_type =
    std::enable_if_t<is_monad_v<MAB> && is_pair_v<value_type_t<MAB> >,
        pair_t<typeof_t<MAB, fst_type_t<value_type_t<MAB> > >, typeof_t<MAB, snd_type_t<value_type_t<MAB> > > >
    >;

template<typename T>
munzip_type<T> munzip(T const& mab) {
    return MonadZip_t<T>::munzip(mab);
}

_FUNCPROG2_END
