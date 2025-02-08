#pragma once

_KIAM_MATH_BEGIN

namespace tp
{

template<typename T, typename V, class BO>
struct reduce_g1_struct_stride
{
    reduce_g1_struct_stride(size_t i, size_t threads, const T *data, size_t n, V r, size_t stride, BO bin_op, V *result) :
        i(i), threads(threads), data(data), n(n), r(r), stride(stride), bin_op(bin_op), result(result){}

    void operator()()
    {
        for(size_t j = i; j < n; j += threads)
            r = bin_op(r, data[j * stride]);
        result[i] = r;
    }

private:
    const size_t i, threads, n, stride;
    const T *data;
    V r;
    BO bin_op;
    V *result;
};

template<typename T, typename V, class BO>
struct reduce_g1_struct
{
    reduce_g1_struct(size_t i, size_t threads, const T *data, size_t n, V r, BO bin_op, V *result) :
        i(i), threads(threads), data(data), n(n), r(r), bin_op(bin_op), result(result){}

    struct iterator
    {
        typedef V value_type;
        typedef std::forward_iterator_tag iterator_category;
        typedef size_t difference_type;
        typedef const value_type *const_pointer;
        typedef value_type const& const_reference;
        typedef const_pointer pointer;
        typedef const_reference reference;

        iterator(const T *data, size_t threads) : data(data), threads(threads){}

        const_reference operator*() const {
            return *data;
        }

        void operator++(){
            data += threads;
        }

        iterator operator+(size_t n) const {
            return iterator(data + n * threads, threads);
        }

        bool operator==(iterator other) const
        {
            assert(threads == other.threads);
            return data == other.data;
        }

        bool operator!=(iterator other) const
        {
            assert(threads == other.threads);
            return data != other.data;
        }

        const T *data;
        const size_t threads;
    };

    void operator()()
    {
        iterator begin(data + i, threads);
        result[i] = std::accumulate(begin, begin + (n - i + threads - 1) / threads, r, bin_op);
    }

private:
    const size_t i, threads, n;
    const T *data;
    const V r;
    BO bin_op;
    V *result;
};

template<typename T, typename V, class BO>
reduce_g1_struct_stride<T, V, BO>
reduce_g1_stride(size_t i, size_t threads, const T *data, size_t n, V r, size_t stride, BO bin_op, V *result){
    return reduce_g1_struct_stride<T, V, BO>(i, threads, data, n, r, stride, bin_op, result);
}

template<typename T, typename V, class BO>
reduce_g1_struct<T, V, BO>
reduce_g1(size_t i, size_t threads, const T *data, size_t n, V r, BO bin_op, V *result){
    return reduce_g1_struct<T, V, BO>(i, threads, data, n, r, bin_op, result);
}

template<typename V, class BO>
struct reduce_g_final_struct
{
    reduce_g_final_struct(size_t i, size_t threads, size_t size, BO bin_op, V *result) :
        i(i), threads(threads), size(size), bin_op(bin_op), result(result){}

    void operator()()
    {
        if(i + threads < size)
            result[i] = bin_op(result[i], result[i + threads]);
    }

private:
    size_t i, threads, size;
    BO bin_op;
    V *result;
};

template<typename V, class BO>
reduce_g_final_struct<V, BO>
reduce_g_final(size_t i, size_t threads, size_t size, BO bin_op, V *result){
    return reduce_g_final_struct<V, BO>(i, threads, size, bin_op, result);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_n(const T *data, size_t n, /*size_t stride,*/ V init, BO1 bin_op1, BO2 bin_op2, boost::basic_thread_pool &pool, unsigned thread_count, math_vector<V> &result)
{
    if(n == 0)
        return init;
    size_t size = thread_count;
    if(size > n)
        size = n;
    result.resize(size);
    V *result_data = result.data_pointer();
    std::vector<boost::BOOST_THREAD_FUTURE<void> > futures;
    {
        futures.resize(size);
        for (size_t i = 0; i < size; ++i){
            boost::packaged_task<void> pt(reduce_g1(i, size, data, n, init, /*stride,*/ bin_op1, result_data));
            futures[i] = pt.get_future();
            pool.submit(boost::move(pt));
        }
        boost::wait_for_all(std::begin(futures), std::end(futures));
    }
    while(size > 1){
        futures.resize((size + 1) / 2);
        for (size_t i = 0; i < futures.size(); ++i){
            boost::packaged_task<void> pt(reduce_g_final(i, futures.size(), size, bin_op2, result_data));
            futures[i] = pt.get_future();
            pool.submit(boost::move(pt));
        }
        boost::wait_for_all(std::begin(futures), std::end(futures));
        size = std::size(futures);
    }
    return result.front();
}

template<typename T, typename V, class BO1, class BO2>
V reduce_n(const T *data, size_t n, size_t stride, V init, BO1 bin_op1, BO2 bin_op2, boost::basic_thread_pool &pool, unsigned thread_count)
{
    math_vector<V> result;
    return reduce_n(data, n, stride, init, bin_op1, bin_op2, pool, thread_count, result);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_vector(math_vector<T> const& v, V init, BO1 bin_op1, BO2 bin_op2, boost::basic_thread_pool &pool, unsigned thread_count, math_vector<V> &result){
    return reduce_n(v.data_pointer(), v.size(), init, bin_op1, bin_op2, pool, thread_count, result);
}

template<typename T, typename V, class BO>
V reduce_vector(math_vector<T> const& v, V init, BO bin_op, boost::basic_thread_pool &pool, unsigned thread_count, math_vector<V> &result){
    return reduce_n(v.data_pointer(), v.size(), init, bin_op, bin_op, pool, thread_count, result);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_vector(math_vector<T> const& v, V init, BO1 bin_op1, BO2 bin_op2, boost::basic_thread_pool &pool, unsigned thread_count)
{
    math_vector<V> result;
    return reduce_n(v.data_pointer(), v.size(), init, bin_op1, bin_op2, pool, thread_count, result);
}

template<typename T, typename V, class BO>
V reduce_vector(math_vector<T> const& v, V init, BO bin_op, boost::basic_thread_pool &pool, unsigned thread_count){
    return reduce_vector(v, init, bin_op, bin_op, pool, thread_count);
}

template<typename T, typename V, class BO>
V reduce_vector(math_vector<T> const& v, V init, BO bin_op, math_vector<V> &result)
{
    boost::basic_thread_pool pool;
    return reduce_vector(v, init, bin_op, bin_op, pool, boost::thread::hardware_concurrency(), result);
}

template<typename T, typename V, class BO>
V reduce_vector(math_vector<T> const& v, V init, BO bin_op)
{
    math_vector<V> result;
    return reduce_vector(v, init, bin_op, result);
}

} // namespace tp

_KIAM_MATH_END
