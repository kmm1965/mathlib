#pragma once

#include "context.hpp"

_KIAM_MATH_BEGIN

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  512
#endif

static const char szHeader[] =
"namespace kiam { namespace math {\n"
"}}";

template<class Callback>
void opencl_exec_callback(Callback& callback, size_t size)
{
    namespace compute = ::boost::compute;
    const std::string Callback_type_name = ::boost::core::demangle(typeid(Callback).name());
    //std::cout << "Hello from opencl_exec_callback, callback class=" << Callback_type_name << std::endl;
    std::ostringstream ss;
    ss << szHeader << std::endl << "__kernel void opencl_kernel(" << Callback_type_name << " callback){" << std::endl
        << "uint i = get_global_id(0); if(i < " << size << ") callback[i]; }";
    compute::kernel kernel = compute::kernel::create_with_source(ss.str(), "opencl_kernel", compute::system::default_context());
    kernel.set_arg(0, sizeof(callback), &callback);
    const size_t
        r = size % BLOCK_SIZE,
        global_work_size = size + (r == 0 ? 0 : BLOCK_SIZE - r);
    compute::system::default_queue().enqueue_1d_range_kernel(kernel, 0, global_work_size, BLOCK_SIZE).wait();
}

_KIAM_MATH_END
