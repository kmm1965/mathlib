#pragma once

#include "context.hpp"

_KIAM_MATH_BEGIN

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  512
#endif

static const char szHeader[] =
"namespace kiam { namespace math {\n"
"}}";

template<typename TAG>
template<class Closure>
void opencl_executor<TAG>::operator()(size_t size, Closure const& closure) const
{
    namespace compute = ::boost::compute;
    const std::string Closure_type_name = ::boost::core::demangle(typeid(Closure).name());
    //std::cout << "Hello from opencl_exec_closure, closure class=" << Closure_type_name << std::endl;
    std::ostringstream ss;
    ss << szHeader << std::endl << "__kernel void opencl_kernel(" << Closure_type_name << " closure){" << std::endl
        << "uint i = get_global_id(0); if(i < " << size << ") closure(i); }";
    compute::kernel kernel = compute::kernel::create_with_source(ss.str(), "opencl_kernel", compute::system::default_context());
    kernel.set_arg(0, sizeof(closure), &closure);
    const size_t
        r = size % BLOCK_SIZE,
        global_work_size = size + (r == 0 ? 0 : BLOCK_SIZE - r);
    compute::system::default_queue().enqueue_1d_range_kernel(kernel, 0, global_work_size, BLOCK_SIZE).wait();
}

_KIAM_MATH_END
