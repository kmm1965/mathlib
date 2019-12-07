#pragma once

#include "single_thread_task.h"
#include "cuda_task.h"

namespace hpc {

class single_cuda_task : public single_thread_task, public cuda_task
{
};

} // namespace hpc
