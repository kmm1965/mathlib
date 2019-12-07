#pragma once

#include "single_thread_task.h"
#include "opencl_task.h"

namespace hpc {

class single_opencl_task : public single_thread_task, public opencl_task
{
protected:
	virtual bool init();
};

}	// namespace hpc
