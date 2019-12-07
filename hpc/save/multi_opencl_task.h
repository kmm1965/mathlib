#pragma once

#include "multi_thread_task.h"
#include "opencl_task.h"

namespace hpc {
class multi_opencl_task : public multi_thread_task, public opencl_task
{
protected:
	multi_opencl_task();

	int getComputeUnitCount(std::size_t deviceNum);
	int getMaxThreadsPerBlock(std::size_t deviceNum);

protected:
	virtual int getThreadCount() const;

protected:
	int openclDeviceCount;
};

}	// namespace hpc
