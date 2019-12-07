#pragma once

#include "multi_thread_task.h"
#include "cuda_task.h"

namespace hpc {

class multi_cuda_task : public multi_thread_task, public cuda_task
{
protected:
	multi_cuda_task();

	int getComputeUnitCount(unsigned int deviceNum);
	int getMaxThreadsPerBlock(unsigned int deviceNum);

protected:
	virtual int getThreadCount() const;
	virtual bool initThread(unsigned int threadNum);

protected:
	unsigned int cudaDeviceCount;
};

}	// namespace hpc
