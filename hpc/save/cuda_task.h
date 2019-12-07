#pragma once

namespace hpc {

class cuda_task
{
protected:
	int getComputeUnitCount();
	int getMaxThreadsPerBlock();
};

}	// namespace hpc
