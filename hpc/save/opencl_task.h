#pragma once

#include "opencl_utils.h"

namespace hpc {

class opencl_task
{
protected:
	bool init();

	cl::Program createProgram(const char *fileName);

	int getComputeUnitCount();
	int getMaxThreadsPerBlock();

public:
	VECTOR_CLASS<cl::Platform> platforms;
	VECTOR_CLASS<cl::Device> devices;
	bool useLocalMem;
	cl::Context context;
	float openclTime;
};

}	// namespace hpc
