#pragma once

#include <driver_types.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#include <boost/throw_exception.hpp>

namespace hpc {

int getCudaDeviceCount();
int cudaDeviceInfo();

size_t getCudaTotalGlobalMem(unsigned int deviceNum);
int getCudaMultiProcessorCount(unsigned int deviceNum);
int getCudaMaxThreadsPerBlock(unsigned int deviceNum);

void getCudaMemInfo(size_t *free, size_t *total);
size_t getCudaFreeMem();
size_t getCudaTotalMem();

void cudaSetDevice(int deviceNum);

class cuda_timer
{
public:
	cuda_timer();
	~cuda_timer();
	float elapsed();

private:
	cudaEvent_t evStart, evStop;
};

struct cuda_stream
{
	cuda_stream();
	~cuda_stream();

	operator cudaStream_t() const {
		return stream;
	}

private:
	cudaStream_t stream;
};

} // namespace hpc

#define CUDA_EXEC(Func, Args) \
	if((error_id = Func Args) != cudaSuccess) \
		BOOST_THROW_EXCEPTION(thrust::system_error(error_id, thrust::cuda_category()))
