#pragma once

#include "mpi_task.h"

namespace hpc {

template<class T>
class multi_cuda_host_mpi_task : public mpi_task<T>
{
	typedef mpi_task<T> base;

protected:
	multi_cuda_host_mpi_task(mpi::communicator &comm) : base(comm)
	{
		processGroupSize = boost::thread::hardware_concurrency() - 1;
		processGroupCount = (base::processCount + processGroupSize - 1) / processGroupSize;
		processGroupNum = base::processNum / processGroupSize;
		processInGroupNum = base::processNum % processGroupSize;
		assert(processGroupNum < processGroupCount);
		if(processGroupNum == processGroupCount - 1)
			processGroupSize = base::processCount - processGroupNum * processGroupSize;
		hostProcessCount = processGroupSize - 1;
		hostProcessNum = processInGroupNum == 0 ? 0 : processInGroupNum - 1;
		assert(hostProcessNum < hostProcessCount);
	}

	virtual int getThreadCount() const {
		return processInGroupNum == 0 ? base::getThreadCount() : 1;
	}

	virtual bool initThread(std::size_t threadNum){
		return processInGroupNum == 0 ? base::initThread(threadNum) : true;
	}
	
	virtual bool iteration(std::size_t threadNum)
	{
		if(processInGroupNum == 0)
			return iterationCuda(threadNum);
		else return iterationHost();
	}

	virtual bool iterationCuda(std::size_t threadNum) = 0;
	virtual bool iterationHost() = 0;

protected:
	uint processGroupSize, processGroupCount, processGroupNum, processInGroupNum;
	uint hostProcessCount, hostProcessNum;
};

} // namespace hpc
