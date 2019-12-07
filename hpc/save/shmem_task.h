#pragma once

namespace hpc {

template<class T>
class shmem_task : public T
{
protected:
	shmem_task()
	{
		nodeNum = shmem_my_pe();
		nodeCount = shmem_n_pes();
	}

protected:
	virtual bool init()
	{
		shmem_tm = shmem_time();
		return T::init();
	}

	virtual void term()
	{
		T::term();
		shmemTime = shmem_time() - shmem_tm;
	}

public:
	std::size_t nodeNum, nodeCount;
	double shmemTime;

private:
	double shmem_tm;
};

}	// namespace hpc
