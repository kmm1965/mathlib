#pragma once

namespace hpc {

template<class T>
class mpi_task : public T
{
protected:
	mpi_task(typename mpi::environment& env_, typename mpi::communicator& comm_) : env(env_), comm(comm_)
	{
		processNum = comm.rank();
		processCount = comm.size(); 
	}

protected:
	virtual bool init()
	{
		mpiTimer.restart();
		return T::init();
	}

	virtual void term()
	{
		T::term();
		mpiTime = mpiTimer.elapsed();
	}

public:
	size_t processNum, processCount;
	double mpiTime;

protected:
	typename mpi::environment& env;
	typename mpi::communicator& comm;

private:
	typename mpi::timer mpiTimer;
};

} // namespace hpc
