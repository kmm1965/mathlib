#pragma once

#include "task.h"

namespace hpc {

class multi_thread_task : public task
{
public:
	void execThread(size_t threadNum);

public:
	virtual bool execTask();

protected:
	virtual int getThreadCount() const = 0;
	virtual bool initThread(size_t threadNum);
	virtual void beforeIteration(size_t threadNum);
	virtual bool iteration(size_t threadNum) = 0;
	virtual void afterIteration(size_t threadNum);
	virtual void termThread(size_t threadNum);

public:
	unsigned int threadCount;
	boost::mutex dataMutex, debugMutex;
	boost::shared_ptr<boost::barrier> barrier;

private:
	class term_thread
	{
	public:
		term_thread(multi_thread_task *pTask, size_t threadNum);
		~term_thread();

	private:
		multi_thread_task *pTask;
		size_t threadNum;
	};
};

}	// namespace hpc
