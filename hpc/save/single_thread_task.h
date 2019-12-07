#pragma once

#include "task.h"

namespace hpc {

class single_thread_task : public task
{
public:
	virtual bool execTask();

protected:
	virtual void beforeIteration();
	virtual bool iteration() = 0;
	virtual void afterIteration();
};

}	// namespace hpc
