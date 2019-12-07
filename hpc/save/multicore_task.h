#pragma once

#include "multi_thread_task.h"

namespace hpc {

class multicore_task : public multi_thread_task
{
protected:
	virtual int getThreadCount() const;
};

}	// namespace hpc
