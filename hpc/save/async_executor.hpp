#pragma once

#include <queue>
#include <vector>

#include "async_task.hpp"

namespace hpc {

	class async_executor
	{
		typedef std::vector<async_task_queue*> queue_list_type;

	public:
		async_executor();
		~async_executor();

		/// async_executor is not copyable.
		BOOST_THREAD_NO_COPYABLE(async_executor)

		void add_queue(async_task_queue &queue);
		void exec();

	private:
		bool queues_empty() const;
		bool exec_queues();

	private:
		queue_list_type m_queues;
	};

}	// namespace hpc
