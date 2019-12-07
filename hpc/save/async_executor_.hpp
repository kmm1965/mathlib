#pragma once

#include <boost/thread/scoped_thread.hpp>
#include <boost/thread/concurrent_queues/sync_queue.hpp>

#include "async_task.hpp"

namespace hpc {

	class async_executor
	{
	public:
		// typedef async_task_base_shared_state closure_type;
		typedef  boost::executors::work work;
		typedef boost::scoped_thread<> thread_t;

		async_executor();
		~async_executor();

		void close(){ work_queue.close(); }
		bool closed(){ return work_queue.closed(); }

		bool try_executing_one();

		template <typename Closure>
		void submit(Closure &closure){
			work_queue.push(work(closure));
		}

		/**
		* \b Requires: This must be called from an scheduled task.
		*
		* \b Effects: reschedule functions until pred()
		*/
		template <typename Pred>
		bool reschedule_until(Pred const& pred)
		{
			do {
				if (!try_executing_one())
					return false;
			} while (!pred());
			return true;
		}

	private:
		void schedule_one_or_yield();
		void worker_thread();

		/// the thread safe work queue
		boost::concurrent::sync_queue<closure_type*> work_queue;
		thread_t m_thread;
	};

} // namespace hpc
