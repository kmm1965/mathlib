#pragma once

#include "async_task.hpp"

namespace hpc
{
	class serial_async_task_queue;

	template<typename F>
	struct serial_async_task_shared_state : async_task_shared_state<F>
	{
		typedef async_task_shared_state<F> super;

	private:
		serial_async_task_shared_state(serial_async_task_shared_state&);

	public:
		serial_async_task_shared_state(const F &f) : super(f){}
	};

	struct serial_async_task : async_task_base<serial_async_task>
	{
		typedef async_task_base<serial_async_task> super;

		template<typename F>
		explicit serial_async_task(const F &f, typename super::future_type *future = 0) : super(f, future){}

		BOOST_THREAD_MOVABLE_ONLY(serial_async_task)

		template<class F>
		static async_task_base_shared_state *create_shared_state(const F &f){
			return new serial_async_task_shared_state<F>(f);
		}
	};

	class serial_async_task_queue : public async_task_queue
	{
	public:
		virtual bool exec();
	};

	struct serial_memcpy_async_task
	{
		template<typename T>
		serial_memcpy_async_task(T *dst, const T *src, size_t count) : dst(dst), src(src), count(count * sizeof(T)){}

		void operator()() const;

	private:
		void * const dst;
		const void * const src;
		size_t const count;
	};

} // namespace hpc
