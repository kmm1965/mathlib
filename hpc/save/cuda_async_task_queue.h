#pragma once

#include "async_task.hpp"
#include "cuda_utils.h"

#include <queue>

#include <cuda_runtime_api.h>

namespace hpc
{
	class cuda_async_task_queue;

	template<typename F>
	struct cuda_async_task_shared_state : async_task_shared_state<F>
	{
		typedef async_task_shared_state<F> super;

	private:
		cuda_async_task_shared_state(cuda_async_task_shared_state&);

	public:
		explicit cuda_async_task_shared_state(const F &f) : super(f){}
	};

	struct cuda_async_task : async_task_base<cuda_async_task>
	{
		typedef async_task_base<cuda_async_task> super;

		template<typename F>
		explicit cuda_async_task(const F &f, typename super::future_type *future = 0) : super(f, future){}

		template<class F>
		static async_task_base_shared_state *create_shared_state(const F &f){
			return new cuda_async_task_shared_state<F>(f);
		}
	};

	class cuda_async_task_queue : public async_task_queue
	{
	public:
		cuda_async_task_queue();

		virtual bool empty() const;
		virtual bool exec();

		cudaStream_t get_stream() const { return m_stream; }

	private:
		task_type *m_current_task;
		cuda_stream m_stream;
	};

	struct cuda_memcpy_async_task
	{
		template<typename T>
		cuda_memcpy_async_task(T *dst, const T *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) : dst(dst), src(src), count(count * sizeof(T)), kind(kind), stream(stream){}

		int operator()() const;

	private:
		void * const dst;
		const void * const src;
		size_t const count;
		cudaMemcpyKind const kind;
		cudaStream_t const stream;
	};

} // namespace hpc
