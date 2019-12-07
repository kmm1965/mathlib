#pragma once

#include "async_task.hpp"
#include "mpi.h"

#include <list>

namespace hpc
{

	class mpi_async_task_queue : public async_task_queue
	{
		struct mpi_async_info
		{
			typedef async_task_queue::task_type task_type;

			mpi_async_info(int request, task_type *async_task) : request(request), async_task(async_task){}

			int request;
			task_type *async_task;
		};

		typedef std::list<mpi_async_info> work_queue_type;

	public:
		virtual bool empty() const;
		virtual bool exec();

	private:
		work_queue_type m_work_tasks;
	};

	struct mpi_async_task_base
	{
	protected:
		mpi_async_task_base(void *buf, int count, MPI_Datatype datatype, int dest_source, int tag, MPI_Comm comm) :
			buf(buf), count(count), datatype(datatype), dest_source(dest_source), tag(tag), comm(comm){}

		template<typename T>
		mpi_async_task_base(T *buf, int count, int dest_source, int tag, MPI_Comm comm) :
			buf(buf), count(count), datatype(mpi::get_mpi_datatype(*buf)), dest_source(dest_source), tag(tag), comm(comm){}

		void *buf;
		const int count;
		const MPI_Datatype datatype;
		const int dest_source;
		const int tag;
		const MPI_Comm comm;
	};

	struct mpi_isend_task : mpi_async_task_base
	{
		mpi_isend_task(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) :
			mpi_async_task_base(const_cast<void*>(buf), count, datatype, dest, tag, comm){}

		template<typename T>
		mpi_isend_task(const T *buf, int count, int dest, int tag, MPI_Comm comm) :
			mpi_async_task_base(const_cast<T*>(buf), count, dest, tag, comm){}
	
		int operator()() const;
	};

	struct mpi_irecv_task : mpi_async_task_base
	{
		mpi_irecv_task(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm) :
			mpi_async_task_base(buf, count, datatype, source, tag, comm){}

		template<typename T>
		mpi_irecv_task(T *buf, int count, int source, int tag, MPI_Comm comm) :
			mpi_async_task_base(buf, count, source, tag, comm){}

		int operator()() const;
	};

} // namespace hpc
