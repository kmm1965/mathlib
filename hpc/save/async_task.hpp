#pragma once

#include <boost/thread/future.hpp>

#include <queue>

namespace hpc
{

	struct async_task_base_shared_state : boost::detail::task_base_shared_state<void>
	{
		virtual ~async_task_base_shared_state(){}
		virtual int exec() = 0;
	};

	template<typename F>
	struct async_task_shared_state : async_task_base_shared_state
	{
		typedef async_task_base_shared_state super;

	private:
		async_task_shared_state(async_task_shared_state&);

	public:
		async_task_shared_state(const F &f) : f(f){}

		F& callable(){ return f; }

		virtual int exec(){
			return f();
		}

		virtual void do_run()
		{
			try {
				f();
			} catch (std::exception &e){
				std::cerr << "Exception in async_task_shared_state::do_run: " << boost::core::demangle(typeid(e).name()) << std::endl << e.what() << std::endl;
				super::mark_exceptional_finish();
			} catch (...){
				std::cerr << "Exception in async_task_shared_state::do_run" << std::endl;
				super::mark_exceptional_finish();
			}
		}

		virtual void do_apply()
		{
			assert(false);
			BOOST_THROW_EXCEPTION(std::runtime_error("Not implemented"));
		}

	private:
		F f;
	};

	struct async_task_base0
	{
		virtual bool is_ready() const = 0;
		virtual void run() = 0;
		virtual int exec() = 0;
		virtual void mark_finished_with_result() = 0;
		virtual void mark_exceptional_finish() = 0;
	};

	template<class AT>
	struct async_task_base : async_task_base0
	{
		typedef boost::BOOST_THREAD_FUTURE<void> future_type;
		typedef boost::shared_ptr<async_task_base_shared_state> task_ptr;

		template<class F>
		explicit async_task_base(const F &f, future_type *parent_future = 0) :
			future_obtained(false), parent_future(parent_future)
		{
			task = task_ptr(AT::create_shared_state(f));
		}

		~async_task_base()
		{
			if (task)
				task->owner_destroyed();
		}

		bool valid() const BOOST_NOEXCEPT {
			return task.get() != 0;
		}

		// result retrieval
		future_type get_future()
		{
			if (!task)
				BOOST_THROW_EXCEPTION(boost::task_moved());
			else if (future_obtained)
				BOOST_THROW_EXCEPTION(boost::future_already_retrieved());
			else {
				future_obtained = true;
				return future_type(task);
			}
		}

		virtual bool is_ready() const {
			return !parent_future || parent_future->is_ready();
		}

		virtual void run()
		{
			if (!task)
				BOOST_THROW_EXCEPTION(boost::task_moved());
			if (!is_ready())
				BOOST_THROW_EXCEPTION(std::runtime_error("The task is not ready yet"));
			task->run();
		}

		virtual int exec()
		{
			if (!task)
				BOOST_THROW_EXCEPTION(boost::task_moved());
			if (!is_ready())
				BOOST_THROW_EXCEPTION(std::runtime_error("The task is not ready yet"));
			return task->exec();
		}

		virtual void mark_finished_with_result()
		{
			if (!task)
				BOOST_THROW_EXCEPTION(boost::task_moved());
			task->mark_finished_with_result();
		}

		virtual void mark_exceptional_finish()
		{
			if (!task)
				BOOST_THROW_EXCEPTION(boost::task_moved());
			task->mark_exceptional_finish();
		}

		template<typename F>
		void set_wait_callback(F f){
			task->set_wait_callback(f, this);
		}

	private:
		task_ptr task;
		bool future_obtained;
		future_type *parent_future;
	};

	struct async_task : async_task_base<async_task>
	{
		typedef async_task_base<async_task> super;

		template<class F>
		explicit async_task(const F &f, typename super::future_type *future = 0) : super(f, future){}

		template<class F>
		static async_task_base_shared_state *create_shared_state(const F &f){
			return new async_task_shared_state<F>(f);
		}
	};

	class async_task_queue
	{
	protected:
		typedef async_task_base0 task_type;

	public:
		virtual bool empty() const;
		virtual bool exec() = 0;

		template<typename AT>
		void submit(async_task_base<AT> &async_task)
		{
			m_tasks.push(&async_task);
		}

	protected:
		std::queue<task_type*> m_tasks;
	};

} // namespace hpc
