#pragma once

namespace hpc {

class task_error : public std::runtime_error
{
public:
	task_error(const char *msg);
};

class task
{
protected:
	task();
	virtual ~task();

public:
	virtual bool execTask() = 0;

protected:
	virtual bool init();
	virtual void term();
	virtual void report();

public:
	boost::timer::cpu_times m_total, m_work;
};

} // namespace hpc
