#pragma once

#ifdef HPC_MPI_HPP
#error HPC_MPI_HPP is already defined
#endif
#define HPC_MPI_HPP

#if !defined(MPI_INCLUDED) && !defined(OMPI_MPI_H)
typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;

struct MPI_Status {
	int count;
	int cancelled;
	int MPI_SOURCE;
	int MPI_TAG;
	int MPI_ERROR;
};
#endif // MPI_INCLUDED

namespace hpc { namespace mpi {

	class communicator;

	namespace detail {

		extern MPI_Comm _MPI_COMM_WORLD, _MPI_COMM_SELF;
		extern MPI_Datatype _MPI_CHAR, _MPI_SHORT, _MPI_INT, _MPI_LONG, _MPI_LONG_LONG,
			_MPI_FLOAT, _MPI_DOUBLE, _MPI_LONG_DOUBLE,
			_MPI_UNSIGNED_CHAR, _MPI_UNSIGNED_SHORT, _MPI_UNSIGNED, _MPI_UNSIGNED_LONG, _MPI_UNSIGNED_LONG_LONG;
		extern MPI_Op _MPI_MAX, _MPI_MIN, _MPI_SUM, _MPI_PROD;

		void send(const communicator& comm, const void *buf, int count, MPI_Datatype datatype, int dest, int tag);
		void recv(const communicator& comm, void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Status *status);
		MPI_Request isend(const communicator& comm, const void *buf, int count, MPI_Datatype datatype, int dest, int tag);
		MPI_Request irecv(const communicator& comm, void *buf, int count, MPI_Datatype datatype, int source, int tag);
		void reduce(const communicator& comm, void *in_values, void *out_values, int n, MPI_Datatype type, MPI_Op op, int root);
		void all_reduce(const communicator& comm, void *in_values, void *out_values, int n, MPI_Datatype type, MPI_Op op);
		void all_to_all_v(const communicator& comm, const void *sendbuf, const int *sendcounts, const int *senddisps, MPI_Datatype sendtype,
			void *recvbuf, const int *recvcounts, const int *recvdisps, MPI_Datatype recvtype);
		void sendrecv(const communicator& comm, const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
			void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Status *status);
		void broadcast(const communicator& comm, void *buffer, int count, MPI_Datatype datatype, int root);
		void wait_all(int count, MPI_Request array_of_requests[]);

	} // namespace detail

	template<typename T>
	struct mpi_datatype;

#define HPC_MPI_DATATYPE(CppType, MPIType) \
	template<> \
	struct mpi_datatype<CppType> \
		{ \
			static MPI_Datatype get_value(){ \
				return detail::_##MPIType; \
			} \
		}

	HPC_MPI_DATATYPE(char, MPI_CHAR);
	HPC_MPI_DATATYPE(short, MPI_SHORT);
	HPC_MPI_DATATYPE(int, MPI_INT);
	HPC_MPI_DATATYPE(long, MPI_LONG);
	HPC_MPI_DATATYPE(long long, MPI_LONG_LONG);
	HPC_MPI_DATATYPE(float, MPI_FLOAT);
	HPC_MPI_DATATYPE(double, MPI_DOUBLE);
	HPC_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);
	HPC_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
	HPC_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);
	HPC_MPI_DATATYPE(unsigned, MPI_UNSIGNED);
	HPC_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);
	HPC_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

#undef HPC_MPI_DATATYPE

	template<typename T>
	MPI_Datatype get_mpi_datatype(){
		return	mpi_datatype<T>::get_value();
	}

	template<typename T>
	struct maximum : public std::binary_function<T, T, T>
	{
#ifdef __CUDACC__
		__device__ __host__
#endif
		const T& operator()(const T& x, const T& y) const {
			return x < y ? y : x;
		}
	};

	template<typename T>
	struct minimum : public std::binary_function<T, T, T>
	{
#ifdef __CUDACC__
		__device__ __host__
#endif
		const T& operator()(const T& x, const T& y) const {
			return x < y ? x : y;
		}
	};

	template<typename T, typename Op>
	struct mpi_op;

	template<typename T>
	struct mpi_op<T, maximum<T> >
	{
		static MPI_Op get_value(){
			return detail::_MPI_MAX;
		}
	};

	template<typename T>
	struct mpi_op<T, minimum<T> >
	{
		static MPI_Op get_value(){
			return detail::_MPI_MIN;
		}
	};

	template<typename T>
	struct mpi_op<T, std::plus<T> >
	{
		static MPI_Op get_value(){
			return detail::_MPI_SUM;
		}
	};

	template<typename T>
	struct mpi_op<T, std::multiplies<T> >
	{
		static MPI_Op get_value(){
			return detail::_MPI_PROD;
		}
	};

	template<typename T, typename Op>
	MPI_Op get_mpi_op(){
		return mpi_op<T, Op>::get_value();
	}

	class mpi_exception : public std::exception
	{
	public:
		typedef std::exception super;

		mpi_exception(const char *func_name, int error);
		~mpi_exception() throw();
		virtual const char *what() const throw();
		int error() const { return m_error; }

	private:
		int m_error;
		std::string m_msg;
	};

	class timer
	{
	public:
		timer();
		void restart();
		double elapsed() const;
		double elapsed_max() const;
		double elapsed_min() const;

		static bool time_is_global();

	private:
		double start_time;
	};

	class request
	{
	public:
		request(MPI_Request request_ = 0) : m_request(request_){}

		operator MPI_Request&(){ return m_request; }

	private:
		MPI_Request m_request;
	};

	class status
	{
	public:
		status(MPI_Status& status) : m_status(status){}

		operator MPI_Status&(){ return m_status; }
		operator const MPI_Status&() const { return m_status; }

	private:
		MPI_Status m_status;
	};

	class environment
	{
	public:
		environment(int argc, char* argv[]);
		~environment();
	};

	enum comm_create_kind { comm_duplicate, comm_take_ownership, comm_attach };

	class communicator
	{
	public:
		communicator();
		communicator(MPI_Comm comm, comm_create_kind kind);
		operator MPI_Comm() const { return m_comm; }

		int size() const;
		int rank() const;
		void barrier() const;
		void abort(int errcode) const;

		template<typename T>
		void send(int dest, int tag, const T *buf, int count) const {
			return detail::send(*this, buf, count, get_mpi_datatype<T>(), dest, tag);
		}

		template<typename T>
		void send(int dest, int tag, const T &value) const {
			send(dest, tag, &value, 1);
		}

		template<typename T>
		status recv(int source, int tag, T *buf, int count) const
		{
			MPI_Status status;
			detail::recv(*this, buf, count, get_mpi_datatype<T>(), source, tag, &status);
			return status;
		}

		template<typename T>
		status recv(int source, int tag, T &value) const {
			return recv(source, tag, &value, 1);
		}

		template<typename T>
		request isend(int dest, int tag, const T *buf, int count) const {
			return detail::isend(*this, buf, count, get_mpi_datatype<T>(), dest, tag);
		}

		template<typename T>
		request isend(int dest, int tag, const T &value) const {
			return isend(dest, tag, &value, 1);
		}

		template<typename T>
		request irecv(int source, int tag, T *buf, int count) const {
			return detail::irecv(*this, buf, count, get_mpi_datatype<T>(), source, tag);
		}

		template<typename T>
		request irecv(int source, int tag, T &value) const {
			return irecv(source, tag, &value, 1);
		}

	private:
		MPI_Comm m_comm;
	};

	template<typename T, typename Op>
	void reduce(const communicator &comm, const T &in_value, T &out_value, Op op, int root){
		detail::reduce(comm, const_cast<T*>(&in_value), &out_value, 1, get_mpi_datatype<T>(), get_mpi_op<T, Op>(), root);
	}

	template<typename T, typename Op>
	void reduce(const communicator &comm, const T &in_value, Op op, int root){
		detail::reduce(comm, const_cast<T*>(&in_value), 0, 1, get_mpi_datatype<T>(), get_mpi_op<T, Op>(), root);
	}

	template<typename T, typename Op>
	void reduce(const communicator& comm, const T* in_values, int n, T* out_values, Op op, int root){
		detail::reduce(comm, const_cast<T*>(in_values), out_values, n, get_mpi_datatype<T>(), get_mpi_op<T, Op>(), root);
	}

	template<typename T, typename Op>
	void all_reduce(const communicator &comm, const T &in_value, T &out_value, Op op)
	{
		detail::all_reduce(comm, const_cast<T*>(&in_value), &out_value, 1, get_mpi_datatype<T>(), get_mpi_op<T, Op>());
	}

	template<typename T, typename Op>
	T all_reduce(const communicator &comm, const T &in_value, Op op)
	{
		T result;
		all_reduce(comm, in_value, result, op);
		return result;
	}

	template<typename T1, typename T2>
	void all_to_all_v(const communicator& comm, const T1 *sendbuf, const int *sendcounts, const int *senddisps,
		T2 *recvbuf, const int *recvcounts, const int *recvdisps)
	{
		detail::all_to_all_v(comm, sendbuf, sendcounts, senddisps, get_mpi_datatype<T1>(),
			recvbuf, recvcounts, recvdisps, get_mpi_datatype<T2>());
	}

	template<typename T1, typename T2>
	status sendrecv(const communicator& comm, const T1 *sendbuf, int sendcount, int dest, int sendtag,
		T2 *recvbuf, int recvcount, int source, int recvtag)
	{
		MPI_Status status;
		detail::sendrecv(comm, sendbuf, sendcount, get_mpi_datatype<T1>(), dest, sendtag,
			recvbuf, recvcount, get_mpi_datatype<T2>(), source, recvtag, &status);
		return status;
	}

	template<typename T>
	void broadcast(const communicator& comm, T &value, int root){
		detail::broadcast(comm, &value, 1, get_mpi_datatype<T>(), root);
	}

	template<typename T>
	void broadcast(const communicator& comm, T *values, int n, int root){
		detail::broadcast(comm, values, n, get_mpi_datatype<T>(), root);
	}

	void wait_all(request *begin, request *end);

} }	// namespace
