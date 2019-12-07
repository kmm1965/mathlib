#pragma once

#ifndef BOOST_MPI_HPP
#error <boost/mpi.h> should be included first
#endif

namespace boost { namespace mpi {

	namespace detail {

		template<typename T1, typename T2>
		void all_to_all_v_impl(const communicator& comm, const T1 *sendbuf, const int *sendcounts, const int *senddisps,
			T2 *recvbuf, const int *recvcounts, const int *recvdisps)
		{
			MPI_Datatype sendtype = get_mpi_datatype(*sendbuf);
			MPI_Datatype recvtype = get_mpi_datatype(*recvbuf);
			BOOST_MPI_CHECK_RESULT(MPI_Alltoallv, (const_cast<T1*>(sendbuf),
				const_cast<int*>(sendcounts), const_cast<int*>(senddisps), sendtype,
				recvbuf, const_cast<int*>(recvcounts), const_cast<int*>(recvdisps), recvtype, comm));
		}

		template<typename T1, typename T2>
		status sendrecv_impl(const communicator& comm, const T1 *sendbuf, int sendcount, int dest, int sendtag,
			T2 *recvbuf, int recvcount, int source, int recvtag)
		{
			status stat;
			BOOST_MPI_CHECK_RESULT(MPI_Sendrecv,
				(sendbuf, sendcount, get_mpi_datatype(*sendbuf), dest, sendtag,
				recvbuf, recvcount, get_mpi_datatype(*recvbuf), source, recvtag, comm, &stat.m_status));
			return stat;
		}
	} // detail

	template<typename T1, typename T2>
	void all_to_all_v(const communicator& comm, const T1 *sendbuf, const int *sendcounts, const int *senddisps,
		T2 *recvbuf, const int *recvcounts, const int *recvdisps)
	{
		detail::all_to_all_v_impl(comm, sendbuf, sendcounts, senddisps, recvbuf, recvcounts, recvdisps);
	}

	template<typename T1, typename T2>
	status sendrecv(const communicator& comm, const T1 *sendbuf, int sendcount, int dest, int sendtag,
		T2 *recvbuf, int recvcount, int source, int recvtag)
	{
		return detail::sendrecv_impl(comm, sendbuf, sendcount, dest, sendtag, recvbuf, recvcount, source, recvtag);
	}

} } // end namespace boost::mpi
