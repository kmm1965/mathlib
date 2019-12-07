#pragma once

#include "mpi.hpp"

#if !defined(MPI_INCLUDED) && !defined(OMPI_MPI_H)
typedef void *MPI_File;
typedef int MPI_Info;
typedef int MPI_Errhandler;
typedef long long MPI_Offset;
#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)
#endif

namespace hpc { namespace mpi {

	namespace detail {

		void file_open(const communicator& comm, const char *filename, int amode, MPI_Info info, MPI_File &fh);
		void file_close(MPI_File &fh);
		void file_preallocate(MPI_File fh, MPI_Offset size);
		void file_seek(MPI_File fh, MPI_Offset offset, int whence);
		void file_seek_shared(MPI_File fh, MPI_Offset offset, int whence);
		void file_read(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_read_all(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_read_all_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
		void file_read_all_end(MPI_File fh, void *buf, MPI_Status *status);
		void file_read_at(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_read_at_all(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_read_at_all_begin(MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype);
		void file_read_at_all_end(MPI_File fh, void *buf, MPI_Status *status);
		void file_read_ordered(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_read_ordered_begin(MPI_File fh, void *buf, int count, MPI_Datatype datatype);
		void file_read_ordered_end(MPI_File fh, void *buf, MPI_Status *status);
		void file_read_shared(MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write_all(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write_all_begin(MPI_File fh, const void *buf, int count, MPI_Datatype datatype);
		void file_write_all_end(MPI_File fh, const void *buf, MPI_Status *status);
		void file_write_at(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write_at_all_begin(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype);
		void file_write_at_all_end(MPI_File fh, const void *buf, MPI_Status *status);
		void file_write_ordered(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_write_ordered_begin(MPI_File fh, const void *buf, int count, MPI_Datatype datatype);
		void file_write_ordered_end(MPI_File fh, const void *buf, MPI_Status *status);
		void file_write_shared(MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
		void file_set_atomicity(MPI_File fh, int flag);
		void file_set_errhandler(MPI_File fh, MPI_Errhandler errhandler);
		void file_set_info(MPI_File fh, MPI_Info info);
		void file_set_size(MPI_File fh, MPI_Offset size);
		void file_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info);
		void file_sync(MPI_File fh);

	} // detail

	class file
	{
		file() : fh(0){}

		file(const communicator& comm, const char *filename, int amode, MPI_Info info = MPI_INFO_NULL) : fh(0){
			detail::file_open(comm, filename, amode, info, fh);
		}

		~file(){
			close();
		}

		void open(const communicator& comm, const char *filename, int amode, MPI_Info info = MPI_INFO_NULL){
			detail::file_open(comm, filename, amode, info, fh);
		}

		void close()
		{
			if (fh){
				detail::file_close(fh);
				fh = 0;
			}
		}

		void preallocate(MPI_Offset size) const {
			detail::file_preallocate(fh, size);
		}

		void seek(MPI_Offset offset, int whence) const {
			detail::file_seek(fh, offset, whence);
		}

		void seek_shared(MPI_Offset offset, int whence) const {
			detail::file_seek_shared(fh, offset, whence);
		}

		template<typename T>
		MPI_Status read(T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		MPI_Status read_all(T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read_all(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void read_all_begin(T *buf, int count) const {
			detail::file_read_all_begin(fh, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status read_all_end(void *buf) const
		{
			MPI_Status status;
			detail::file_read_all_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status read_at(MPI_Offset offset, T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read_at(fh, offset, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		MPI_Status read_at_all(MPI_Offset offset, T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read_at_all(fh, offset, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void read_at_all_begin(MPI_Offset offset, T *buf, int count) const {
			detail::file_read_at_all_begin(fh, offset, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status read_at_all_end(void *buf) const
		{
			MPI_Status status;
			detail::file_read_at_all_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status read_ordered(T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read_ordered(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void read_ordered_begin(T *buf, int count) const {
			detail::file_read_ordered_begin(fh, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status read_ordered_end(void *buf) const
		{
			MPI_Status status;
			detail::file_read_ordered_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status read_shared(T *buf, int count) const
		{
			MPI_Status status;
			detail::file_read_shared(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		MPI_Status write(const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		MPI_Status write_all(const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write_all(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void write_all_begin(const T *buf, int count) const {
			detail::file_write_all_begin(fh, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status write_all_end(const void *buf) const
		{
			MPI_Status status;
			detail::file_write_all_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status write_at(MPI_Offset offset, const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write_at(fh, offset, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		MPI_Status write_at_all(MPI_Offset offset, const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write_at_all(fh, offset, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void write_at_all_begin(MPI_Offset offset, const T *buf, int count) const {
			detail::file_write_at_all_begin(fh, offset, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status write_at_all_end(const void *buf) const
		{
			MPI_Status status;
			detail::file_write_at_all_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status write_ordered(const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write_ordered(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		template<typename T>
		void write_ordered_begin(const T *buf, int count) const {
			detail::file_write_ordered_begin(fh, buf, count, get_mpi_datatype<T>());
		}

		MPI_Status write_ordered_end(const void *buf) const
		{
			MPI_Status status;
			detail::file_write_ordered_end(fh, buf, &status);
			return status;
		}

		template<typename T>
		MPI_Status write_shared(const T *buf, int count) const
		{
			MPI_Status status;
			detail::file_write_shared(fh, buf, count, get_mpi_datatype<T>(), &status);
			return status;
		}

		void set_atomicity(int flag) const {
			detail::file_set_atomicity(fh, flag);
		}

		void set_errhandler(MPI_Errhandler errhandler) const {
			detail::file_set_errhandler(fh, errhandler);
		}

		void set_info(MPI_Info info) const {
			detail::file_set_info(fh, info);
		}

		void set_size(MPI_Offset size) const {
			detail::file_set_size(fh, size);
		}

		void set_view(MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info) const {
			detail::file_set_view(fh, disp, etype, filetype, datarep, info);
		}

		void sync() const {
			detail::file_sync(fh);
		}

		void test(const communicator& comm) const
		{
			const int count = 5;
			double buf[count];

			file f;
			f.open(comm, "test", 0);
			f.close();
			f.preallocate(0);
			f.seek(0, 0);
			f.seek_shared(0, 0);
			f.read(buf, count);
			f.read_all(buf, count);
			f.read_all_begin(buf, count);
			f.read_all_end(buf);
			f.read_at(0, buf, count);
			f.read_at_all(0, buf, count);
			f.read_at_all_begin(0, buf, count);
			f.read_at_all_end(buf);
			f.read_ordered(buf, count);
			f.read_ordered_begin(buf, count);
			f.read_ordered_end(buf);
			f.read_shared(buf, count);
			f.write(buf, count);
			f.write_all(buf, count);
			f.write_all_begin(buf, count);
			f.write_all_end(buf);
			f.write_at(0, buf, count);
			f.write_at_all(0, buf, count);
			f.write_at_all_begin(0, buf, count);
			f.write_at_all_end(buf);
			f.write_ordered(buf, count);
			f.write_ordered_begin(buf, count);
			f.write_ordered_end(buf);
			f.write_shared(buf, count);
			f.set_atomicity(0);
			f.set_errhandler(0);
			f.set_info(0);
			f.set_size(0);
			f.set_view(0, 0, 0, "test", 0);
			f.sync();
		}

	private:
		MPI_File fh;
	};

} } // end namespace hpc::mpi
