#pragma once

#ifndef COMM_ABORT
#if defined(BOOST_MPI_HPP) || defined(HPC_MPI_HPP)
#define COMM_ABORT() comm.abort(-1)
#else
#define COMM_ABORT()
#endif
#endif

#ifndef DEMANGLE
#define DEMANGLE(name) name
#endif

#ifdef BOOST_SYSTEM_SYSTEM_ERROR_HPP
#define CATCH_BOOST_SYSTEM_ERROR() \
	} catch (const boost::system::system_error &e){ \
		std::cerr << "System error " << DEMANGLE(typeid(e).name()) << std::endl << e.what() \
		<< ", error code=" << e.code() << std::endl; \
		COMM_ABORT();
#else
#define CATCH_BOOST_SYSTEM_ERROR()
#endif

#define CATCH_EXCEPTIONS() \
	catch (const std::bad_alloc&){ \
		std::cout << std::endl; \
		std::cerr << "Insufficient memory" << std::endl; \
		COMM_ABORT(); \
		return -1; \
	CATCH_BOOST_SYSTEM_ERROR() \
	} catch (const std::exception &e){ \
		std::cout << std::endl; \
		std::cerr << "Exception " << DEMANGLE(typeid(e).name()) << std::endl << e.what() << std::endl; \
		COMM_ABORT(); \
		return -1; \
	} catch (...){ \
		std::cout << std::endl; \
		std::cerr << "Unknown exception" << std::endl; \
		COMM_ABORT(); \
		return -1; \
	}
