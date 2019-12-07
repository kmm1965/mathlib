#pragma once

#ifndef CL_HPP_
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

namespace hpc {

const char *getOpenCLErrorMsg(cl_int errorCode);

int getOpenCLDeviceCount();
int OpenCLDeviceInfo();

size_t getOpenCLGlobalMemSize(std::size_t deviceNum);
int getOpenCLMaxComputeUnit(std::size_t deviceNum);
int getOpenCLMaxWorkGroupSize(std::size_t deviceNum);

}	// namespace hpc
