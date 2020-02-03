#include "pch.h"
#include "../gridop2_src/gridop2_dim_cpp.h"
#include "../gridop2_src/catch.h"

int main(int argc, char* argv[])
{
    try {
        const int cuda_devices = hpc::getCudaDeviceCount();
        std::cout << "Number of CUDA devices: " << cuda_devices << std::endl;
        if (cuda_devices < 1) {
            std::cerr << "No CUDA devices found" << std::endl;
            return 1;
        }
        cudaError_t error_id;
        cudaDeviceProp deviceProp;
        CUDA_EXEC(cudaGetDeviceProperties, (&deviceProp, 0));
        std::cout
            << "Device: " << deviceProp.name << " (" << deviceProp.major << '.' << deviceProp.minor << ')' << std::endl
            << "Total global memory: " << deviceProp.totalGlobalMem << std::endl
            << "Free global memory: " << hpc::getCudaFreeMem() << std::endl
            << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl
            << "Multi Processor count: " << deviceProp.multiProcessorCount << std::endl;
        return main1(argc, argv);
    } CATCH_EXCEPTIONS()
}
