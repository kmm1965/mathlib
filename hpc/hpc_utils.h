#pragma once

namespace hpc {

std::string readFile(const char *fileName);
size_t shrRoundUp(size_t group_size, size_t global_size);

}	// namespace hpc
