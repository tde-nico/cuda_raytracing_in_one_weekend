#ifndef UTILS_HPP
# define UTILS_HPP

# include <iostream>
# include <cuda.h>

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
# define CHECK(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t res, const char *func, const char *file, const int line)
{
	if (!res)
		return ;
	std::cerr << "CUDA error = " << static_cast<unsigned int>(res);
	std::cerr << " at " << file << ":" << line << " '" << func << "' \n";
	cudaDeviceReset();
	exit(1);
}



#endif
