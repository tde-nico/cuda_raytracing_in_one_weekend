#ifndef UTILS_CUH
# define UTILS_CUH

# include <iostream>
# include <cuda.h>
# include <curand_kernel.h>
# include <cmath>

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
# define CHECK(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/**
 * @brief A function to check for errors
 * @param res the cudaError result of the checked function
 * @param func the function that was called
 * @param file the string of the origin file
 * @param line the number of line where the error occurred
 * 
 * This function checks the result of a called function and throws an
 * error waring with detailed info and stops the execution
*/
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
