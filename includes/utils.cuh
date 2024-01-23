#ifndef UTILS_CUH
# define UTILS_CUH

# include <iostream>
# include <cuda.h>
# include <curand_kernel.h>
# include <cmath>

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


# define PI 3.1415926535897932385f
inline float	degrees_to_radiants(float degrees)
{
	return (degrees * PI / 180.0f);
}


#endif
