#include "raytracer.hpp"

__global__ void	render(float *buf)
{
	int	x;
	int	y;
	int	i;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
		return ;
	i = W*y*3 + x*3;
	buf[i] = float(x) / W;
	buf[i + 1] = float(y) / H;
	buf[i + 2] = 0.2;
}

void	print(float *buf)
{
	size_t	index;
	float	r;
	float	g;
	float	b;

	std::cout << "P3\n" << W << " " << H << "\n255\n";
	for (int y = H-1; y >= 0; --y)
	{
		for (int x = 0; x < W; ++x)
		{
			index = W*y*3 + x*3;
			r = int(255.99 * buf[index]);
			g = int(255.99 * buf[index + 1]);
			b = int(255.99 * buf[index + 2]);
			std::cout << r << " " << g << " " << b << "\n";
		}
	}
}

int	main(void)
{
	float	*buf;

	CHECK(cudaMallocManaged((void **)&buf, BSIZE));
	
	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	render<<<blocks, threads>>>(buf);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	print(buf);

	CHECK(cudaFree(buf));

	return (0);
}

