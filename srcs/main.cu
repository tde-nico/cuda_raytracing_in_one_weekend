#include "raytracer.hpp"

__device__ vec3	ray_color(const ray &r)
{
	vec3	unit_direction;
	float	t;

	unit_direction = unit_vector(r.direction());
	t = 0.5f * (unit_direction.y() + 1.0f);
	return (vec3(1.0, 1.0, 1.0) * (1.0f-t) + vec3(0.5, 0.7, 1.0) * t);
}

__global__ void	render(vec3 *buf, vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical)
{
	int		x;
	int		y;
	int		i;
	float	u;
	float	v;
	ray		r;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
		return ;
	i = W*y + x;
	u = float(x) / float(W);
	v = float(y) / float(H);
	r = ray(origin, lower_left_corner + horizontal * u + vertical * v);

	buf[i] = ray_color(r);
}

void	write_color(std::ostream &out, vec3 pixel)
{
	out << int(255.99 * pixel.r()) << ' '
		<< int(255.99 * pixel.g()) << ' '
		<< int(255.99 * pixel.b()) << '\n';
}

void	print(vec3 *buf)
{
	std::cout << "P3\n" << W << " " << H << "\n255\n";
	for (int y = H-1; y >= 0; --y)
	{
		for (int x = 0; x < W; ++x)
			write_color(std::cout, buf[W*y + x]);
	}
}

/*
https://raytracing.github.io/v3/books/RayTracingInOneWeekend.html
https://github.com/RayTracing/raytracing.github.io/
https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
https://github.com/rogerallen/raytracinginoneweekendincuda?tab=readme-ov-file
*/

int	main(void)
{
	vec3	*buf;

	CHECK(cudaMallocManaged((void **)&buf, BSIZE));

	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	render<<<blocks, threads>>>(buf, ORIGIN, LOWER_LEFT_CORNER, HORIZONTAL, VERTICAL);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	print(buf);

	CHECK(cudaFree(buf));

	return (0);
}

