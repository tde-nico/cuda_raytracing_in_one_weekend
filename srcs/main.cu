#include "raytracer.hpp"
#include <time.h>
#include <float.h>


__device__ vec3	ray_color(const ray &r, hittable **world)
{
	t_hit_record	rec;
	float			t;
	vec3			unit_direction;

	if ((*world)->hit(r, 0.0f, FLT_MAX, rec))
		return ((rec.normal + vec3(1,1,1)) * 0.5f);
	unit_direction = unit_vector(r.direction());
	t = 0.5f * (unit_direction.y() + 1.0f);
	return (vec3(1.0, 1.0, 1.0) * (1.0f-t) + vec3(0.5, 0.7, 1.0) * t);
}

__global__ void	render(vec3 *buf, vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, hittable **world)
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

	buf[i] = ray_color(r, world);
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


__global__ void	create_world(hittable **d_list, hittable **d_world)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;
	d_list[0] = new sphere(vec3(0,0,-1), 0.5);
	d_list[1] = new sphere(vec3(0,-100.5,-1), 100);
	*d_world = new hittable_list(d_list, 2);
}

__global__ void	free_world(hittable **d_list, hittable **d_world)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;
	delete d_list[0];
	delete d_list[1];
	delete *d_world;
}

int	main(void)
{
	vec3			*buf;
	hittable_list	**d_list;
	hittable_list	**d_world;
	clock_t			start;
	clock_t			stop;

	CHECK(cudaMallocManaged((void **)&buf, BSIZE));
	CHECK(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
	CHECK(cudaMalloc((void **)&d_world, sizeof(hittable *)));

	create_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	start = clock();
	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	render<<<blocks, threads>>>(buf, ORIGIN, LOWER_LEFT_CORNER, HORIZONTAL, VERTICAL, (hittable **)d_world);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "Took: " << (stop - start) / CLOCKS_PER_SEC << "\n";

	print(buf);

	free_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_list));
	CHECK(cudaFree(d_world));
	CHECK(cudaFree(buf));

	return (0);
}

