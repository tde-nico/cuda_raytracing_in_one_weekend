#include "raytracer.hpp"
#include "camera.hpp"
#include "material.hpp"
#include <time.h>
#include <float.h>


__device__ vec3	ray_color(const ray &r, hittable **world, curandState *rand_state)
{
	ray				curr_ray;
	vec3			att;
	t_hit_record	rec;
	float			t;

	curr_ray = r;
	att = vec3(1, 1, 1);
	for (int i = 0; i < REFRACTION; ++i)
	{
		if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec))
		{
			ray		scattered;
			vec3	attenuation;
			if (rec.mat->scatter(curr_ray, rec, attenuation, scattered, rand_state))
			{
				att *= attenuation;
				curr_ray = scattered;
			}
			else
				return (vec3(0, 0, 0));
		}
		else
		{
			vec3 unit_direction = unit_vector(curr_ray.direction());
			t = 0.5f * (unit_direction.y() + 1.0f);
			return ((vec3(1,1,1)*(1.0f-t) + vec3(0.5,0.7,1.0)*t) * att);
		}
	}
	return (vec3(0, 0, 0));
}

__global__ void	render_init(curandState *rand_state)
{
	int		x;
	int		y;
	int		i;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
		return ;
	i = W*y + x;

	curand_init(SEED, i, 0, &rand_state[i]);
}

__global__ void	render(vec3 *buf, camera **cam, hittable **world, curandState *rand_state)
{
	int			x;
	int			y;
	int			i;
	curandState	state;
	vec3		color;
	float		u;
	float		v;
	ray			r;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
		return ;
	i = W*y + x;

	state = rand_state[i];
	color = vec3(0, 0, 0);
	for (int s = 0; s < SAMPLES; ++s)
	{
		u = float(x + curand_uniform(&state)) / float(W);
		v = float(y + curand_uniform(&state)) / float(H);
		r = (*cam)->get_ray(u, v, &state);
		color += ray_color(r, world, &state);
	}
	rand_state[i] = state;
	color /= float(SAMPLES);
	color[0] = std::sqrt(color[0]);
	color[1] = std::sqrt(color[1]);
	color[2] = std::sqrt(color[2]);
	buf[i] = color;
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


__global__ void	create_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;
	d_list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
	d_list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
	d_list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
	d_list[3] = new sphere(vec3(-1,0,-1), 0.5, new dielectric(1.5));
	d_list[4] = new sphere(vec3(-1,0,-1), -0.45, new dielectric(1.5));
	*d_world = new hittable_list(d_list, 5);
	vec3	lookfrom(3, 3, 2);
	vec3	lookat(0, 0, -1);
	float	dist_to_focus = (lookfrom - lookat).length();
	float	aperture = 2.0f;
	*d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 20.0, ASPECT_RATIO, aperture, dist_to_focus);
}

__global__ void	free_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;
	for (int i = 0; i < 5; ++i)
	{
		delete ((sphere *)d_list[i])->mat;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int	main(void)
{
	vec3			*buf;
	hittable_list	**d_list;
	hittable_list	**d_world;
	curandState		*d_rand_state;
	camera			**d_camera;
	clock_t			start;
	clock_t			stop;

	std::cerr << "Rendering a " << W << "x" << H << " image with " << SAMPLES;
	std::cerr << " samples per pixel in " << BLOCK_W << "x" << BLOCK_H << " blocks.\n";

	CHECK(cudaMallocManaged((void **)&buf, BSIZE));
	CHECK(cudaMalloc((void **)&d_rand_state, PIXELS * sizeof(curandState)));
	CHECK(cudaMalloc((void **)&d_list, 5*sizeof(hittable *)));
	CHECK(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	CHECK(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	start = clock();
	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	render_init<<<blocks, threads>>>(d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	render<<<blocks, threads>>>(buf, d_camera, (hittable **)d_world, d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "Took: " << ((double)(stop - start)) / CLOCKS_PER_SEC << "\n";

	print(buf);

	free_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_camera));
	CHECK(cudaFree(d_list));
	CHECK(cudaFree(d_world));
	CHECK(cudaFree(d_rand_state));
	CHECK(cudaFree(buf));

	return (0);
}

