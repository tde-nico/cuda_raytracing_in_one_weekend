#include "raytracer.cuh"
#include "camera.cuh"
#include "material.cuh"
#include <time.h>
#include <float.h>
#include <stdio.h>

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
		//if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec))
		if (O_hit((hittable_list *)*world, curr_ray, 0.001f, FLT_MAX, rec))
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

__global__ void	rand_init(curandState *rand_state)
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

	#if SHARED
		__shared__ vec3	share_sam[BLOCK_H][BLOCK_W];
	#endif


	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
	{
		#if SHARED
			share_sam[threadIdx.y][threadIdx.x] = vec3(-1,-1,-1);
		#endif
		return ;
	}
	i = W*y + x;

	state = rand_state[i];
	color = vec3(0, 0, 0);
	for (int s = 0; s < SAMPLES; ++s)
	{
		u = float(x + curand_uniform(&state)) / float(W);
		v = float(y + curand_uniform(&state)) / float(H);
		r = O_get_ray(*cam, u, v, &state);
		color += ray_color(r, world, &state);
	}
	color /= float(SAMPLES);

	#if SHARED
		vec3	sam;
		float	counter;

		share_sam[threadIdx.y][threadIdx.x] = color;
		__syncthreads();
		sam = vec3(0,0,0);
		counter = 0;

		if (threadIdx.x+1 < blockDim.x)
		{
			sam += share_sam[threadIdx.y][threadIdx.x+1];
			++counter;
		}
		if (threadIdx.y+1 < blockDim.y)
		{
			sam += share_sam[threadIdx.y+1][threadIdx.x];
			++counter;
		}
		if (threadIdx.x-1 < blockDim.x)
		{
			sam += share_sam[threadIdx.y][threadIdx.x-1];
			++counter;
		}
		if (threadIdx.y-1 < blockDim.y)
		{
			sam += share_sam[threadIdx.y-1][threadIdx.x];
			++counter;
		}

		color = (1-WEIGHT) * color + WEIGHT * sam / counter;
	#endif

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


#define RND (curand_uniform(&local_rand_state))
__global__ void	create_world(hittable **d_list, hittable **d_world, camera **d_camera, curandState *rand_state)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;

	curandState local_rand_state = *rand_state;
	d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
	int i = 1;
	for(int a = -11; a < 11; ++a)
	{
		for(int b = -11; b < 11; ++b)
		{
			float choose_mat = RND;
			vec3 center(a+RND,0.2,b+RND);
			if(choose_mat < 0.8f)
				d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
			else if(choose_mat < 0.95f)
				d_list[i++] = new sphere(center, 0.2, new metal(
					vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
			else
				d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
		}
	}
	d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
	d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
	d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	*rand_state = local_rand_state;
	*d_world  = new hittable_list(d_list, 22*22+1+3);

	vec3 lookfrom(13,2,3);
	vec3 lookat(0,0,0);
	float dist_to_focus = 10.0; (lookfrom-lookat).length();
	float aperture = 0.1;
	*d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, ASPECT_RATIO, aperture, dist_to_focus);
}

__global__ void	free_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return ;
	for (int i = 0; i < 22*22+1+3; ++i)
	{
		delete ((sphere *)d_list[i])->mat;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int	main(void)
{
	//vec3			*h_buf;
	vec3			*d_buf;
	hittable_list	**d_list;
	hittable_list	**d_world;
	curandState		*d_rand_state;
	curandState		*d_rand_state2;
	camera			**d_camera;
	clock_t			start;
	clock_t			stop;

	std::cerr << "Rendering a " << W << "x" << H << " image with " << SAMPLES;
	std::cerr << " samples per pixel in " << BLOCK_W << "x" << BLOCK_H << " blocks.\n";

	CHECK(cudaMallocManaged((void **)&d_buf, BSIZE));
	//h_buf = (vec3 *)malloc(BSIZE);
	//CHECK(cudaMalloc((void **)&d_buf, BSIZE));
	//CHECK(cudaMemcpy(d_buf, h_buf, BSIZE, cudaMemcpyHostToDevice));

	CHECK(cudaMalloc((void **)&d_rand_state, PIXELS * sizeof(curandState)));
	CHECK(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));
	CHECK(cudaMalloc((void **)&d_list, (22*22+1+3)*sizeof(hittable *)));
	CHECK(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	CHECK(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	rand_init<<<1, 1>>>(d_rand_state2);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	create_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera, d_rand_state2);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	rand_init<<<blocks, threads>>>(d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	start = clock();
	render<<<blocks, threads>>>(d_buf, d_camera, (hittable **)d_world, d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "Took: " << ((double)(stop - start)) / CLOCKS_PER_SEC << "\n";

	//CHECK(cudaMemcpy(h_buf, d_buf, BSIZE, cudaMemcpyDeviceToHost));

	print(d_buf);
	//print(h_buf);

	free_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_camera));
	CHECK(cudaFree(d_list));
	CHECK(cudaFree(d_world));
	CHECK(cudaFree(d_rand_state));
	CHECK(cudaFree(d_rand_state2));
	CHECK(cudaFree(d_buf));
	//free(h_buf);

	return (0);
}

