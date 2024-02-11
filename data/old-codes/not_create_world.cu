#include "raytracer.cuh"
#include "camera.cuh"
#include "material.cuh"
#include <time.h>
#include <float.h>


__device__ vec3	ray_color(const ray &r, hittable_list **world, curandState *rand_state)
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
		if (O_hit(*world, curr_ray, 0.001f, FLT_MAX, rec))
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

__global__ void	render(vec3 *buf, camera *cam, hittable_list **world, curandState *rand_state)
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
		//r = (*cam)->get_ray(u, v, &state);
		r = O_get_ray(cam, u, v, &state);
		color += ray_color(r, world, &state);
	}
	//rand_state[i] = state;
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




#define RND (curand_uniform(&local_rand_state))


int	main(void)
{
	//vec3			*h_buf;
	vec3			*d_buf;
	//hittable_list	**d_list;
	//hittable_list	**d_world;
	curandState		*d_rand_state;
	//curandState		*d_rand_state2;
	clock_t			start;
	clock_t			stop;

	std::cerr << "Rendering a " << W << "x" << H << " image with " << SAMPLES;
	std::cerr << " samples per pixel in " << BLOCK_W << "x" << BLOCK_H << " blocks.\n";

	CHECK(cudaMallocManaged((void **)&d_buf, BSIZE));
	//h_buf = (vec3 *)malloc(BSIZE);
	//CHECK(cudaMalloc((void **)&d_buf, BSIZE));
	//CHECK(cudaMemcpy(d_buf, h_buf, BSIZE, cudaMemcpyHostToDevice));



	// create
	/*
	curandState local_rand_state = *rand_state;
	d_list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
	d_list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
	*d_world = new hittable_list(d_list, 2);

	vec3	lookfrom(3, 3, 2);
	vec3	lookat(0, 0, -1);
	float	dist_to_focus = (lookfrom - lookat).length();
	float	aperture = 2.0f;
	*d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 20.0, ASPECT_RATIO, aperture, dist_to_focus);
	*/



	// list
	hittable	**d_list;
	hittable	**h_list;
	h_list = (hittable **)malloc(2*sizeof(hittable *));
	CHECK(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
	

	// new lambertian(vec3(0.8, 0.8, 0.0))
	lambertian	*h_mat;
	lambertian	*d_mat;
	h_mat = new lambertian(vec3(0.8, 0.8, 0.0));
	CHECK(cudaMalloc((void **)&d_mat, sizeof(lambertian)));
	CHECK(cudaMemcpy(d_mat, h_mat, sizeof(lambertian), cudaMemcpyHostToDevice));


	// s0
	sphere	*d_s0;
	sphere	*h_s0 = new sphere(vec3(0,0,-1), 0.5, d_mat);
	CHECK(cudaMalloc((void **)&d_s0, sizeof(sphere)));
	CHECK(cudaMemcpy(d_s0, h_s0, sizeof(sphere), cudaMemcpyHostToDevice));
	h_list[0] = d_s0;

	// s1
	sphere	*d_s1;
	sphere	*h_s1 = new sphere(vec3(0,-100.5,-1), 100, d_mat);
	CHECK(cudaMalloc((void **)&d_s1, sizeof(sphere)));
	CHECK(cudaMemcpy(d_s1, h_s1, sizeof(sphere), cudaMemcpyHostToDevice));
	h_list[1] = d_s1;

	CHECK(cudaMemcpy(d_list, h_list, 2*sizeof(hittable *), cudaMemcpyHostToDevice));


	// world
	hittable_list	*h_world;
	hittable_list	*d_world;
	h_world = new hittable_list(d_list, 2);
	CHECK(cudaMalloc((void **)&d_world, sizeof(hittable_list)));
	CHECK(cudaMemcpy(d_world, h_world, sizeof(hittable_list), cudaMemcpyHostToDevice));

	hittable_list	**d_world_ptr;
	CHECK(cudaMalloc((void **)&d_world_ptr, sizeof(hittable_list *)));
	CHECK(cudaMemcpy(d_world_ptr, &d_world, sizeof(hittable_list *), cudaMemcpyHostToDevice));



	// camera
	camera	*h_camera;
	camera	*d_camera;

	vec3	lookfrom(3, 3, 2);
	vec3	lookat(0, 0, -1);
	float	dist_to_focus = (lookfrom - lookat).length();
	float	aperture = 2.0f;
	h_camera = new camera(lookfrom, lookat, vec3(0,1,0), 20.0, ASPECT_RATIO, aperture, dist_to_focus);
	CHECK(cudaMalloc((void **)&d_camera, sizeof(camera)));
	CHECK(cudaMemcpy(d_camera, h_camera, sizeof(camera), cudaMemcpyHostToDevice));




	std::cerr << d_list << " " << d_s0 << " " << d_s1 << " " << d_world << " " << d_camera << "\n";
	std::cerr << h_list[0] << ' ' << h_list[1] << '\n';
	std::cerr << h_world->list << '\n';



	CHECK(cudaMalloc((void **)&d_rand_state, PIXELS * sizeof(curandState)));
	//CHECK(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));
	//CHECK(cudaMalloc((void **)&d_list, (2)*sizeof(hittable *)));
	//CHECK(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	//CHECK(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	// rand_init<<<1, 1>>>(d_rand_state2);
	// CHECK(cudaGetLastError());
	// CHECK(cudaDeviceSynchronize());
	// create_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera, d_rand_state2);
	// CHECK(cudaGetLastError());
	// CHECK(cudaDeviceSynchronize());

	dim3 blocks(W / BLOCK_W + 1, H / BLOCK_H + 1);
	dim3 threads(BLOCK_W, BLOCK_H);
	rand_init<<<blocks, threads>>>(d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	start = clock();
	render<<<blocks, threads>>>(d_buf, d_camera, d_world_ptr, d_rand_state);
	fflush(stdout);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "Took: " << ((double)(stop - start)) / CLOCKS_PER_SEC << "\n";

	//CHECK(cudaMemcpy(h_buf, d_buf, BSIZE, cudaMemcpyDeviceToHost));

	print(d_buf);
	//print(h_buf);


	//free
	/*
	for (int i = 0; i < 2; ++i)
	{
		delete ((sphere *)d_list[i])->mat;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;

	free_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera);
	*/




	CHECK(cudaFree(d_camera));
	CHECK(cudaFree(d_world));
	CHECK(cudaFree(d_s1));
	CHECK(cudaFree(d_s0));
	CHECK(cudaFree(d_mat));
	CHECK(cudaFree(d_list));

	delete h_camera;
	delete h_world;
	delete h_s1;
	delete h_s0;
	delete h_mat;
	delete h_list;

	//CHECK(cudaGetLastError());
	//CHECK(cudaDeviceSynchronize());
	//CHECK(cudaFree(d_camera));
	//CHECK(cudaFree(d_list));
	//CHECK(cudaFree(d_world));
	CHECK(cudaFree(d_rand_state));
	//CHECK(cudaFree(d_rand_state2));
	CHECK(cudaFree(d_buf));
	//free(h_buf);

	return (0);
}

