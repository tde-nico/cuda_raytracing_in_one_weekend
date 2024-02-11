#include "raytracer.cuh"


// #################### RENDER ####################


/**
 * @brief Computes the path of a given ray
 * @param r the ray to analyze
 * @param world the current world where the ray is been analyzed
 * @param rand_state the current random state
 * @return the color of the resulting ray's path
 * 
 * This function computes the path of a give ray, by hitting a material and
 * scattering until no hits occours or when approaching the refraction limit
*/
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


/**
 * @brief Computes the color of a given pixel
 * @param buf the final buffer image
 * @param cam the camera where the rays are coming from
 * @param world the current world to analyze with the rays
 * @param rand_state the current random state
 * 
 * This function computes the color of a given pixel by scattering
 * a fixed amount of sample rays to approximate the color of the given area,
 * it can also improve the quality of the resulting image by using also the
 * samples of the adiacent pixels via shared memory and executing some
 * gamma corretion to correct the output color.
*/
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
	__shared__ vec3	share_sam[BLOCK_H][BLOCK_W];


	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= W || y >= H)
	{
		#if SHARED
			share_sam[threadIdx.y][threadIdx.x] = vec3(0, 0, 0);
		#endif
		return ;
	}
	i = W*y + x;

	state = rand_state[i * SAMPLES + threadIdx.z];

	if (!threadIdx.z)
		share_sam[threadIdx.y][threadIdx.x] = vec3(0, 0, 0);
	__syncthreads();

	u = float(x + curand_uniform(&state)) / float(W);
	v = float(y + curand_uniform(&state)) / float(H);
	r = O_get_ray(*cam, u, v, &state);
	color = ray_color(r, world, &state);

	color /= float(SAMPLES);
	atomicAdd(&share_sam[threadIdx.y][threadIdx.x][0], color[0]);
	atomicAdd(&share_sam[threadIdx.y][threadIdx.x][1], color[1]);
	atomicAdd(&share_sam[threadIdx.y][threadIdx.x][2], color[2]);
	if (threadIdx.z)
		return ;
	__syncthreads();
	color = share_sam[threadIdx.y][threadIdx.x];


	#if SHARED
		vec3	sam;
		float	counter;

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


// #################### INIT ####################


/**
 * @brief Initialize a random state
 * @param rand_state the current random state
 * 
 * This function initialize the given random state
*/
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

	curand_init(SEED, i * SAMPLES + threadIdx.z, 0, &rand_state[i * SAMPLES + threadIdx.z]);
}


#define RND (curand_uniform(&local_rand_state))
/**
 * @brief Initialize the current world
 * @param d_list the list of objects in the world
 * @param d_world the world to render
 * @param d_camera the camera to render from
 * @param rand_state the current random state
 * 
 * This function initialize the given random state
*/
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
	*d_world  = new hittable_list(d_list, 22*22+1+3);

	vec3 lookfrom(13,2,3);
	vec3 lookat(0,0,0);
	float dist_to_focus = 10.0; (lookfrom-lookat).length();
	float aperture = 0.1;
	*d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, ASPECT_RATIO, aperture, dist_to_focus);
}


// #################### FREE ####################


/**
 * @brief Deletes the world data on the device
 * @param d_list the list of objects in the world
 * @param d_world the world to render
 * @param d_camera the camera to render from
 * 
 * This function deletes all the previously created objects in the
 * world and the world itself with its list
*/
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


// #################### UTILS ####################


/**
 * @brief Writes color in the oust stream
 * @param out the output stream
 * @param pixel the color to write
 * 
 * This function writes a color given as uniform vec3 into
 * an out ostream
*/
inline void	write_color(std::ostream &out, vec3 pixel)
{
	out << int(255.99 * pixel.r()) << ' '
		<< int(255.99 * pixel.g()) << ' '
		<< int(255.99 * pixel.b()) << '\n';
}


/**
 * @brief Writes the image into the output
 * @param buf the matrix representing the image
 * 
 * This function writes the image into the output stream
 * by using ppm format
*/
inline void	print(vec3 *buf)
{
	std::cout << "P3\n" << W << " " << H << "\n255\n";
	for (int y = H-1; y >= 0; --y)
	{
		for (int x = 0; x < W; ++x)
			write_color(std::cout, buf[W*y + x]);
	}
}


// #################### MAIN ####################


/**
 * @brief The main function
 * 
 * In this function anfter allocating the host needed memory,
 * the world initializer is called, then a kernel is deployed
 * for every pixel to compute the colors, that are printed after
 * the coputation, ending with the memory deallocation.
*/
int	main(void)
{
	#if !MANAGED
		vec3		*h_buf;
	#endif
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

	#if MANAGED
		CHECK(cudaMallocManaged((void **)&d_buf, BSIZE));
	#else
		CHECK(cudaMallocHost((void **)&h_buf, BSIZE));
		CHECK(cudaMalloc((void **)&d_buf, BSIZE));
		CHECK(cudaMemcpy(d_buf, h_buf, BSIZE, cudaMemcpyHostToDevice));
	#endif

	CHECK(cudaMalloc((void **)&d_rand_state, PIXELS * SAMPLES * sizeof(curandState)));
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
	dim3 threads(BLOCK_W, BLOCK_H, SAMPLES);
	rand_init<<<blocks, threads>>>(d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	start = clock();
	render<<<blocks, threads>>>(d_buf, d_camera, (hittable **)d_world, d_rand_state);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stop = clock();
	std::cerr << "Took: " << ((double)(stop - start)) / CLOCKS_PER_SEC << "\n";

	#if MANAGED
		print(d_buf);
	#else
		start = clock();
		CHECK(cudaMemcpy(h_buf, d_buf, BSIZE, cudaMemcpyDeviceToHost));
		print(h_buf);
		stop = clock();
		std::cerr << "Took: " << (double)(stop - start) << "\n";
	#endif

	free_world<<<1,1>>>((hittable **)d_list, (hittable **)d_world, d_camera);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaFree(d_camera));
	CHECK(cudaFree(d_list));
	CHECK(cudaFree(d_world));
	CHECK(cudaFree(d_rand_state));
	CHECK(cudaFree(d_rand_state2));
	CHECK(cudaFree(d_buf));
	#if !MANAGED
		CHECK(cudaFreeHost(h_buf));
	#endif

	return (0);
}

