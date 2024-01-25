// -rdc=true stands for Relocatable Device Code
__global__ void	sample(curandState state, int x, int y, camera **cam, hittable **world, vec3 *final_color)
{
	float		u;
	float		v;
	ray			r;
	vec3		color;
	__shared__ vec3	samples[SAMPLES];


	int	id = threadIdx.x + blockIdx.x * blockDim.x;
	int	stride = SAMPLES / 2;

	u = float(x + curand_uniform(&state)) / float(W);
	v = float(y + curand_uniform(&state)) / float(H);
	r = O_get_ray(*cam, u, v, &state);
	color = ray_color(r, world, &state);

	samples[0] = color;

	for (int i = stride; i > 0; i >>= 1)
	{
		if (id < i)
			samples[id] += samples[id + i];
		__syncthreads();
	}
	if (!id)
		*final_color = samples[0];
}
