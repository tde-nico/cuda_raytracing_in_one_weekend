#ifndef RAY_CUH
# define RAY_CUH

# include "vec3.cuh"


class ray
{
	public:
		vec3	orig;
		vec3	dir;

		__device__ ray() {}
		__device__ ray(const vec3 &origin, const vec3 &direction): orig(origin), dir(direction) {}
		
		__device__ vec3	origin() const { return (this->orig); }
		__device__ vec3	direction() const { return (this->dir); }

		__device__ vec3	at(const double t) const { return (this->orig + this->dir * t); }
};


#endif
