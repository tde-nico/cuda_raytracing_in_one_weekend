#ifndef MATERIAL_HPP
# define MATERIAL_HPP

#include "raytracer.hpp"

__device__ vec3	unit_sphere_rand(curandState *s)
{
	vec3	p;
	p = vec3(curand_uniform(s),curand_uniform(s),curand_uniform(s)) * 2.0f - vec3(1,1,1);
	while (p.length_squared() >= 1.0f)
	return (p);
}

__device__ vec3	reflect(const vec3 &v, const vec3 &u)
{
	return (v - u * dot(v, u) *2.0f);
}


class material
{
	public:
		__device__ virtual bool scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const = 0;
};


class lambertian: public material
{
	public:
		vec3	albedo;

		__device__ lambertian(const vec3 &a): albedo(a) {}
		
		__device__ virtual bool scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const override
		{
			vec3	target;

			target = rec.normal + unit_sphere_rand(state);
			scattered = ray(rec.p, target - rec.p);
			attenuation = this->albedo;
			return (true);
		}
};


class metal: public material
{
	public:
		vec3	albedo;
		float	fuzz;

		__device__ metal(const vec3 &a, float f): albedo(a), fuzz(f < 1 ? f : 1) {}
		
		__device__ virtual bool scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const override
		{
			vec3	reflected;

			reflected = reflect(unit_vector(r_in.direction()), rec.normal);
			scattered = ray(rec.p, reflected + unit_sphere_rand(state) * this->fuzz);
			attenuation = this->albedo;
			return (dot(scattered.direction(), rec.normal) > 0.0f);
		}
};


#endif
