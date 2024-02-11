#ifndef MATERIAL_CUH
# define MATERIAL_CUH

# include "ray.cuh"
# include "hittable.cuh"


/**
 * @brief generates a random uniform vec3
 * @param s the pointer of the curandState
 * @return a random uniform vec3
 * 
 * Given the current cuda random state, it generates a uniform vec3
*/
__device__ vec3	unit_sphere_rand(curandState *s)
{
	vec3	p;

	p = vec3(curand_uniform(s),curand_uniform(s),curand_uniform(s)) * 2.0f - vec3(1,1,1);
	while (p.length_squared() >= 1.0f)
		p = vec3(curand_uniform(s),curand_uniform(s),curand_uniform(s)) * 2.0f - vec3(1,1,1);
	return (p);
}

/**
 * @brief Computes a reflection
 * @param v the direction of the ray
 * @param u the normal of the ray
 * @return the reflected ray
 * 
 * Computes the reflection of a given ray
*/
__device__ vec3	reflect(const vec3 &v, const vec3 &u)
{
	return (v - u * dot(v, u) * 2.0f);
}


/**
 * @brief Computes an approssimated refraction
 * @param cos cosine of the refraction
 * @param ir the refraction index
 * @return the refraction probability
 * 
 * An algorithm of approximation of glass refraction by Christophe Schlick
*/
__device__ float	schlick(float cos, float ir)
{
	float	r;

	r = (1.0f - ir) / (1.0f + ir);
	r = r * r;
	return (r + (1.0f - r)*pow((1.0f - cos), 5.0f));
}


/**
 * @brief Computes a refraction
 * @param v direction unit vector
 * @param u outward unit vector
 * @param ni_over_nt normals refraction indices
 * @param refracted the output refraction
 * @return a bool: true if the ray is refracted else false
 * 
 * Computes a refraction by its parameters and fills the refracted parameter with
 * the output refraction given by the Snell's law.
*/
__device__ bool	refract(const vec3 &v, const vec3 &u, float ni_over_nt, vec3 &refracted)
{
	vec3	uv;
	float	dt;
	float	discriminant;

	uv = unit_vector(v);
	dt = dot(uv, u);
	discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0)
	{
		refracted = (uv - u * dt) * ni_over_nt - u * std::sqrt(discriminant);
		return (true);
	}
	return (false);
}


/**
 * @brief The abstract class of materials
 * 
 * Represents the materials with their ray scattering virtual function.
*/
class material
{
	public:
		/**
		 * @brief Computes the scatter of a given ray
		 * @param r_in the input ray
		 * @param rec the hit point informations
		 * @param attenuation filled with material attenuation
		 * @param scattered the scattered ray
		 * @param state the random state
		 * @return true if scattered else false
		 * 
		 * Computes the scatter of a given ray
		*/
		__device__ virtual bool	scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const = 0;
};


/**
 * @brief A lambertian material derived from material class
 * 
 * A lambertian material derived from material class
*/
class lambertian: public material
{
	public:
		vec3	albedo;

		/**
		 * @brief A lambertian constructor
		 * @param a the lambertian albedo
		*/
		__device__ lambertian(const vec3 &a): albedo(a) {}

		/**
		 * @brief Computes the scatter of a given ray
		 * @param r_in the input ray
		 * @param rec the hit point informations
		 * @param attenuation filled with material attenuation
		 * @param scattered the scattered ray
		 * @param state the random state
		 * @return true if scattered else false
		 * 
		 * Computes the scatter of a given ray
		*/
		__device__ virtual bool	scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const override
		{
			vec3	target;

			target = rec.p + rec.normal + unit_sphere_rand(state);
			scattered = ray(rec.p, target - rec.p);
			attenuation = this->albedo;
			return (true);
		}
};


/**
 * @brief A metal material derived from material class
 * 
 * A metal material derived from material class
*/
class metal: public material
{
	public:
		vec3	albedo;
		float	fuzz;

		/**
		 * @brief A metal constructor
		 * @param a the metal albedo
		 * @param f the fuzz uniform value
		*/
		__device__ metal(const vec3 &a, float f): albedo(a), fuzz(f < 1 ? f : 1) {}

		/**
		 * @brief Computes the scatter of a given ray
		 * @param r_in the input ray
		 * @param rec the hit point informations
		 * @param attenuation filled with material attenuation
		 * @param scattered the scattered ray
		 * @param state the random state
		 * @return true if scattered else false
		 * 
		 * Computes the scatter of a given ray
		*/
		__device__ virtual bool	scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const override
		{
			vec3	reflected;

			reflected = reflect(unit_vector(r_in.direction()), rec.normal);
			scattered = ray(rec.p, reflected + unit_sphere_rand(state) * this->fuzz);
			attenuation = this->albedo;
			return (dot(scattered.direction(), rec.normal) > 0.0f);
		}
};


/**
 * @brief A dielectric material derived from material class
 * 
 * A dielectric material derived from material class
*/
class dielectric: public material
{
	public:
		float	ir;

		/**
		 * @brief A dielectric constructor
		 * @param ir the dielectric refraction index
		*/
		__device__ dielectric(float refraction_index): ir(refraction_index) {}

		/**
		 * @brief Computes the scatter of a given ray
		 * @param r_in the input ray
		 * @param rec the hit point informations
		 * @param attenuation filled with material attenuation
		 * @param scattered the scattered ray
		 * @param state the random state
		 * @return true if scattered else false
		 * 
		 * Computes the scatter of a given ray
		*/
		__device__ virtual bool	scatter(const ray &r_in, const t_hit_record &rec,
			vec3 &attenuation, ray &scattered, curandState *state) const override
		{
			vec3	outward_normal;
			vec3	reflected;
			float	ni_over_nt;
			vec3	refracted;
			float	reflect_prob;
			float	cos;

			reflected = reflect(r_in.direction(), rec.normal);
			attenuation = vec3(1, 1, 1);
			if (dot(r_in.direction(), rec.normal) > 0.0f)
			{
				outward_normal = -rec.normal;
				ni_over_nt = this->ir;
				cos = dot(r_in.direction(), rec.normal) / r_in.direction().length();
				cos = std::sqrt(1.0f - this->ir*this->ir*(1 - cos*cos));
			}
			else
			{
				outward_normal = rec.normal;
				ni_over_nt = 1.0f / this->ir;
				cos = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
			}
			if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
				reflect_prob = schlick(cos, this->ir);
			else
				reflect_prob = 1.0f;
			if (curand_uniform(state) < reflect_prob)
				scattered = ray(rec.p, reflected);
			else
				scattered = ray(rec.p, refracted);
			return (true);
		}
};


#endif
