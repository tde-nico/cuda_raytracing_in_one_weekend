#ifndef SPHERE_CUH
# define SPHERE_CUH

# include "vec3.cuh"
# include "hittable.cuh"


/**
 * @brief A sphere object derived by hittable
 * 
 * This class represents a sphere with its core functions
*/
class sphere: public hittable
{
	public:
		vec3		center;
		float		radius;
		material	*mat;

		/**
		 * @brief Empty Constructor
		*/
		__device__ sphere() {}

		/**
		 * @brief Standard Constructor
		 * @param c the center of the sphere
		 * @param r the radius of the sphere
		 * @param m the material of the sphere
		*/
		__device__ sphere(vec3 c, float r, material *m): center(c), radius(r), mat(m) {};

		__device__ virtual bool	hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const override;
};


/**
 * @brief A function to check if hitted
 * @param r the ray to analyze
 * @param t_min the minimum range span to hit
 * @param t_max the maximum range span to hit
 * @param rec the output structure filled with hit record data
 * @return the boolean value of true if hitten or false if not
 * 
 * This function is called to check if a certain ray is hitting the current sphere
*/
__device__ bool	sphere::hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const
{
	vec3	oc;
	float	a;
	float	half_b;
	float	c;
	float	discriminant;
	float	sqrtd;
	float	root;

	oc = r.origin() - this->center;
	a = r.direction().length_squared();
	half_b = dot(oc, r.direction());
	c = oc.length_squared() - this->radius * this->radius;
	
	discriminant = half_b*half_b - a*c;
	if (discriminant < 0)
		return (false);
	sqrtd = std::sqrt(discriminant);

	root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max)
	{
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max)
			return (false);
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	rec.normal = (rec.p - this->center) / this->radius;
	rec.mat = this->mat;

	return (true);
}


/**
 * @brief An optimized version of sphere->hit
 * @param r the ray to analyze
 * @param t_min the minimum range span to hit
 * @param t_max the maximum range span to hit
 * @param rec the output structure filled with hit record data
 * @return the boolean value of true if hitten or false if not
 * 
 * An optimized version of sphere->hit which skips a lot of vtable
 * fetches and som function calls to gain performances at runtime
*/
__device__ bool	O_hit(sphere *s, const ray &r, float t_min, float t_max, t_hit_record &rec)
{
	vec3	oc;
	float	a;
	float	half_b;
	float	c;
	float	discriminant;
	float	sqrtd;
	float	root;

	oc = r.orig - s->center;
	a = r.dir.length_squared();
	half_b = dot(oc, r.dir);
	c = oc.length_squared() - s->radius * s->radius;
	
	discriminant = half_b*half_b - a*c;
	if (discriminant < 0)
		return (false);
	sqrtd = std::sqrt(discriminant);

	root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max)
	{
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max)
			return (false);
	}

	rec.t = root;
	rec.p = r.orig + r.dir * rec.t;
	rec.normal = (rec.p - s->center) / s->radius;
	rec.mat = s->mat;

	return (true);
}

#endif
