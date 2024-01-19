#ifndef SPHERE_HPP
# define SPHERE_HPP

# include "vec3.hpp"
# include "hittable.hpp"


class sphere: public hittable
{
	public:
		vec3		center;
		float		radius;
		material	*mat;
		
		__device__ sphere() {}
		__device__ sphere(vec3 c, float r, material *m): center(c), radius(r), mat(m) {};

		__device__ virtual bool	hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const override;
};


__device__ bool	sphere::hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const
{
	vec3	oc;
	float	a;
	float	half_b;
	float	c;
	float	discriminant;
	float	sqrtd;
	float	root;

	oc = r.origin() - center;
	a = r.direction().length_squared();
	half_b = dot(oc, r.direction());
	c = oc.length_squared() - radius * radius;
	
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


#endif
