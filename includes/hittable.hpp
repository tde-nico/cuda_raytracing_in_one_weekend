#ifndef HITTABLE_HPP
# define HITTABLE_HPP

# include "ray.hpp"

class material;

typedef struct s_hit_record
{
	vec3		p;
	vec3		normal;
	float		t;
	material	*mat;
}	t_hit_record;


class hittable
{
	public:
		__device__ virtual bool	hit(const ray &r, float	t_min, float t_max, t_hit_record &rec) const = 0;
};


#endif
