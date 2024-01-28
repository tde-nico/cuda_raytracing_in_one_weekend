#ifndef HITTABLE_CUH
# define HITTABLE_CUH

# include "ray.cuh"

class material;

/**
 * @brief this struct contains hit record informations
 * This struct is made of useful informations about hits
 * like the hitted material, the hit distance, the point
 * of impact and the normal of the hitten point.
*/
typedef struct s_hit_record
{
	vec3		p;
	vec3		normal;
	float		t;
	material	*mat;
}	t_hit_record;


/**
 * @brief this class represents an hittable object
 * This is an abstract class with only virtual methods which
 * will get overrided by derived classes.
*/

class hittable
{
	public:
		__device__ virtual bool	hit(const ray &r, float	t_min, float t_max, t_hit_record &rec) const = 0;
};


#endif
