#ifndef HITTABLE_LIST_CUH
# define HITTABLE_LIST_CUH

# include "hittable.cuh"
# include "sphere.cuh"


/**
 * @brief A list of hittables
 * 
 * This class contains a list of hittables objects, it also provides an hit
 * function to find the closest hitted object from the list
*/
class hittable_list: public hittable
{
	public:
		hittable	**list;
		int			size;

		/**
		 * @brief Empty constructor
		*/
		__device__ hittable_list() {}
		
		/**
		 * @brief Inizialized constructor
		 * @param l a list of hittables
		 * @param n the size of the list
		 * 
		 * A constructor that is initialized with given values
		*/
		__device__ hittable_list(hittable **l, int n): list(l), size(n) {}
	
		__device__ virtual bool	hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const override;
};


/**
 * @brief Check if some of the listed objects is hitted and returns the closest
 * @param r the ray to analyze
 * @param t_min the minimum range span to hit
 * @param t_max the maximum range span to hit
 * @param rec the output structure filled with hit record data
 * @return the boolean value of true if hitten or false if not
 * 
 * it searches the closest hittable object from the given ray
*/
__device__ bool	hittable_list::hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const
{
	t_hit_record	tmp;
	bool			hit_anything;
	float			closest;

	hit_anything = false;
	closest = t_max;
	for (int i = 0; i < this->size; ++i)
	{
		if (this->list[i]->hit(r, t_min, closest, tmp))
		{
			hit_anything = true;
			closest = tmp.t;
			rec = tmp;
		}
	}
	return (hit_anything);
}


/**
 * @brief An optimized version of hittable_list->hit
 * @param r the ray to analyze
 * @param t_min the minimum range span to hit
 * @param t_max the maximum range span to hit
 * @param rec the output structure filled with hit record data
 * @return the boolean value of true if hitten or false if not
 * 
 * An optimized version of hittable_list->hit which skips a lot of vtable
 * fetches and som function calls to gain performances at runtime
*/
__device__ bool	O_hit(hittable_list *h, ray &r, float t_min, float t_max, t_hit_record &rec)
{
	t_hit_record	tmp;
	bool			hit_anything;
	float			closest;

	hit_anything = false;
	closest = t_max;
	for (int i = 0; i < h->size; ++i)
	{
		if (O_hit((sphere *)h->list[i], r, t_min, closest, tmp))
		{
			hit_anything = true;
			closest = tmp.t;
			rec = tmp;
		}
	}
	return (hit_anything);
}


#endif
