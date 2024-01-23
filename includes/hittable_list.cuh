#ifndef HITTABLE_LIST_CUH
# define HITTABLE_LIST_CUH

# include "hittable.cuh"


class hittable_list: public hittable
{
	public:
		hittable	**list;
		int			size;

		__device__ hittable_list() {}
		__device__ hittable_list(hittable **l, int n): list(l), size(n) {}
	
		__device__ virtual bool	hit(const ray &r, float t_min, float t_max, t_hit_record &rec) const override;
};


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


#endif
