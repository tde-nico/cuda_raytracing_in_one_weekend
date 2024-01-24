#ifndef RAYTRACER_CUH
# define RAYTRACER_CUH


# include "utils.cuh"
# include "vec3.cuh"
# include "ray.cuh"
# include "hittable.cuh"
# include "hittable_list.cuh"
# include "sphere.cuh"


# define W 1200
# define H 800
# define PIXELS W * H

# define SAMPLES 25

# define BSIZE PIXELS * sizeof(vec3)
# define BLOCK_W 8
# define BLOCK_H 8

# define SEED 42
# define REFRACTION 10

# define SHARED 1
# define WEIGHT 0.5f


/*
2.07126
*/

#endif
