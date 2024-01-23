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

# define SAMPLES 32

# define BSIZE PIXELS * sizeof(vec3)
# define BLOCK_W 8
# define BLOCK_H 8

# define SEED 42
# define REFRACTION 50


/*
SAMPLES 400
REFRACTION 50
800+

SAMPLES 32
REFRACTION 8
61 / 62

W 1200 -> 800
H 800 -> 600
29 / 30


10.2347
10.2802

*/

#endif
