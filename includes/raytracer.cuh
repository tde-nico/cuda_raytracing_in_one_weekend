#ifndef RAYTRACER_CUH
# define RAYTRACER_CUH


# include <time.h>
# include <float.h>
# include <stdio.h>

# include "utils.cuh"
# include "vec3.cuh"
# include "ray.cuh"
# include "hittable.cuh"
# include "hittable_list.cuh"
# include "sphere.cuh"
# include "camera.cuh"
# include "material.cuh"


# define W 1200
# define H 800
# define PIXELS W * H

# define SAMPLES 32

# define BSIZE PIXELS * sizeof(vec3)
# define BLOCK_W 8
# define BLOCK_H 8

# define SEED 42
# define REFRACTION 100

# define SHARED 1
# define WEIGHT 0.5f

# define MANEGED 0


#endif
