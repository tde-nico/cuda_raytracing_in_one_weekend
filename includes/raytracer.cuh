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

# define SAMPLES 64
# define REFRACTION 50

# define BSIZE PIXELS * sizeof(vec3)
# define BLOCK_W 32
# define BLOCK_H 32

# define SEED 89

# define MANAGED 0

# define SHARED 0
# define WEIGHT 0.5f


#endif
