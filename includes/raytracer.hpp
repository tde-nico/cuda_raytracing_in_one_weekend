#ifndef RAYTRACER_HPP
# define RAYTRACER_HPP


# include "utils.hpp"
# include "vec3.hpp"
# include "ray.hpp"
# include "hittable.hpp"
# include "hittable_list.hpp"
# include "sphere.hpp"


# define W 800
# define H 800
# define PIXELS W * H

# define SAMPLES 100

# define BSIZE 3 * PIXELS * sizeof(float)
# define BLOCK_W 8
# define BLOCK_H 8

# define SEED 42


#endif
