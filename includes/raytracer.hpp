#ifndef RAYTRACER_HPP
# define RAYTRACER_HPP


# include "utils.hpp"
# include "vec3.hpp"
# include "ray.hpp"
# include "hittable.hpp"
# include "hittable_list.hpp"
# include "sphere.hpp"


# define W 1200
# define H 800
# define PIXELS W * H

# define SAMPLES 32

# define BSIZE 3 * PIXELS * sizeof(float)
# define BLOCK_W 8
# define BLOCK_H 8

# define SEED 42
# define REFRACTION 8


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

*/

#endif
