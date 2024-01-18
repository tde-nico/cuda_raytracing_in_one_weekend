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
# define ASPECT_RATIO float(W) / float(H)

# define BSIZE 3 * PIXELS * sizeof(float)
# define BLOCK_W 8
# define BLOCK_H 8

# define VIEW_H 2.0f
# define VIEW_W ASPECT_RATIO * VIEW_H
# define FOCAL_LEN 1.0f

# define ORIGIN vec3(0, 0, 0)
# define HORIZONTAL vec3(VIEW_W, 0, 0)
# define VERTICAL vec3(0, VIEW_H, 0)
# define LOWER_LEFT_CORNER vec3(-VIEW_W/2, -VIEW_H/2, -FOCAL_LEN)

#endif
