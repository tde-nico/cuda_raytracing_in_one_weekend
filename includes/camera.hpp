#ifndef CAMERA_HPP
# define CAMERA_HPP


# include "raytracer.hpp"


# define ASPECT_RATIO float(W) / float(H)
# define VIEW_H 2.0f
# define VIEW_W ASPECT_RATIO * VIEW_H
# define FOCAL_LEN 1.0f

# define ORIGIN vec3(0, 0, 0)
# define HORIZONTAL vec3(VIEW_W, 0, 0)
# define VERTICAL vec3(0, VIEW_H, 0)
# define LOWER_LEFT_CORNER vec3(-VIEW_W/2, -VIEW_H/2, -FOCAL_LEN)


class camera
{
	public:
		__device__ camera() {}

		__device__ ray	get_ray(float u, float v)
		{
			return (ray(ORIGIN, LOWER_LEFT_CORNER + HORIZONTAL*u + VERTICAL*v - ORIGIN));
		}
};


#endif
