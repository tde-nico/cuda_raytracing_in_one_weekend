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
		vec3	origin;
		vec3	lower_left_corner;
		vec3	horizontal;
		vec3	vertical;

		__device__ camera(/*vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect*/)
		{
			/*vec3	v;
			vec3	u;
			vec3	w;
			float	theta;
			float	half_h;
			float	half_w;

			theta = vfov * M_PI / 180;
			half_h = tan(theta / 2);
			half_w = aspect * half_h;
			this->origin = lookfrom;
			w = unit_vector(lookfrom - lookat);
			u = unit_vector(cross(vup, w));
			v = cross(w, u);
			this->lower_left_corner = this->origin - u*half_w - v*half_h - w;
			this->horizontal = u * half_w * 2;
			this->vertical = v * half_h * 2;*/
			this->origin = ORIGIN;
			this->lower_left_corner = LOWER_LEFT_CORNER;
			this->horizontal = HORIZONTAL;
			this->vertical = VERTICAL;
		}

		__device__ ray	get_ray(float u, float v)
		{
			return (ray(this->origin, this->lower_left_corner +
				this->horizontal*u + this->vertical*v - this->origin));
		}
};


#endif
