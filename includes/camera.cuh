#ifndef CAMERA_CUH
# define CAMERA_CUH

# include "ray.cuh"


# define ASPECT_RATIO float(W) / float(H)
# define VIEW_H 2.0f
# define VIEW_W ASPECT_RATIO * VIEW_H
# define FOCAL_LEN 1.0f

# define ORIGIN vec3(0, 0, 0)
# define HORIZONTAL vec3(VIEW_W, 0, 0)
# define VERTICAL vec3(0, VIEW_H, 0)
# define LOWER_LEFT_CORNER vec3(-VIEW_W/2, -VIEW_H/2, -FOCAL_LEN)


/**
 * @brief generates a random uniform vec3
 * @param s the pointer of the curandState
 * @return a random uniform vec3
 * 
 * Given the current cuda random state, it generates a uniform vec3
*/
__device__ vec3	unit_disk_rand(curandState *s)
{
	vec3	p;

	p = vec3(curand_uniform(s),curand_uniform(s),0)*2.0f - vec3(1,1,0);
	while (dot(p, p) >= 1.0f)
		p = vec3(curand_uniform(s),curand_uniform(s),0)*2.0f - vec3(1,1,0);
	return (p);
}


/**
 * @brief The class representing the camera (POV)
 * 
 * This class represents the camera with its point of view, with its origin,
 * focus, and render plane, it also provides a ray generator.
*/
class camera
{
	public:
		vec3	origin;
		vec3	lower_left_corner;
		vec3	horizontal;
		vec3	vertical;
		vec3	v;
		vec3	u;
		vec3	w;
		float	lens_radius;

		/**
		 * @brief The camera constructor
		 * @param lookfrom the origin point
		 * @param lookat where to look at
		 * @param vup the camera relative up direction
		 * @param vfov the vertical fov
		 * @param aspect the aspect ratio of the camera
		 * @param aperture the aperture of the lens of the camera
		 * @param focus_dist the distance of perfect focus from the camera
		 * 
		 * given the initialization parameters, it creates the needed vectors to
		 * then generate new rays from the camera
		*/
		__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist)
		{
			float	theta;
			float	half_h;
			float	half_w;

			this->lens_radius = aperture / 2.0f;
			theta = vfov * float(M_PI) / 180.0f;
			half_h = tan(theta / 2.0f);
			half_w = aspect * half_h;
			this->origin = lookfrom;
			this->w = unit_vector(lookfrom - lookat);
			this->u = unit_vector(cross(vup, this->w));
			this->v = cross(this->w, this->u);
			this->lower_left_corner = this->origin - this->u*half_w*focus_dist - this->v*half_h*focus_dist - this->w*focus_dist;
			this->horizontal = this->u * half_w * focus_dist * 2.0f;
			this->vertical = this->v * half_h * focus_dist * 2.0f;			
		}

		/**
		 * @brief Generates a ray from the camera
		 * @param s the uniform pointed x coordinate
		 * @param t the uniform pointed y coordinate
		 * @param state the random state
		 * @return the ray pointing (x,y)
		 * 
		 * It generates a ray from the camera in the direction (x,y) with a slight
		 * shift given by the lens focus
		*/
		__device__ ray	get_ray(float s, float t, curandState *state)
		{
			vec3	offset;
			vec3	rd;

			rd = unit_disk_rand(state) * this->lens_radius;
			offset = this->u * rd.x() + this->v * rd.y();
			return (ray(this->origin + offset, this->lower_left_corner +
				this->horizontal*s + this->vertical*t - this->origin - offset));
		}
};


/**
 * @brief An optimized version of camera->get_ray
 * @param c the camera
 * @param s the uniform pointed x coordinate
 * @param t the uniform pointed y coordinate
 * @param state the random state
 * @return the ray pointing (x,y)
 * 
 * An optimized version of camera->get_ray used to skip some operations of fetching
 * from the class vtable at runtime to gain performance
*/
__device__ ray	O_get_ray(camera *c, float s, float t, curandState *state)
{
	vec3	offset;
	vec3	rd;

	rd = unit_disk_rand(state) * c->lens_radius;
	offset = c->u * rd.e[0] + c->v * rd.e[1];
	return (ray(c->origin + offset, c->lower_left_corner +
		c->horizontal*s + c->vertical*t - c->origin - offset));
}


#endif
