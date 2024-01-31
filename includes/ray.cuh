#ifndef RAY_CUH
# define RAY_CUH

# include "vec3.cuh"


/**
 * @brief Class representing rays
 * 
 * A class that represents rays as a tuple of origin and direction
 * as vec3 and some utilities functions
*/
class ray
{
	public:
		vec3	orig;
		vec3	dir;

		/**
		 * @brief Empty Costructor
		*/
		__device__ ray() {}

		/**
		 * @brief Standard Costructor
		 * @param origin the origin point of the ray
		 * @param direction the direction of the ray
		*/
		__device__ ray(const vec3 &origin, const vec3 &direction): orig(origin), dir(direction) {}
		
		/**
		 * @brief A getter for origin
		 * @return the origin of the ray
		*/
		__device__ vec3	origin() const { return (this->orig); }
		
		/**
		 * @brief A getter for direction
		 * @return the direction of the ray
		*/
		__device__ vec3	direction() const { return (this->dir); }

		/**
		 * @brief A scalar product of the ray
		 * @return the product of the scalar and the ray 
		*/
		__device__ vec3	at(const double t) const { return (this->orig + this->dir * t); }
};


#endif
