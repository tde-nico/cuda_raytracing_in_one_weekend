#ifndef VEC3_CUH
# define VEC3_CUH


# include "utils.cuh"


class vec3
{
	public:
		float	e[3];

		/**
		 * @brief Empty Constructor
		 * 
		 * it initializes all the three values at 0
		*/
		__host__ __device__ vec3(): e{0,0,0} {}
		
		/**
		 * @brief Standard Constructor
		 * @param e0 the x of the vec3
		 * @param e1 the y of the vec3
		 * @param e2 the z of the vec3
		*/
		__host__ __device__ vec3(float e0, float e1, float e2): e{e0,e1,e2} {}

		/**
		 * @brief A getter for x
		 * @return the x of the vec3
		*/
		__host__ __device__ float	x() const { return (this->e[0]); }
	
		/**
		 * @brief A getter for y
		 * @return the y of the vec3
		*/
		__host__ __device__ float	y() const { return (this->e[1]); }

		/**
		 * @brief A getter for z
		 * @return the z of the vec3
		*/
		__host__ __device__ float	z() const { return (this->e[2]); }

		/**
		 * @brief A getter for x
		 * @return the x of the vec3
		*/
		__host__ __device__ float	r() const { return (this->e[0]); }

		/**
		 * @brief A getter for y
		 * @return the y of the vec3
		*/
		__host__ __device__ float	g() const { return (this->e[1]); }

		/**
		 * @brief A getter for z
		 * @return the z of the vec3
		*/
		__host__ __device__ float	b() const { return (this->e[2]); }

		/**
		 * @brief A vec3 inverter
		 * @return the inverted vec3
		*/
		__host__ __device__ vec3	operator-() const { return (vec3(-this->e[0], -this->e[1], -this->e[2])); }

		/**
		 * @brief An indexed getter
		 * @param i the index of the value to get
		 * @return the value indexed at i
		*/
		__host__ __device__ float	operator[](int i) const { return (this->e[i]); }

		/**
		 * @brief An indexed getter
		 * @param i the index of the value to get
		 * @return the value indexed at i
		*/
		__host__ __device__ float	&operator[](int i) { return (this->e[i]); }

		/**
		 * @brief The sum operator
		 * @param v the vec3 to sum with
		 * @return the summed result
		*/
		__host__ __device__ vec3	&operator+=(const vec3 &v)
		{
			this->e[0] += v.e[0];
			this->e[1] += v.e[1];
			this->e[2] += v.e[2];
			return (*this);
		}

		/**
		 * @brief The multiplication operator
		 * @param v the vec3 to multiply with
		 * @return the multiplied result
		*/
		__host__ __device__ vec3	&operator*=(const vec3 &v)
		{
			this->e[0] *= v.e[0];
			this->e[1] *= v.e[1];
			this->e[2] *= v.e[2];
			return (*this);
		}

		/**
		 * @brief The multiplication operator
		 * @param t the scalar to multiply with
		 * @return the multiplied result
		*/
		__host__ __device__ vec3	&operator*=(const float t)
		{
			this->e[0] *= t;
			this->e[1] *= t;
			this->e[2] *= t;
			return (*this);
		}

		/**
		 * @brief The division operator
		 * @param t the scalar to divide with
		 * @return the divided result
		*/
		__host__ __device__ vec3	&operator/=(const float t)
		{
			const float	tmp = 1/t;
			this->e[0] *= tmp;
			this->e[1] *= tmp;
			this->e[2] *= tmp;
			return (*this);
		}

		/**
		 * @brief Computes the absolute length of the vec3
		 * @return the absolute length of the vec3
		*/
		__host__ __device__ float	length() const { return (std::sqrt(this->length_squared())); }

		/**
		 * @brief Computes the squared length of the vec3
		 * @return the squared length of the vec3
		*/
		__host__ __device__ float	length_squared() const
		{
			return (this->e[0] * this->e[0]
				+ this->e[1] * this->e[1]
				+ this->e[2] * this->e[2]);
		}
};

/**
 * @brief The istream operator
 * @param in the istream
 * @param v the vec3
 * @return The input streaming of the vec3
*/
inline std::istream	&operator>>(std::istream &in, vec3 &v)
{
	return (in >> v.e[0] >> v.e[1] >> v.e[2]);
}

/**
 * @brief The ostream operator
 * @param out the ostream
 * @param v the vec3
 * @return The output streaming of the vec3
*/
inline std::ostream	&operator<<(std::ostream &out, const vec3 &v)
{
	return (out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2]);
}

/**
 * @brief The sum operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The sum of u and v
*/
__host__ __device__ inline vec3	operator+(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]));
}

/**
 * @brief The sub operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The subtraction of u and v
*/
__host__ __device__ inline vec3	operator-(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]));
}

/**
 * @brief The mul operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The multiplication of u and v
*/
__host__ __device__ inline vec3	operator*(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]));
}

/**
 * @brief The mul operator
 * @param t the scalar
 * @param v the vec3
 * @return The multiplication of t and v
*/
__host__ __device__ inline vec3	operator*(const float t, const vec3 &v)
{
	return (vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t));
}

/**
 * @brief The mul operator
 * @param v the vec3
 * @param t the scalar
 * @return The multiplication of v and t
*/
__host__ __device__ inline vec3	operator*(const vec3 &v, const float t)
{
	return (vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t));
}

/**
 * @brief The div operator
 * @param v the vec3
 * @param t the scalar
 * @return The division of v and t
*/
__host__ __device__ inline vec3	operator/(const vec3 &v, const float t)
{
	const float	tmp = 1/t;
	return (vec3(v.e[0] * tmp, v.e[1] * tmp, v.e[2] * tmp));
}

/**
 * @brief The div operator
 * @param t the scalar
 * @param v the vec3
 * @return The division of v and t
*/
__host__ __device__ inline vec3	operator/(const float t, const vec3 &v)
{
	const float	tmp = 1/t;
	return (vec3(v.e[0] * tmp, v.e[1] * tmp, v.e[2] * tmp));
}

/**
 * @brief The div operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The division of u and v
*/
__host__ __device__ inline vec3	operator/(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]));
}

/**
 * @brief The dot operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The dot of u and v
*/
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
	return (u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]);
}

/**
 * @brief The cross operator
 * @param u the first vec3
 * @param v the second vec3
 * @return The cross of u and v
*/
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]));
}

/**
 * @brief Computes the unit of a vector
 * @param v the vec3
 * @return The unit of v
*/
__host__ __device__ inline vec3	unit_vector(vec3 v)
{
	return (v / v.length());
}


#endif
