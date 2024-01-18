#ifndef VEC3_HPP
# define VEC3_HPP

# include <cmath>

# include "utils.hpp"


class vec3
{
	public:
		float	e[3];

		__host__ __device__ vec3(): e{0,0,0} {}
		__host__ __device__ vec3(float e0, float e1, float e2): e{e0,e1,e2} {}

		__host__ __device__ float	x() const { return (this->e[0]); }
		__host__ __device__ float	y() const { return (this->e[1]); }
		__host__ __device__ float	z() const { return (this->e[2]); }
		__host__ __device__ float	r() const { return (this->e[0]); }
		__host__ __device__ float	g() const { return (this->e[1]); }
		__host__ __device__ float	b() const { return (this->e[2]); }

		__host__ __device__ vec3	operator-() const { return (vec3(-this->e[0], -this->e[1], -this->e[2])); }
		__host__ __device__ float	operator[](int i) const { return (this->e[i]); }
		__host__ __device__ float	&operator[](int i) { return (this->e[i]); }

		__host__ __device__ vec3	&operator+=(const vec3 &v)
		{
			this->e[0] += v.e[0];
			this->e[1] += v.e[1];
			this->e[2] += v.e[2];
			return (*this);
		}

		__host__ __device__ vec3	&operator*=(const float t)
		{
			this->e[0] *= t;
			this->e[1] *= t;
			this->e[2] *= t;
			return (*this);
		}

		__host__ __device__ vec3	&operator/=(const float t)
		{
			this->e[0] /= t;
			this->e[1] /= t;
			this->e[2] /= t;
			return (*this);
		}

		__host__ __device__ float	length() const { return (std::sqrt(this->length_squared())); }
		__host__ __device__ float	length_squared() const
		{
			return (this->e[0] * this->e[0]
				+ this->e[1] * this->e[1]
				+ this->e[2] * this->e[2]);
		}
};

inline std::istream	&operator>>(std::istream &in, vec3 &v)
{
	return (in >> v.e[0] >> v.e[1] >> v.e[2]);
}

inline std::ostream	&operator<<(std::ostream &out, const vec3 &v)
{
	return (out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2]);
}

__host__ __device__ inline vec3	operator+(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]));
}

__host__ __device__ inline vec3	operator-(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]));
}

__host__ __device__ inline vec3	operator*(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]));
}

__host__ __device__ inline vec3	operator*(const vec3 &v, const float t)
{
	return (vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t));
}

__host__ __device__ inline vec3	operator/(const vec3 &v, const float t)
{
	return (vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t));
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
	return (u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
	return (vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]));
}

__host__ __device__ inline vec3	unit_vector(vec3 v)
{
	return (v / v.length());
}


#endif
