#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/utils/LoadObjFile.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <vector>
#include <string>
#include <time.h>

#include <fstream>
#include <sstream>

#include <iostream>
#include <optional>

#include "paths.hpp"

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

void saveToFile(std::string const& filename,
                std::size_t width,
                std::size_t height,
                std::vector<Colour> const& image);

// Declarations
class Camera;
class Shape;
class Sampler;
class Light;
class BRDF;
class Material;

struct Pixel {
	// struct used for assigning pixels to threads for multithreaded rendering
	unsigned int x, y;
};

struct Slab {
	Pixel start, end;
};

struct World
{
	World() {
		bbox_hit_calls = 0;
		shape_hit_calls = 0;
	}
    std::size_t width, height, max_depth;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
	long long int bbox_hit_calls;
	long long int shape_hit_calls;

	void build_world_pinhole_matte_point(bool multithread);
	void build_world_pinhole_matte_directional(bool multithread);
	void build_world_fisheye_matte(bool multithread);
	void build_world_pinhole_specular(bool multithread);
	void build_world_mirror_reflection(bool ao, bool area_instead_of_point);
	void build_world_multithread();
	void build_world_regular_grid();
	void build_world_mesh(std::string mesh_filename, std::string mesh_fileRoot);
	void build_test();
	void build_bvh();
	void build_world_bvh(int num_clumps);
};

struct ShadeRec
{
	ShadeRec() {
		bbox_hit_calls = 0;
		shape_hit_calls = 0;
	}

	glm::mat4 inv_modelMat;
	Colour color;
    float t;
	size_t depth;
	atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
	int bbox_hit_calls = 0;
	int shape_hit_calls = 0;
};

// Abstract classes defining the interfaces for concrete entities

std::string vec3_to_string(atlas::math::Vector vec) {
	return "(" + std::to_string(vec.x) + "," + std::to_string(vec.y) + "," + std::to_string(vec.z) + ")";
}

float rand_f() {
	return (float)rand() / (float)RAND_MAX;
}

float rand_f(float min, float max) {
	float r = rand_f();
	return min + r * (max - min);
}

class Camera
{
public:
    Camera();

    virtual ~Camera() = default;

    virtual void renderScene(World& world) const = 0;

	virtual void renderScene_multithreaded(World& world) const = 0;

    void setEye(atlas::math::Point const& eye);

    void setLookAt(atlas::math::Point const& lookAt);

    void setUpVector(atlas::math::Vector const& up);

    void computeUVW();

protected:
    atlas::math::Point mEye;
    atlas::math::Point mLookAt;
    atlas::math::Point mUp;
    atlas::math::Vector mU, mV, mW;
};

class Pinhole : public Camera {
public:
	Pinhole(float distance_to_screen) :
		d{distance_to_screen}
	{}
	
	void renderScene(World& world) const;

	void renderPixel(World& world, Pixel& pixel, bool count_hit_calls) const;

	void renderScene_multithreaded(World& world) const;
	
protected:
	float d;
};

class Orthographic : public Camera {
public:
	Orthographic(float distance_to_screen) :
		d{ distance_to_screen } {
	}

	void renderScene(World& world) const;

	void renderPixel(World& world, Pixel& pixel, bool count_hit_calls) const;

	void renderScene_multithreaded(World& world) const;

protected:
	float d;
};

class FishEye : public Camera {
// from Ray Tracing From the Ground Up
public:
	FishEye(float fov) :
		psi_max{fov}
	{};
	
	atlas::math::Vector ray_direction(atlas::math::Point& p, int hres, int vres,
								float s, float& r) const;
	
	void renderScene(World& world) const;

	void renderScene_multithreaded(World& world) const;
	
protected:
	float psi_max; // fov in degrees
};

class Sampler
{
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndeces();

    virtual void generateSamples() = 0;

    atlas::math::Point sampleUnitSquare();

	atlas::math::Point sampleHemisphere();

	void map_samples_to_hemisphere(const float exp);

protected:
    std::vector<atlas::math::Point> mSamples;
	std::vector<atlas::math::Point> hemisphere_samples;
    std::vector<int> mShuffledIndeces;

    int mNumSamples;
    int mNumSets;
    unsigned long mCount;
    int mJump;
};

class Regular : public Sampler
{
public:
    Regular(int numSamples, int numSets);

    void generateSamples();
};

class Random : public Sampler
{
public:
    Random(int numSamples, int numSets);

    void generateSamples();
};

class MultiJittered : public Sampler
{
public:
    MultiJittered(int numSamples, int numSets);

    void generateSamples();
};

class BRDF
{
public:
    virtual ~BRDF() = default;

    virtual Colour fn(ShadeRec const& sr,
                      atlas::math::Vector const& reflected,
                      atlas::math::Vector const& incoming) const   = 0;
    virtual Colour rho(ShadeRec const& sr,
                       atlas::math::Vector const& reflected) const = 0;
	virtual void set_sampler(std::shared_ptr<Sampler>& sampler_, float exp_) {
		(void)exp_;
		sampler = sampler_;
	}
protected:
	std::shared_ptr<Sampler> sampler;
};

class Lambertian : public BRDF {
public:
	Lambertian(float kd_) :
		kd{kd_}
	{};
	Lambertian() :
		kd{0.8f}
	{};
	
	Colour fn(ShadeRec const& sr,
                atlas::math::Vector const& reflected,
                atlas::math::Vector const& incoming) const {
		return rho(sr,reflected) * glm::one_over_pi<float>() * glm::dot(incoming, sr.normal);
	};
					  
	Colour rho(ShadeRec const& sr,
                atlas::math::Vector const& reflected) const {
		(void)reflected;
		return kd*sr.color;
	}
	
private:
	float kd;
};

class GlossySpecular : public BRDF {
public:
	GlossySpecular(float k,float e) :
		ks{k},
		exp{e}
	{};
	
	GlossySpecular() :
		ks{0.9f},
		exp{2.0f}
	{};
	
	Colour fn(ShadeRec const& sr,
                atlas::math::Vector const& reflected,
                atlas::math::Vector const& incoming) const {
		Colour L{0,0,0};
		float ndotwi = glm::dot(sr.normal,incoming);
		atlas::math::Vector r = (-1.0f*incoming + 2.0f * sr.normal * ndotwi);
		float rdotwo = glm::dot(r,reflected);
		
		if (rdotwo > 0.0f) {
			L = Colour{1,1,1} * (ks * pow(rdotwo, exp));
			//printf("wi:<%f,%f,%f>\tn:<%f,%f,%f>\two:<%f,%f,%f>\tr:<%f,%f,%f>\n",incoming.x,incoming.y,incoming.z,sr.normal.x,sr.normal.y,sr.normal.z,reflected.x,reflected.y,reflected.z,r.x,r.y,r.z);
		}
		
		return L;
	};
	
	Colour rho(ShadeRec const& sr,
                atlas::math::Vector const& reflected) const {
		// not sure which part of fn should be in here bcz the only time reflected is used
		// is with r which requires incoming which isn't passed to this function?
		(void)sr;
		(void)reflected;
		return Colour{0,0,0};
	};

	void set_sampler(std::shared_ptr<Sampler> sampler_, float exp_) {
		sampler = sampler_;
		sampler->map_samples_to_hemisphere(exp_);
	}

	Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wi, float& pdf) {
		float ndotwo = glm::dot(sr.normal, wo);
		atlas::math::Vector r = -wo + 2.0f * sr.normal * ndotwo; // direction of mirror reflection

		atlas::math::Vector w = r;
		atlas::math::Vector u = glm::normalize(glm::cross(atlas::math::Vector(0.00424f, 1.0f, 0.00764f), w));
		atlas::math::Vector v = glm::cross(u, v);

		atlas::math::Point sp = sampler->sampleHemisphere();
		wi = sp.x * u + sp.y * v + sp.z * w; // reflected ray direction

		if (glm::dot(sr.normal, wi) < 0.0f)
			wi = -sp.x * u - sp.y * v + sp.z * w;

		float phong_lobe = pow(glm::dot(r, wi), exp);
		pdf = phong_lobe * glm::dot(sr.normal, wi);

		return ks * Colour(1.0f) * phong_lobe;
	};
	
private:
	float ks, exp;
};

class PerfectSpecular : public BRDF {
public:
	PerfectSpecular() : kr(1.0) {};
	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const {
		(void)sr;
		(void)reflected;
		(void)incoming;
		return Colour{ 0,0,0 };
	};
	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const {
		(void)reflected;
		(void)sr;
		return Colour{ 0,0,0 };
	};
	Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wi) {
		float ndotwo = glm::dot(sr.normal, wo);
		wi = -wo + 2.0f * sr.normal * ndotwo;

		return (kr * sr.color / (glm::dot(sr.normal,wi)));
	}
private:
	float kr;
};

class BTDF {
	virtual Colour sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wt) = 0;

	virtual bool tir(const ShadeRec& sr) = 0;
};

class PerfectTransmitter : public BTDF {
	// inspired by Ray Tracing from the Ground Up
	// and https://github.com/terraritto/RayTracing
public:
	PerfectTransmitter() {
		ior = 1.0f;
		kt = 1.0f;
	}
	PerfectTransmitter(float ior_, float kt_) {
		ior = ior_;
		kt = kt_;
	}

	Colour sample_f(const ShadeRec& sr, const atlas::math::Vector& wo, atlas::math::Vector& wt) {
		atlas::math::Vector n = sr.normal;
		float cos_thetai = glm::dot(n, wo);
		float eta = ior;

		if (cos_thetai < 0.0f) {
			cos_thetai = -cos_thetai;
			n = -n;
			eta = 1.0f / eta;
		}

		float temp = 1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta);
		float cos_theta2 = sqrt(temp);
		wt = -wo / eta - (cos_theta2 - cos_thetai / eta) * n;

		return (kt / (eta * eta) * Colour(1.0f, 1.0f, 1.0f) / fabs(glm::dot(sr.normal,wt)));
	}

	bool tir(const ShadeRec& sr) {
		atlas::math::Vector wo = -sr.ray.d;
		float cos_thetai = glm::dot(sr.normal,wo);
		float eta = ior;

		if (cos_thetai < 0.0f) {
			eta = 1.0f / eta;
		}

		return (1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta) < 0.0f);
	}

private:
	float ior, kt;
};

class FresnelTranmitter : public BTDF {
	// inspired by Ray Tracing from the Ground Up
	// and https://github.com/terraritto/RayTracing

};

class Material
{
public:
    virtual ~Material() = default;

    virtual Colour shade(ShadeRec& sr, World& world) = 0;
	
	void addBRDF(std::shared_ptr<BRDF> brdf) {
		brdfs.push_back(brdf);
	};
	
	std::vector<std::shared_ptr<BRDF>> getBRDFs() {
		return brdfs;
	};

	virtual Colour get_Le(ShadeRec& sr) const {
		(void)sr;
		return Colour(0.0f);
	}
	
private:
	std::vector<std::shared_ptr<BRDF>> brdfs;
};

class Matte : public Material {
public:
	Matte() {
		addBRDF(std::make_shared<Lambertian>(0.8f));
	};
	
	Colour shade(ShadeRec& sr, World& world);

private:
};

class Phong : public Material {
public:
	Phong() {
		addBRDF(std::make_shared<Lambertian>(0.6f));
		addBRDF(std::make_shared<GlossySpecular>(0.8f,5.0f));
	};
	Phong(float kd, float ks, float exp) {
		addBRDF(std::make_shared<Lambertian>(kd));
		addBRDF(std::make_shared<GlossySpecular>(ks, exp));
	};
	
	Colour shade(ShadeRec& sr, World& world);
};

class Reflective : public Phong {
public:
	Reflective()
		: Phong(), reflective_brdf(new PerfectSpecular)
	{};

	Colour shade(ShadeRec& sr, World& world);
private:
	PerfectSpecular* reflective_brdf;
};

class GlossyReflector : public Phong {
public:
	GlossyReflector(std::shared_ptr<Sampler> sampler, float exp)
		: Phong(0.0f,0.0f,1000.0f), glossy_specular_brdf(new GlossySpecular)
	{
		glossy_specular_brdf->set_sampler(sampler, exp);
	};
	Colour shade(ShadeRec& sr, World& world);
private:
	GlossySpecular* glossy_specular_brdf;
};

class Emissive : public Material {
public:
	Emissive(Colour clr, float radiance_factor)
		: ce(clr), ls(radiance_factor) {}
	Colour shade(ShadeRec& sr, World& world);
	Colour get_Le(ShadeRec& sr) const;
private:
	float ls; // radiance scaling factor
	Colour ce;
};

class Transparent : public Phong {
	// inspired by Ray Tracing from the Ground Up
	// and https://github.com/terraritto/RayTracing
public:
	Transparent() : Phong() {
		reflective_brdf = std::make_shared<PerfectSpecular>();
		specular_btdf = std::make_shared<PerfectTransmitter>();
	}

	virtual Colour shade(ShadeRec& sr, World& world);

private:
	Colour trace_ray(ShadeRec& sr, World& world, atlas::math::Ray<atlas::math::Vector>& ray);
	std::shared_ptr<PerfectSpecular> reflective_brdf;
	std::shared_ptr<PerfectTransmitter> specular_btdf;
};

class BBox {
	// from Ray Tracing from the Ground Up
public:
	float x0, x1, y0, y1, z0, z1;

	BBox()
		: x0(-1.0f), x1(1.0f), y0(-1.0f), y1(1.0f), z0(-1.0f), z1(1.0f) {}	

	BBox(float x0_, float x1_, float y0_, float y1_, float z0_, float z1_)
		: x0(x0_), x1(x1_), y0(y0_), y1(y1_), z0(z0_), z1(z1_) {}

	BBox(atlas::math::Point const p0, atlas::math::Point const p1)
		: x0(p0.x), x1(p1.x), y0(p0.y), y1(p1.y), z0(p0.z), z1(p1.z) {}

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) {
		sr.bbox_hit_calls += 1;
		const float kEpsilon{ 0.01f };
		float ox = ray.o.x; float oy = ray.o.y; float oz = ray.o.z;
		float dx = ray.d.x; float dy = ray.d.y; float dz = ray.d.z;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		float a = 1.0f / dx;
		if (a >= 0) {
			tx_min = (x0 - ox) * a;
			tx_max = (x1 - ox) * a;
		} else {
			tx_min = (x1 - ox) * a;
			tx_max = (x0 - ox) * a;
		}

		float b = 1.0f / dy;
		if (b >= 0) {
			ty_min = (y0 - oy) * b;
			ty_max = (y1 - oy) * b;
		} else {
			ty_min = (y1 - oy) * b;
			ty_max = (y0 - oy) * b;
		}

		float c = 1.0f / dz;
		if (c >= 0) {
			tz_min = (z0 - oz) * c;
			tz_max = (z1 - oz) * c;
		} else {
			tz_min = (z1 - oz) * c;
			tz_max = (z0 - oz) * c;
		}

		float t0, t1;

		// find largest entering t value

		if (tx_min > ty_min)
			t0 = tx_min;
		else
			t0 = ty_min;

		if (tz_min > t0)
			t0 = tz_min;

		// find smallest exiting t value

		if (tx_max < ty_max)
			t1 = tx_max;
		else
			t1 = ty_max;

		if (tz_max < t1)
			t1 = tz_max;

		return (t0 < t1 && t1 > kEpsilon);
	}

	bool inside(atlas::math::Point const& p) {
		return ((p.x > x0&& p.x < x1) && (p.y > y0&& p.y < y1) && (p.z > z0&& p.z < z1));
	}
};

class Shape
{
public:
    Shape();
    virtual ~Shape() = default;

    // if t computed is less than the t in sr, it and the color should be updated in sr
    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
                     ShadeRec& sr) = 0;
					 
	virtual std::string type_str() const { return "none\n";  }

    void setColour(Colour const& col);

    Colour getColour() const;

	virtual atlas::math::Vector get_normal(const atlas::math::Point& p) {
		(void)p;
		printf("Error: only Rectangles can be used as area lights.\n");
		assert(false);
		return atlas::math::Vector(0.0f);
	}

	virtual float pdf(const ShadeRec& sr) {
		(void)sr;
		printf("Error: only Rectangles can be used as area lights.\n");
		assert(false);
		return 0.0f;
	}

	virtual atlas::math::Point sample(atlas::math::Point& sp) {
		printf("Error: only Rectangles can be used as area lights.\n");
		assert(false);
		return sp;
	}
	
	void setMaterial(std::shared_ptr<Material> const& material) {
		mMaterial = material;
	}

    std::shared_ptr<Material> getMaterial() const {
		return mMaterial;
	}

	virtual bool hasBBox() { return true; }

	virtual BBox get_bounding_box() { return BBox(); }

	virtual void add_object(std::shared_ptr<Shape> obj) {}

	virtual atlas::math::Point get_center() {
		printf("Cannot use get_center() on a shape that isn't a BVHnode.\n");
		assert(false);
		return atlas::math::Point(0.0f);
	}

	virtual float distance(std::shared_ptr<Shape> n2) {
		(void)n2;
		printf("Cannot use distance() on a shape that isn't a BVHnode.\n");
		assert(false);
		return 0.0f;
	}

protected:
    virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                              float& tMin) const = 0;

    Colour mColour;
	std::shared_ptr<Material> mMaterial;
};

// Concrete classes which we can construct and use in our ray tracer

class Rectangle : public Shape {
public:
	Rectangle(const atlas::math::Point& p, const atlas::math::Vector& _a, const atlas::math::Vector& _b, Colour clr) 
		:	p0(p),
			a(_a),
			b(_b)
	{
		mColour = clr;
		auto temp1 = glm::length(_a);
		a_len_squared = temp1 * temp1;
		auto temp2 = glm::length(_b);
		b_len_squared = temp2 * temp2;
		area = temp1 * temp2;
		inv_area = 1.0f / area;
		normal = glm::normalize(glm::cross(_a, _b));
	}

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) {
		sr.shape_hit_calls += 1;
		float t{};
		bool intersect{ intersectRay(ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t) {
			sr.normal = normal;
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
		}
		return intersect;
	}

	std::string type_str() const {
		return "rectangle";
	};

	atlas::math::Vector get_normal(const atlas::math::Point& p) {
		(void)p;
		return normal;
	}

	float pdf(const ShadeRec& sr) {
		(void)sr;
		return inv_area;
	}

	atlas::math::Point sample(atlas::math::Point& sp) {
		return p0 + sp.x * a + sp.y * b;
	}

	BBox get_bounding_box() {
		float delta = 0.0001f;

		return(BBox(glm::min(p0.x, p0.x + a.x + b.x) - delta, glm::max(p0.x, p0.x + a.x + b.x) + delta,
			glm::min(p0.y, p0.y + a.y + b.y) - delta, glm::max(p0.y, p0.y + a.y + b.y) + delta,
			glm::min(p0.z, p0.z + a.z + b.z) - delta, glm::max(p0.z, p0.z + a.z + b.z) + delta));
	}

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const {
		const float kEpsilon{ 0.01f };

		float t = glm::dot((p0 - ray.o), normal) / glm::dot(ray.d, normal);

		if (t <= kEpsilon)
			return false;

		atlas::math::Point p = ray.o + t * ray.d;
		atlas::math::Vector d = p - p0;

		float ddota = glm::dot(d, a);

		if (ddota < 0.0f || ddota > a_len_squared)
			return false;

		float ddotb = glm::dot(d, b);

		if (ddotb < 0.0f || ddotb > b_len_squared)
			return false;

		tMin = t;
		return true;
	}

	atlas::math::Point p0;   			// corner vertex 
	atlas::math::Vector a;				// side
	atlas::math::Vector b;				// side
	float a_len_squared;				// square of the length of side a
	float b_len_squared;				// square of the length of side b
	atlas::math::Vector normal;

	float area;			// for rectangular lights
	float inv_area;		// for rectangular lights
};

class Plane : public Shape {
public:
	Plane(atlas::math::Point p0, atlas::math::Point p1, atlas::math::Point p2, Colour colour);
	
	Plane(atlas::math::Vector normal, atlas::math::Point p, Colour colour);
	
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
             ShadeRec& sr);
			 
	std::string type_str() const {
		return "plane";
	};

	bool hasBBox() { return false; }
	
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const;
					  
	atlas::math::Vector normal_;
	atlas::math::Point p_;
};

class Triangle : public Shape {
public:
	Triangle(atlas::math::Vector v0, atlas::math::Vector v1, atlas::math::Vector v2, Colour colour);
	
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
             ShadeRec& sr);
			 
	std::string type_str() const {
		return "triangle";
	};

	void print() {
		printf("(%f, %f, %f)  (%f, %f, %f)  (%f, %f, %f)\n", v0_.x, v0_.y, v0_.z, v1_.x, v1_.y, v1_.z, v2_.x, v2_.y, v2_.z);
	}

	BBox get_bounding_box() {
		float delta = 0.000001f;

		float min_x = std::min(std::min(v0_.x, v1_.x), v2_.x) - delta;
		float max_x = std::max(std::max(v0_.x, v1_.x), v2_.x) + delta;
										  		 		 
		float min_y = std::min(std::min(v0_.y, v1_.y), v2_.y) - delta;
		float max_y = std::max(std::max(v0_.y, v1_.y), v2_.y) + delta;
										  		 		 
		float min_z = std::min(std::min(v0_.z, v1_.z), v2_.z) - delta;
		float max_z = std::max(std::max(v0_.z, v1_.z), v2_.z) + delta;

		return BBox(min_x, max_x, min_y, max_y, min_z, max_z);
	}
			 
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const;
					  
	atlas::math::Vector v0_, v1_, v2_;
	atlas::math::Vector normal_;
};

class Box : public Shape {
public:
	Box(atlas::math::Point p0, atlas::math::Point p1, Colour clr);
	
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
             ShadeRec& sr);
	
	std::string type_str() const {
		return "box";
	};
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const;
					  
	atlas::math::Vector get_normal(const int face_hit) const;
	
	float x0,x1,y0,y1,z0,z1;
	
};

class Sphere : public Shape
{
public:
    Sphere(atlas::math::Point center, float radius);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
             ShadeRec& sr);

	std::string type_str() const {
		return "sphere";
	};

	BBox get_bounding_box() {
		atlas::math::Point p0 = mCentre - mRadius;
		atlas::math::Point p1 = mCentre + mRadius;

		float delta = 0.0001f;

		return(BBox(glm::min(p0.x, p1.x) - delta, glm::max(p0.x, p1.x) + delta,
			glm::min(p0.y, p1.y) - delta, glm::max(p0.y, p1.y) + delta,
			glm::min(p0.z, p1.z) - delta, glm::max(p0.z, p1.z) + delta));
	}

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const;
					  
	atlas::math::Point mCentre;
    float mRadius;
    float mRadiusSqr;
	glm::mat4 modelMat, inv_modelMat;
};

class Compound : public Shape {
public:
	Compound() {}

	virtual void set_material(std::shared_ptr<Material>& material_ptr) {
		int num_objects = (int)objects.size();

		for (int j = 0; j < num_objects; j++)
			objects[j]->setMaterial(material_ptr);
	}

	virtual void add_object(std::shared_ptr<Shape> object_ptr) {
		objects.push_back(object_ptr);
	}

	int get_num_objects() {
		return (int)objects.size();
	}

	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) {
		bool hit = false;
		int num_objects = (int)objects.size();

		for (int j = 0; j < num_objects; j++)
			if (objects[j]->hit(ray, sr) && sr.material) {
				hit = true;
				mMaterial = objects[j]->getMaterial();	// lhs is GeometricObject::material_ptr
			}

		return (hit);
	}

	std::vector<std::shared_ptr<Shape>> objects;
private:
	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const { 
		(void)ray;
		(void)tMin;
		return false;
	}
};

class Grid : public Compound {
	// from Ray Tracing from the Ground Up
public:
	Grid() {}

	virtual BBox get_bounding_box() {
		return bbox;
	}

	void setup_cells() {
		atlas::math::Point p0 = min_coordinates();
		atlas::math::Point p1 = max_coordinates();

		bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
		bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p1.z;

		int num_objects = (int)objects.size();
		float wx = p1.x - p0.x;
		float wy = p1.y - p0.y;
		float wz = p1.z - p0.z;
		float multiplier = 2.0f;
		float s = std::pow(wx * wy * wz / num_objects, 0.3333333f);
		nx = (int)(multiplier * wx / s + 1);
		ny = (int)(multiplier * wy / s + 1);
		nz = (int)(multiplier * wz / s + 1);

		int num_cells = nx * ny * nz;
		cells.reserve(num_objects);

		for (int j = 0; j < num_cells; j++) {
			cells.push_back(NULL);
		}

		std::vector<int> counts;
		counts.reserve(num_cells);

		for (int j = 0; j < num_cells; j++) {
			counts.push_back(0);
		}

		BBox obj_bbox;
		int index;

		for (int j = 0; j < num_objects; j++) {
			obj_bbox = objects[j]->get_bounding_box();

			int ixmin = (int)clamp((float)(obj_bbox.x0 - p0.x) * nx / (p1.x - p0.x), (float)0, (float)(nx - 1));
			int iymin = (int)clamp((float)(obj_bbox.y0 - p0.y) * ny / (p1.y - p0.y), (float)0, (float)(ny - 1));
			int izmin = (int)clamp((float)(obj_bbox.z0 - p0.z) * nz / (p1.z - p0.z), (float)0, (float)(nz - 1));
			int ixmax = (int)clamp((float)(obj_bbox.x1 - p0.x) * nx / (p1.x - p0.x), (float)0, (float)(nx - 1));
			int iymax = (int)clamp((float)(obj_bbox.y1 - p0.y) * ny / (p1.y - p0.y), (float)0, (float)(ny - 1));
			int izmax = (int)clamp((float)(obj_bbox.z1 - p0.z) * nz / (p1.z - p0.z), (float)0, (float)(nz - 1));

			for (int iz = izmin; iz <= izmax; iz++) {
				for (int iy = iymin; iy <= iymax; iy++) {
					for (int ix = ixmin; ix <= ixmax; ix++) {
						index = ix + nx * iy + nx * ny * iz;

						if (counts[index] == 0) {
							cells[index] = objects[j];
							counts[index] += 1;
						} else {
							if (counts[index] == 1) {
								std::shared_ptr<Compound> compound_ptr = std::make_shared<Compound>();
								compound_ptr->add_object(cells[index]);
								compound_ptr->add_object(objects[j]);

								cells[index] = compound_ptr;

								counts[index] += 1;
							} else {
								cells[index]->add_object(objects[j]);
								counts[index] += 1;
							}
						}
					}
				}
			}
		}
		objects.erase(objects.begin(), objects.end());
		counts.erase(counts.begin(), counts.end());
	}

	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) {
		float ox = ray.o.x;
		float oy = ray.o.y;
		float oz = ray.o.z;
		float dx = ray.d.x;
		float dy = ray.d.y;
		float dz = ray.d.z;

		float x0 = bbox.x0;
		float y0 = bbox.y0;
		float z0 = bbox.z0;
		float x1 = bbox.x1;
		float y1 = bbox.y1;
		float z1 = bbox.z1;

		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;

		// the following code includes modifications from Shirley and Morley (2003)

		float a = 1.0f / dx;
		if (a >= 0) {
			tx_min = (x0 - ox) * a;
			tx_max = (x1 - ox) * a;
		} else {
			tx_min = (x1 - ox) * a;
			tx_max = (x0 - ox) * a;
		}

		float b = 1.0f / dy;
		if (b >= 0) {
			ty_min = (y0 - oy) * b;
			ty_max = (y1 - oy) * b;
		} else {
			ty_min = (y1 - oy) * b;
			ty_max = (y0 - oy) * b;
		}

		float c = 1.0f / dz;
		if (c >= 0) {
			tz_min = (z0 - oz) * c;
			tz_max = (z1 - oz) * c;
		} else {
			tz_min = (z1 - oz) * c;
			tz_max = (z0 - oz) * c;
		}

		float t0, t1;

		if (tx_min > ty_min)
			t0 = tx_min;
		else
			t0 = ty_min;

		if (tz_min > t0)
			t0 = tz_min;

		if (tx_max < ty_max)
			t1 = tx_max;
		else
			t1 = ty_max;

		if (tz_max < t1)
			t1 = tz_max;

		if (t0 > t1)
			return(false);


		// initial cell coordinates

		int ix, iy, iz;

		if (bbox.inside(ray.o)) {  			// does the ray start inside the grid?
			ix = (int)clamp((float)((float)(ox - x0) * (float)nx / (float)(x1 - x0)), 0.0f, (float)(nx - 1));
			iy = (int)clamp((float)((float)(oy - y0) * (float)ny / (float)(y1 - y0)), 0.0f, (float)(ny - 1));
			iz = (int)clamp((float)((float)(oz - z0) * (float)nz / (float)(z1 - z0)), 0.0f, (float)(nz - 1));
		} else {
			atlas::math::Point p = ray.o + t0 * ray.d;  // initial hit point with grid's bounding box
			ix = (int)clamp((float)((p.x - x0) * nx / (x1 - x0)), 0.0f, (float)(nx - 1));
			iy = (int)clamp((float)((p.y - y0) * ny / (y1 - y0)), 0.0f, (float)(ny - 1));
			iz = (int)clamp((float)((p.z - z0) * nz / (z1 - z0)), 0.0f, (float)(nz - 1));
		}

		// ray parameter increments per cell in the x, y, and z directions

		float dtx = (tx_max - tx_min) / nx;
		float dty = (ty_max - ty_min) / ny;
		float dtz = (tz_max - tz_min) / nz;

		float 	tx_next, ty_next, tz_next;
		int 	ix_step, iy_step, iz_step;
		int 	ix_stop, iy_stop, iz_stop;

		if (dx > 0) {
			tx_next = tx_min + (ix + 1) * dtx;
			ix_step = +1;
			ix_stop = nx;
		} else {
			tx_next = tx_min + (nx - ix) * dtx;
			ix_step = -1;
			ix_stop = -1;
		}

		if (dx == 0.0f) {
			tx_next = std::numeric_limits<float>::max();
			ix_step = -1;
			ix_stop = -1;
		}


		if (dy > 0) {
			ty_next = ty_min + (iy + 1) * dty;
			iy_step = +1;
			iy_stop = ny;
		} else {
			ty_next = ty_min + (ny - iy) * dty;
			iy_step = -1;
			iy_stop = -1;
		}

		if (dy == 0.0f) {
			ty_next = std::numeric_limits<float>::max();
			iy_step = -1;
			iy_stop = -1;
		}

		if (dz > 0) {
			tz_next = tz_min + (iz + 1) * dtz;
			iz_step = +1;
			iz_stop = nz;
		} else {
			tz_next = tz_min + (nz - iz) * dtz;
			iz_step = -1;
			iz_stop = -1;
		}

		if (dz == 0.0f) {
			tz_next = std::numeric_limits<float>::max();
			iz_step = -1;
			iz_stop = -1;
		}


		// traverse the grid

		while (true) {
			std::shared_ptr<Shape> object_ptr = cells[ix + nx * iy + nx * ny * iz];

			if (tx_next < ty_next && tx_next < tz_next) {
				if (object_ptr && object_ptr->hit(ray, sr) && sr.t < tx_next) {
					mMaterial = object_ptr->getMaterial();
					return (true);
				}

				tx_next += dtx;
				ix += ix_step;

				if (ix == ix_stop)
					return (false);
			} else {
				if (ty_next < tz_next) {
					if (object_ptr && object_ptr->hit(ray, sr) && sr.t < ty_next) {
						mMaterial = object_ptr->getMaterial();
						return (true);
					}

					ty_next += dty;
					iy += iy_step;

					if (iy == iy_stop)
						return (false);
				} else {
					if (object_ptr && object_ptr->hit(ray, sr) && sr.t < tz_next) {
						mMaterial = object_ptr->getMaterial();
						return (true);
					}

					tz_next += dtz;
					iz += iz_step;

					if (iz == iz_stop)
						return (false);
				}
			}
		}
	}



private:
	std::vector<std::shared_ptr<Shape>> cells;
	BBox bbox;
	int nx, ny, nz;
	
	atlas::math::Point min_coordinates() {
		BBox bbox1;
		atlas::math::Point p0(std::numeric_limits<float>::max());
		const float kEpsilon{ 0.01f };

		int num_objects = (int)objects.size();

		for (int j = 0; j < num_objects; j++) {
			bbox1 = objects[j]->get_bounding_box();

			if (bbox1.x0 < p0.x)
				p0.x = bbox1.x0;
			if (bbox1.y0 < p0.y)
				p0.y = bbox1.y0;
			if (bbox1.z0 < p0.z)
				p0.z = bbox1.z0;
		}

		p0.x -= kEpsilon; p0.y -= kEpsilon; p0.z -= kEpsilon;

		return p0;
	}

	atlas::math::Point max_coordinates() {
		BBox bbox1;
		atlas::math::Point p0(-std::numeric_limits<float>::max());
		const float kEpsilon{ 0.01f };

		int num_objects = (int)objects.size();

		for (int j = 0; j < num_objects; j++) {
			bbox1 = objects[j]->get_bounding_box();

			if (bbox1.x0 > p0.x)
				p0.x = bbox1.x0;
			if (bbox1.y0 > p0.y)
				p0.y = bbox1.y0;
			if (bbox1.z0 > p0.z)
				p0.z = bbox1.z0;
		}

		p0.x += kEpsilon; p0.y += kEpsilon; p0.z += kEpsilon;

		return p0;
	}

	inline float clamp(float x, float min, float max) {
		return (x < min ? min : (x > max ? max : x));
	}
};

class Light {
public:
	virtual atlas::math::Vector getDirection(atlas::math::Point& hitPoint) = 0;

	void scaleRadiance(float b);

	void setColour(Colour const& c);

	virtual Colour L(ShadeRec& sr, World& world);

	virtual bool isAmbient() {
		return false;
	};

	virtual bool isArea() {
		return false;
	};

	virtual float G(ShadeRec const& sr) const {
		(void)sr;
		return 1; // only different with AreaLight
	}

	virtual float pdf(ShadeRec const& sr) const {
		(void)sr;
		return 1; // only different with AreaLight
	}

	virtual atlas::math::Point get_sample_point() {
		printf("Only area lights should call get_sample_point().\n");
		assert(false);
		return atlas::math::Point(0.0f);
	}

	virtual void set_sampler(std::shared_ptr<Sampler> sampler_) {
		sampler = sampler_;
	}

protected:
	Colour mColour;
	float mRadiance;
	std::shared_ptr<Sampler> sampler;
};

class AreaLight : public Light {
public:
	AreaLight(std::shared_ptr<Shape>& obj, std::shared_ptr<Material>& mat, std::shared_ptr<Sampler> sampler_) {
		object_ptr = obj;
		material_ptr = mat;
		set_sampler(sampler_);
	}
	atlas::math::Vector getDirection(atlas::math::Point& hitPoint) {
		sample_point = get_sample();
		light_normal = object_ptr->get_normal(sample_point);
		wi = glm::normalize(sample_point - hitPoint);
		//printf("sp: %s | hit: %s | wi: %s norm: %s\n", vec3_to_string(sample_point).c_str(), vec3_to_string(hitPoint).c_str(), vec3_to_string(wi).c_str(), vec3_to_string(light_normal).c_str());
		return wi;
	}
	Colour L(ShadeRec& sr, World& world) {
		(void)world;
		(void)sr;
		float ndotd = glm::dot(-light_normal, wi);
		if (ndotd > 0.0f) {
			return material_ptr->get_Le(sr);
		} else {
			return Colour(0.0f);
		}
	}
	float G(ShadeRec const& sr) const {
		float ndotd = glm::dot(-light_normal, wi);
		float d2 = glm::distance(sample_point, sr.ray(sr.t));
		d2 = d2 * d2;
		//printf("(%f, %f, %f) (%f, %f, %f)\n", sample_point.x, sample_point.y, sample_point.z, sr.ray(sr.t).x, sr.ray(sr.t).y, sr.ray(sr.t).z);
		//printf("%f %f\n", ndotd, d2);
		return ndotd / d2;
	}
	float pdf(ShadeRec const& sr) const {
		return object_ptr->pdf(sr);
	}
	bool isArea() {
		return true;
	};
	atlas::math::Point get_sample_point() {
		return sample_point;
	}

private:
	atlas::math::Point get_sample() {
		atlas::math::Point p = sampler->sampleUnitSquare();
		return object_ptr->sample(p);
	}
	std::shared_ptr<Shape> object_ptr;
	std::shared_ptr<Material> material_ptr;
	atlas::math::Point sample_point; // sample point on object surface
	atlas::math::Vector light_normal; // normal at sample point
	atlas::math::Vector wi; // unit vector from hit point to sample point
};

class Directional : public Light {
public:
	Directional(atlas::math::Vector d, Colour clr, float radiance) {
		mColour = clr;
		mRadiance = radiance;
		dir = glm::normalize(d);
	};

	Colour L(ShadeRec& sr, World& world);

	atlas::math::Vector getDirection(atlas::math::Point& hitPoint);

protected:
	atlas::math::Vector dir;
};

class Ambient : public Light {
public:
	Ambient(Colour clr, float radiance) {
		mColour = clr;
		mRadiance = radiance;
	}

	Colour L(ShadeRec& sr, World& world);

	bool isAmbient() {
		return true;
	};

	atlas::math::Vector getDirection(atlas::math::Point& hitPoint);
};

class PointLight : public Light {
public:
	PointLight(atlas::math::Point p1, Colour clr, float radiance) {
		mColour = clr;
		mRadiance = radiance;
		p = p1;
	}

	Colour L(ShadeRec& sr, World& world);

	atlas::math::Vector getDirection(atlas::math::Point& hitPoint);

private:
	atlas::math::Point p;
};

class AmbientOccluder : public Light {
public:
	AmbientOccluder(Colour clr, float radiance, Colour min_amnt) {
		mRadiance = radiance;
		mColour = clr;
		min_amount = min_amnt;
	}
	atlas::math::Vector getDirection(atlas::math::Point& hitPoint);
	Colour L(ShadeRec& sr, World& world);

	void set_sampler(std::shared_ptr<Sampler> sampler_) {
		sampler = sampler_;
		sampler->map_samples_to_hemisphere(1.0f);
	}

private:
	atlas::math::Point getHemisphereSample() {
		return sampler->sampleHemisphere();
	}
	atlas::math::Vector u, v, w;
	Colour min_amount;
};

class Mesh {
public:
	std::vector<atlas::math::Point> vertices;
	std::vector<std::vector<int>> faces;
	std::vector<atlas::math::Vector> normals;
	std::unordered_map<std::string, std::shared_ptr<Material>> materials;
	std::unordered_map<std::string, Colour> colours;
	glm::mat4 modelMat;
	glm::mat4 inv_modelMat;

	Mesh() {
		glm::mat4 temp(1.0f);
		modelMat = temp;
		glm::mat4 temp2(1.0f);
		inv_modelMat = temp;
	}

	void build_mesh(std::string& filename, std::string& fileRoot) {
		// code modified from a template by mauricio
		filename = fileRoot + filename;

		// loadObjMesh returns an std::optional, which is basically an easy way
		// of determining whether the load was successful. If you are using the
		// included material files, you can place them in the same directory as the
		// obj file and just pass that directory in as the second argument. If not,
		// then just pass the path to the file.
		std::optional<atlas::utils::ObjMesh> result = atlas::utils::loadObjMesh(filename, fileRoot);
		if (!result) {
			// The mesh failed to load.
			printf("Mesh failed to load.\n");
			return;
		}

		// The object is held in a state similar to a pointer, so dereference to get
		// the object back.
		atlas::utils::ObjMesh mesh = *result;

		// The ObjMesh class consists of the following data:
		// 1. A vector of shapes
		// 2. A vector of materials.
		//
		// The materials hold all of the diffuse, specular, and other rendering
		// properties of the objects. If you didn't pass any path to find the
		// material (or if it couldn't find them) there will only be a single
		// material with default values. For details on what values the struct
		// contains, you can find its definition in the tiny_obj_loader.h header
		// that gets downloaded automatically with Atlas (under _deps).
		std::vector<tinyobj::material_t> mats = mesh.materials;

		for (auto& shape : mesh.shapes) {
			// Each shape is conformed by the following:
			// 1. bool hasNormals: tells you whether the particular shape has
			// normals or not.
			// 2. bool hasTextureCoords: same thing but for texture coordinates.
			// 3. Vector of Vertices: the list of vertices, see below for the
			// contents of the utils::Vertex struct.
			// 4. Vector of indices: the indices of the faces.
			// 5. Vector of material ids. These are the indices you can use into the
			// material array we discussed earlier.
			// 6. Vector of smoothing groups. Not really important for this course.

			if (shape.hasNormals) {
				//fmt::print("Shape has normals\n");
			}

			if (shape.hasTextureCoords) {
				//fmt::print("Shape has texture coordinates\n");
			}
			/*
			for (auto& vertex : shape.vertices) {
				// The vertex struct contains:
				// 1. Position
				// 2. Normal (valid only if hasNormals is true)
				// 3. texCoord (valid only if hasTextureCoords is true)
				// 5. The index of the vertex (not related to the face indices).
				// 6. face The face id of the vertex (not needed).
				//
				// For the purposes of ray tracing, what you're most interested in
				// are the vertices, normals and texture coords (if applicable).
				fmt::print("position = ({}, {}, {})\n", vertex.position.x,
					vertex.position.y, vertex.position.z);
				fmt::print("normal = ({}, {}, {})\n", vertex.normal.x,
					vertex.normal.y, vertex.normal.z);
				fmt::print("texture coords = ({}, {})\n", vertex.texCoord.x,
					vertex.texCoord.y);
			}*/

			// You can build the triangles for each shape by iterating over every
			// triplet of indices from shape.indices and using those to access
			// shape.vertices to grab the corresponding positions, normals, and
			// texture coordinates.


			// my code
			for (auto& vertex : shape.vertices) {
				atlas::math::Point vert(vertex.position.x, vertex.position.y, vertex.position.z);
				atlas::math::Vector norm(vertex.normal.x, vertex.normal.y, vertex.normal.z);
				
				vertices.push_back(vert);
				normals.push_back(norm);
			}

			int num_indices = (int)shape.indices.size();
			for (int i = 0; i < num_indices; i += 3) {
				int i0 = (int)shape.indices[i];
				int i1 = (int)shape.indices[i + 1];
				int i2 = (int)shape.indices[i + 2];
				std::vector<int> face = { i0, i1, i2 };
				faces.push_back(face);
			}
		}
	}

	/*void build_mesh_and_material(char* mesh_filename, char* material_filename) {
		build_materials(material_filename);
		build_mesh(mesh_filename, true);
	}

	void build_materials(char* material_filename) {
		(void)material_filename;
	}

	void build_mesh(char* mesh_filename, bool materials_file) {
		(void)materials_file;
		printf("Parsing mesh file\n");

		std::string temp2(mesh_filename);

		std::ifstream infile("./../" + temp2);
		if (!infile.is_open()) {
			printf("Mesh file could not be opened. Make sure the file is located in the correct directory.\n");
			printf("File should be located one directory up from where this prorgram is being called.\n");
			printf("For example, running the program from my visual studio setup runs from the build folder.\n");
			printf("Therefore, I put the mesh file within the parent of the build folder.\n");
			assert(false);
		}
		std::string line;
		std::string word;

		while (std::getline(infile, line)) {
			std::istringstream iss(line, std::istringstream::in);
			if (line[0] == 'v' && line[1] == ' ') {
				iss >> word;
				int j = 0;
				atlas::math::Point vertex(0.0f);
				while (iss >> word) {
					vertex[j++] = std::stof(word);
				}
				vertices.push_back(vertex);
			} else if (line[0] == 'f' && line[1] == ' ') {
				iss >> word;
				std::vector<int> face;
				while (iss >> word) {
					//face[j++] = (int)std::stoi(word.substr(0, word.find("//"))) - 1;
					int temp = std::stoi(word);
					if (temp < 0) temp = (int)vertices.size() + temp;
					face.push_back(temp);
				}
				if ((int)face.size() == 4) {
					// turn rectangle into 2 triangles
					// vertices are top left, top right, bottom right, bottom left in cornell box obj
					faces.push_back(std::vector<int>({ face[0], face[1], face[2] }));
					faces.push_back(std::vector<int>({ face[0], face[2], face[3] }));
				} else {
					faces.push_back(face);
				}
			} else {
				continue;
			}
		}

		printf("Done converting mesh file into list of vertices and faces.\n");
		printf("Total polygons: %d\n", (int)faces.size());
	}*/
};

class FlatTriangle : public Shape {
public:
	FlatTriangle(std::shared_ptr<Mesh>& mesh, int face_i, Colour& clr, std::shared_ptr<Material>& mat) {
		mesh_ptr = mesh;
		mColour = clr;
		mMaterial = mat;
		v0_i = mesh_ptr->faces[face_i][0];
		v1_i = mesh_ptr->faces[face_i][1];
		v2_i = mesh_ptr->faces[face_i][2];

		atlas::math::Point v0 = mesh_ptr->vertices[v0_i];
		atlas::math::Point v1 = mesh_ptr->vertices[v1_i];
		atlas::math::Point v2 = mesh_ptr->vertices[v2_i];

		if (mesh_ptr->normals.size() > 0)
			normal = mesh_ptr->normals[face_i];
		else
			normal = -1.0f * glm::normalize(glm::cross((v1 - v0), (v2 - v0)));
	}
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) {
		float t{};
		sr.shape_hit_calls += 1;

		// convert ray from world space to model space
		glm::vec4 temp_o(ray.o.x, ray.o.y, ray.o.z, 1.0f);
		glm::vec4 temp_d(ray.d.x, ray.d.y, ray.d.z, 0.0f);
		temp_o = mesh_ptr->inv_modelMat * temp_o;
		temp_d = mesh_ptr->inv_modelMat * temp_d;

		atlas::math::Ray<atlas::math::Vector> model_ray;
		model_ray.o = temp_o;
		model_ray.d = temp_d;

		bool intersect{ intersectRay(model_ray, t) };

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t) {
			sr.normal = glm::transpose(mesh_ptr->inv_modelMat) * glm::vec4(normal,1.0f);
			sr.color = mColour;
			sr.t = t;
			sr.material = mMaterial;
		}
		return intersect;
	}

	BBox get_bounding_box() {
		float delta = 0.000001f;

		atlas::math::Point v0 = mesh_ptr->vertices[v0_i];
		atlas::math::Point v1 = mesh_ptr->vertices[v1_i];
		atlas::math::Point v2 = mesh_ptr->vertices[v2_i];

		float min_x = std::min(std::min(v0.x, v1.x), v2.x) - delta;
		float max_x = std::max(std::max(v0.x, v1.x), v2.x) + delta;

		float min_y = std::min(std::min(v0.y, v1.y), v2.y) - delta;
		float max_y = std::max(std::max(v0.y, v1.y), v2.y) + delta;

		float min_z = std::min(std::min(v0.z, v1.z), v2.z) - delta;
		float max_z = std::max(std::max(v0.z, v1.z), v2.z) + delta;

		atlas::math::Point p0(min_x, min_y, min_z);
		atlas::math::Point p1(max_x, max_y, max_z);

		p0 = mesh_ptr->modelMat * glm::vec4(p0, 1.0f);
		p1 = mesh_ptr->modelMat * glm::vec4(p1, 1.0f);

		return BBox(p0, p1);
	}
private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const {
		// from ray tracing from the ground up
		atlas::math::Point v0_ = mesh_ptr->vertices[v0_i];
		atlas::math::Point v1_ = mesh_ptr->vertices[v1_i];
		atlas::math::Point v2_ = mesh_ptr->vertices[v2_i];

		float a = v0_[0] - v1_[0], b = v0_[0] - v2_[0], c = ray.d[0], d = v0_[0] - ray.o[0];
		float e = v0_[1] - v1_[1], f = v0_[1] - v2_[1], g = ray.d[1], h = v0_[1] - ray.o[1];
		float i = v0_[2] - v1_[2], j = v0_[2] - v2_[2], k = ray.d[2], l = v0_[2] - ray.o[2];

		float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
		float q = g * i - e * k, s = e * j - f * i;

		float inv_denom = 1.0f / (a * m + b * q + c * s);
		float e1 = d * m - b * n - c * p;
		float beta = e1 * inv_denom;

		if (beta < 0.0f)
			return false;

		float r = e * l - h * i;
		float e2 = a * n + d * q + c * r;
		float gamma = e2 * inv_denom;

		if (gamma < 0.0f)
			return false;

		if (beta + gamma > 1.0f)
			return false;

		float e3 = a * p - b * r + d * s;
		float t = (float)(e3 * inv_denom);
		const float kEpsilon{ 0.01f };

		if (atlas::core::geq(t, kEpsilon)) {
			tMin = t;
			return true;
		}

		return false;
	}
	int v0_i, v1_i, v2_i;
	std::shared_ptr<Mesh> mesh_ptr;
	atlas::math::Vector normal; 
};

class BVHnode : public Shape {
public:
	BVHnode() {}

	BVHnode(std::shared_ptr<Shape>& obj) {
		leaf = true;
		shape = obj;
		bbox = obj->get_bounding_box();
		compute_center();
	}

	BVHnode(std::shared_ptr<BVHnode>& n1, std::shared_ptr<BVHnode>& n2) {
		leaf = false;
		children.push_back(n1);
		children.push_back(n2);

		compute_bbox();
		compute_center();
	}

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) {
		/*
		float t{};
		bool intersect{intersectRay(ray, t)};

		// update ShadeRec info about new closest hit
		if (intersect && t < sr.t)
		{
			sr.color = mColour;
			sr.t     = t;
			sr.normal = normal_;
			sr.material = mMaterial;
		}
		return intersect;
		*/
		if (bbox.hit(ray, sr)) {
			if (leaf) {
				return shape->hit(ray, sr);
			} else {
				bool hit = false;
				for (auto& node : children)
					hit |= node->hit(ray, sr);
				return hit;
			}
		}
		return false;
	}

	std::string type_str() const {
		return "bvhnode";
	};

	BBox get_bounding_box() {
		return bbox;
	}

	atlas::math::Point get_center() { return center; }

	bool isLeaf() { return leaf; }

	float distance(std::shared_ptr<Shape> n2) {
		atlas::math::Point center2 = n2->get_center();
		return glm::distance(center, center2);
	}

private:
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const {
		(void)tMin;
		(void)ray;
		return false;
	}

	void compute_bbox() {
		float minx = 0.0f, miny = 0.0f, minz = 0.0f, maxx = 0.0f, maxy = 0.0f, maxz = 0.0f;
		int i = 0;
		for (auto& node : children) {
			BBox nbox = node->get_bounding_box();
			if (i == 0) {
				i += 1;
				minx = nbox.x0;
				miny = nbox.y0;
				minz = nbox.z0;
				maxx = nbox.x1;
				maxy = nbox.y1;
				maxz = nbox.z1;
				if (node->isLeaf()) {
					if (maxx < minx) {
						maxx = minx;
						minx = nbox.x1;
					}
					if (maxy < miny) {
						maxy = miny;
						miny = nbox.y1;
					}
					if (maxz < minz) {
						maxz = minz;
						minz = nbox.z1;
					}
				}
			} else {
				if (nbox.x0 < minx)
					minx = nbox.x0;
				if (nbox.x1 > maxx)
					maxx = nbox.x1;

				if (nbox.y0 < miny)
					miny = nbox.y0;
				if (nbox.y1 > maxy)
					maxy = nbox.y1;

				if (nbox.z0 < minz)
					minz = nbox.z0;
				if (nbox.z1 > maxz)
					maxz = nbox.z1;

				if (node->isLeaf()) {
					if (nbox.x1 < minx)
						minx = nbox.x1;
					if (nbox.x0 > maxx)
						maxx = nbox.x0;

					if (nbox.y1 < miny)
						miny = nbox.y1;
					if (nbox.y0 > maxy)
						maxy = nbox.y0;

					if (nbox.z1 < minz)
						minz = nbox.z1;
					if (nbox.z0 > maxz)
						maxz = nbox.z0;
				}
			}
		}
		bbox = BBox(minx, maxx, miny, maxy, minz, maxz);
	}

	void compute_center() {
		float minx = bbox.x0;
		float maxx = bbox.x1;
		float miny = bbox.y0;
		float maxy = bbox.y1;
		float minz = bbox.z0;
		float maxz = bbox.z1;

		if (leaf) {
			// not guaranteed that min is always x0,y0,z0
			// like we are with an internal node
			minx = bbox.x0;
			miny = bbox.y0;
			minz = bbox.z0;
			maxx = bbox.x1;
			maxy = bbox.y1;
			maxz = bbox.z1;
			if (maxx < minx) {
				maxx = minx;
				minx = bbox.x1;
			}
			if (maxy < miny) {
				maxy = miny;
				miny = bbox.y1;
			}
			if (maxz < minz) {
				maxz = minz;
				minz = bbox.z1;
			}
		}

		center.x = minx + (maxx - minx) / 2;
		center.y = miny + (maxy - miny) / 2;
		center.z = minz + (maxz - minz) / 2;
	}

	std::shared_ptr<Shape> shape;
	std::vector<std::shared_ptr<BVHnode>> children;
	std::shared_ptr<BVHnode> parent;
	BBox bbox;
	atlas::math::Point center;
	bool leaf;
};

class Mapping {
public:
	virtual void get_texel_coordinates(const atlas::math::Point& local_hit_point, const int hres, const int vres, int& row, int& col) const = 0;
};

class SphericalMapping : public Mapping {
	void get_texel_coordinates(const atlas::math::Point& local_hit_point, const int hres, const int vres, int& row, int& col) const {
		float theta = acos(local_hit_point.y);
		float phi = atan2(local_hit_point.x, local_hit_point.z);
		if (phi < 0.0f)
			phi += glm::two_pi<float>();
		
		float u = phi * glm::one_over_two_pi<float>();
		float v = 1.0f - theta * glm::one_over_pi<float>();

		col = (int)((hres - 1) * u);
		row = (int)((vres - 1) * v);
	}
};

class Image {
	// took my earth ppm file from:
	// http://www.cs.utah.edu/~danielr/raytrace/hw5/
	// also my read_ppm_file was inspired by the same link
public:
	Image(std::string& filename) {
		read_ppm_file(filename);
	}

	void read_ppm_file(std::string& filename) {
		int width, height, channels;
		// If you find that your textures are flipped vertically, add this line:
		// stbi_set_flip_vertically_on_load(1);
		unsigned char* data =
			stbi_load(filename.c_str(), &width, &height, &channels, 0);
		hres = width;
		vres = height;
		if (data) {
			if (channels == 1) {
				// 8-bit grayscale image.
			} else if (channels == 3) {
				// 8-bit RGB image.
				int width_bytes = width * 3;

				for (int row = 0; row < height; row++) {
					pixels.push_back(std::vector<Colour>());
					for (int col_byte = 0; col_byte < width_bytes; col_byte+=3) {
						unsigned char r = data[row * width_bytes + col_byte];
						unsigned char g = data[row * width_bytes + col_byte + 1];
						unsigned char b = data[row * width_bytes + col_byte + 2];
						Colour clr((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
						pixels[row].push_back(clr);
					}
				}
			} else if (channels == 4) {
				// 8-bit RGBA image.
			}

			// Once you're done with the image
			stbi_image_free(data);
		} else {
			// Image failed to load.
			stbi_image_free(data);
		}
	}

	void read_ppm_file_old(std::string& filename) {
		// doesn't work properly
		char* err = "Currently only support .ppm files that are in P6 format.\n";

		std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
		if (!in) {
			fmt::print("Error opening texture file {}.\n", filename);
			return;
		}
		if (in.get() != 'P' || !in) {
			fmt::print("Error reading first char of texture file (expected \'P\').\n");
			printf(err);
			return;
		}
		if (in.get() != '6' || !in) {
			fmt::print("Error reading first char of texture file (expected \'6\').\n");
			printf(err);
			return;
		}

		int max;
		in >> hres >> vres >> max;
		in.get();
		if (!in) {
			fmt::print("Error reading metadata of texture file.\n");
			return;
		}

		int num_bytes = hres * vres * 3;

		char* bytes = new char[num_bytes];

		in.read(bytes, num_bytes);

		pixels.clear();

		int temp = hres * 3;

		for (int i = 0; i < vres; i++) {
			pixels.push_back(std::vector<Colour>());
			for (int j = 0; j < temp; j+=3) {
				unsigned char r = ((unsigned char)bytes[i*temp + j]);
				unsigned char g = ((unsigned char)bytes[i*temp + j + 1]);
				unsigned char b = ((unsigned char)bytes[i*temp + j + 2]);
				Colour clr;
				clr.r = (float)r / 255.0f;
				clr.g = (float)g / 255.0f;
				clr.b = (float)b / 255.0f;
				pixels[i].push_back(clr);
			}
		}
	}

	Colour get_color(int row, int col) {
		if (row < 0 || col < 0)
			// phantom segfaults that have happened 2 / ~200 renders in release mode that i cant reproduce in debug but i suspect is caused by this
			return { 0.0f,0.0f,0.0f };  
		return pixels[row][col];
	}

	int hres, vres;
	std::vector<std::vector<Colour>> pixels;
};

class Texture {
public:
	virtual Colour get_color(ShadeRec const& sr) const = 0;
};

class ImageTexture : public Texture {
public:
	ImageTexture(std::shared_ptr<Image>& image, std::shared_ptr<Mapping>& mapping) {
		image_ptr = image;
		mapping_ptr = mapping;
		hres = image->hres;
		vres = image->vres;
	}

	virtual Colour get_color(ShadeRec const& sr) const {
		int row = 0, col = 0;

		atlas::math::Point hitPoint(sr.ray(sr.t));
		hitPoint = sr.inv_modelMat * glm::vec4(hitPoint,1.0f);

		if (mapping_ptr)
			mapping_ptr->get_texel_coordinates(hitPoint, hres, vres, row, col);
		else {
			//row = (int)(sr.v * (vres - 1));
			//col = (int)(sr.u * (hres - 1));
			row = 0;
			col = 0;
		}

		return (image_ptr->get_color(row, col));
	}
private:
	int hres, vres;
	std::shared_ptr<Image> image_ptr;
	std::shared_ptr<Mapping> mapping_ptr;
};

class PlaneChecker : public Texture {
	// from ray tracing from the ground up
public:
	PlaneChecker(float size_, Colour& clr1, Colour& clr2) {
		size = size_;
		colour1 = clr1;
		colour2 = clr2;
		outline_width = 0.0f;
	}

	PlaneChecker(float size_, Colour& clr1, Colour& clr2, float outline_width_, Colour& outline_clr) {
		PlaneChecker(size_, clr1, clr2);
		outline_width = outline_width_;
		outline_colour = outline_clr;
	}

	Colour get_color(ShadeRec const& sr) const {
		atlas::math::Point hitPoint(sr.ray(sr.t));
		float x = hitPoint.x;
		float z = hitPoint.z;
		int ix = (int)floor(x / size);
		int iz = (int)floor(z / size);
		float fx = (float)(x / size - ix);
		float fz = (float)(z / size - iz);
		float width = 0.5f * outline_width / size;
		bool in_outline = (fx < width || fx > 1.0f - width) || (fz < width || fz > 1.0f - width);

		if ((ix + iz) % 2 == 0) {
			if (!in_outline) {
				return colour1;
			}
		} else {
			if (!in_outline) {
				return colour2;
			}
		}
		return outline_colour;
	}
private:
	Colour colour1, colour2, outline_colour;
	float size, outline_width;
};

class SV_Lambertian : public BRDF {
public:
	SV_Lambertian(std::shared_ptr<Texture>& tex) {
		texture_ptr = tex;
		kd = 1.0f;
	}

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const {
		(void)reflected;
		(void)incoming;
		return kd * texture_ptr->get_color(sr) * glm::one_over_pi<float>();
	}

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const {
		(void)reflected;
		return kd * texture_ptr->get_color(sr);
	}

	Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wi) {
		(void)wi;
		(void)wo;
		return kd * texture_ptr->get_color(sr);
	}
private:
	float kd;
	std::shared_ptr<Texture> texture_ptr;
};

class SV_Matte : public Material {
public:
	SV_Matte(std::shared_ptr<Texture>& tex) {
		brdf = std::make_shared<SV_Lambertian>(tex);
	}

	Colour shade(ShadeRec& sr, World& world) {
		Colour sumColour{ 0,0,0 };
		atlas::math::Point hitPoint{ sr.ray(sr.t) };
		if (world.ambient) {
			Colour lc = brdf->rho(sr, hitPoint) * world.ambient->L(sr, world);
			sumColour.r += lc.r;
			sumColour.g += lc.g;
			sumColour.b += lc.b;
		}
		for (auto const& l : world.lights) {
			// first check to see if the object is in shadow
			// loop through every object and see if there is an intersection using a ray with
			// hitpoint as origin and the direction towards the light as direction
			atlas::math::Ray<atlas::math::Vector> shadow_ray;
			shadow_ray.o = hitPoint;
			shadow_ray.d = l->getDirection(hitPoint);
			bool shadow_hit = false;
			ShadeRec shadow_sr;
			shadow_sr.t = std::numeric_limits<float>::max();
			for (auto obj_ptr2 : world.scene) {
				if (obj_ptr2->hit(shadow_ray, shadow_sr)) {
					if (!l->isArea() || shadow_sr.t < glm::dot((l->get_sample_point() - shadow_ray.o), shadow_ray.d)) {
						shadow_hit = true;
						break;
					}
				}
			}
			if (shadow_hit) continue;
			atlas::math::Vector toLight{ shadow_ray.d };
			Colour lc = l->L(sr, world);
			Colour brdf_clr = brdf->fn(sr, atlas::math::Vector{}, toLight);
			float ndotwi = glm::max(glm::dot(toLight, sr.normal), 0.0f);
			lc = lc * brdf_clr * ndotwi * l->G(sr) / l->pdf(sr);

			sumColour.r += glm::max(lc.r, 0.0f);
			sumColour.g += glm::max(lc.g, 0.0f);
			sumColour.b += glm::max(lc.b, 0.0f);
		}
		// clamp colours to less than or equal to 1 while maintaing their ratio
		int largest = 0;
		if (sumColour.g > sumColour[largest]) largest = 1;
		if (sumColour.b > sumColour[largest]) largest = 2;

		if (sumColour[largest] > 1.0f) {
			sumColour = sumColour / sumColour[largest];
		}
		return sumColour;
	}

private:
	std::shared_ptr<SV_Lambertian> brdf;
};