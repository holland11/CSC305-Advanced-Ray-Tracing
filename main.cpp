#include "assignment.hpp"
#include <stdio.h>
#include <thread>
#include <algorithm>
#include <random>
#include <atlas/core/Timer.hpp>

using atlas::core::Timer;

/*
To cmake, navigate to ./build then run 
	>> cmake ..
To build, navigate to ./build then run 
	>> cmake --build .
To compile & run
	>> cmake --build .; .\Debug\a2.exe;
*/


// ******* Function Member Implementation *******

// ***** Shape function members *****
Shape::Shape() : mColour{0, 0, 0}
{}

void Shape::setColour(Colour const& col)
{
    mColour = col;
}

Colour Shape::getColour() const
{
    return mColour;
}

// ***** Camera function members *****
Camera::Camera() :
    mEye{0.0f, 0.0f, 500.0f},
    mLookAt{0.0f},
    mUp{0.0f, 1.0f, 0.0f},
    mU{1.0f, 0.0f, 0.0f},
    mV{0.0f, 1.0f, 0.0f},
    mW{0.0f, 0.0f, 1.0f}
{}

void Camera::setEye(atlas::math::Point const& eye)
{
    mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
    mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
    mUp = up;
}

void Camera::computeUVW()
{
	mW = (mEye - mLookAt);
	mW = glm::normalize(mW);
	
	mU = glm::cross(mW, mUp);
	mU = glm::normalize(mU);
	
	mV = glm::cross(mW,mU);
}

void Orthographic::renderScene(World& world) const {
	world.image = std::vector<Colour>(world.width * world.height);

	for (unsigned int row = 0; row < world.height; row++) {
		if (row % 200 == 0)
			fmt::print("Starting renderPixel() on row {}\n", row);
		for (unsigned int col = 0; col < world.width; col++) {
			Pixel pixel{ col, row };
			renderPixel(world, pixel, true);
		}
	}
}

void Orthographic::renderPixel(World& world, Pixel& pixel, bool count_hit_calls) const {
	atlas::math::Ray<atlas::math::Vector> ray;
	ray.d = mLookAt - mEye;

	atlas::math::Point sample_point, pixel_point;

	float r = 0;
	float g = 0;
	float b = 0;

	pixel_point.y = pixel.y - (world.height * 0.5f) + 0.5f;
	pixel_point.x = pixel.x - (world.width * 0.5f) + 0.5f;

	for (int sample_num = 0; sample_num < world.sampler->getNumSamples(); sample_num++) {
		ShadeRec sr = ShadeRec();
		sr.t = std::numeric_limits<float>::max();
		sr.depth = 0;
		bool hit = false;

		sample_point = world.sampler->sampleUnitSquare();

		auto true_point = pixel_point + sample_point;

		ray.o = true_point;

		for (auto obj_ptr : world.scene) {
			hit |= obj_ptr->hit(ray, sr);
		}

		if (hit && sr.material) {
			// use sr.t & sr.normal to get the shading of the sample
			sr.ray = ray;
			Colour clr = sr.material->shade(sr, world);

			r += clr.r;
			g += clr.g;
			b += clr.b;
		} else {
			r += world.background[0];
			g += world.background[1];
			b += world.background[2];
		}
		if (count_hit_calls) {
			world.bbox_hit_calls += sr.bbox_hit_calls;
			world.shape_hit_calls += sr.shape_hit_calls;
		}
	}
	r = r / world.sampler->getNumSamples();
	g = g / world.sampler->getNumSamples();
	b = b / world.sampler->getNumSamples();
	world.image[pixel.x + pixel.y * world.width] = Colour{ r,g,b };
}

void Orthographic::renderScene_multithreaded(World& world) const {
	Timer<float> timer;
	timer.start();

	world.image = std::vector<Colour>(world.width * world.height);

	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		printf("std::thread::hardware_concurrency() returned 0. (Supposed to return the number of threads that can be created.)\n\
				Setting num_threads to 1, but if the program doesn't run correctly, consider changing multithread to false in the main function.\n");
		num_threads = 1;
	}
	if (num_threads == 12) num_threads = 8; // save some threads so my pc doesnt run like doodoo
	unsigned int slabs_x = 8;
	unsigned int slabs_y = 6;

	unsigned int num_slabs = slabs_x * slabs_y;
	float slabs_per_thread = 1.0f * num_slabs / num_threads;

	unsigned int width_per_slab = (unsigned int)(world.width / slabs_x);
	unsigned int height_per_slab = (unsigned int)(world.height / slabs_y);

	printf("num_threads: %u \n"
		"slabs per thread: %f\n"
		"slab size(w,h): (%u,%u)\n", num_threads, slabs_per_thread, width_per_slab, height_per_slab);

	std::vector<Slab> slabs;

	for (unsigned int i = 0; i < slabs_y; i++) {
		for (unsigned int j = 0; j < slabs_x; j++) {
			Pixel start{ width_per_slab * j, height_per_slab * i };
			Pixel end{ width_per_slab * (j + 1), height_per_slab * (i + 1) };
			if ((j + 1) >= num_threads)
				end.x = (unsigned int)world.width;
			if ((i + 1) >= slabs_per_thread)
				end.y = (unsigned int)world.height;
			slabs.push_back(Slab{ start,end });
		}
	}

	unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::default_random_engine{ seed };
	std::shuffle(std::begin(slabs), std::end(slabs), rng);

	std::vector<std::vector<Slab>> thread_slabs(num_threads);
	for (unsigned int i = 0; i < num_slabs; i++) {
		thread_slabs[i % num_threads].push_back(slabs[i]);
	}

	std::vector<std::thread> threads;
	std::vector<unsigned int> slabs_completed(num_threads, 0);

	float t = timer.elapsed();
	printf("slab preparation took: %f\n", t);
	printf("Threads being dispatched.\n");

	for (unsigned int i = 0; i < num_threads; i++) {
		threads.push_back(std::thread([this, i, &slabs_completed, &num_slabs](World& w, std::vector<Slab>& slabs) {
			for (auto& slab : slabs) {
				for (unsigned int row = slab.start.y; row < slab.end.y; row++) {
					for (unsigned int col = slab.start.x; col < slab.end.x; col++) {
						Pixel pixel{ col, row };
						renderPixel(w, pixel, false);
					}
				}
				/*slabs_completed[i] += 1;
				unsigned int total = 0;
				for (auto el : slabs_completed)
					total += el;
				printf("Roughly %u/%u slabs completed\n", total, num_slabs);*/
			}
			}, std::ref(world), std::ref(thread_slabs[i])));
	}

	for (unsigned int i = 0; i < num_threads; i++) {
		threads[i].join();
	}
}

void Pinhole::renderScene(World& world) const {
	/*
	ray.o = mEye
	creawte points sample_point and pixel_point
	
	for each row
		for each column
			avg_colour {0,0,0}
			
			create shadeRec
			
			for each sample
				hit = false
				sample_point = world.sampler->sampleUnitSquare()
				set pixel_point x and y to center of current pixel
				set ray.d = normalize(pixel_point + sample_point)

				for each object in scene
					temp = object->hit(ray, sr)
					if temp: hit = true
				if hit:
					avg_colour += sr.color
				else:
					avg_colour += background
	*/
	world.image = std::vector<Colour>(world.width * world.height);

	for (unsigned int row = 0; row < world.height; row++) {
		if (row % 200 == 0)
			fmt::print("Starting renderPixel() on row {}\n", row);
		for (unsigned int col = 0; col < world.width; col++) {
			Pixel pixel{ col, row };
			renderPixel(world, pixel, true);
		}
	}
}

void Pinhole::renderPixel(World& world, Pixel& pixel, bool count_hit_calls) const {
	atlas::math::Ray<atlas::math::Vector> ray;
	ray.o = mEye;

	atlas::math::Point sample_point, pixel_point;

	float r = 0;
	float g = 0;
	float b = 0;

	pixel_point.y = pixel.y - (world.height * 0.5f) + 0.5f;
	pixel_point.x = pixel.x - (world.width * 0.5f) + 0.5f;

	for (int sample_num = 0; sample_num < world.sampler->getNumSamples(); sample_num++) {
		ShadeRec sr = ShadeRec();
		sr.t = std::numeric_limits<float>::max();
		sr.depth = 0;
		bool hit = false;

		sample_point = world.sampler->sampleUnitSquare();

		auto true_point = pixel_point + sample_point;

		auto vec_from_eye_to_sample = (mU * true_point.x) + (mV * true_point.y) - (d * mW);

		ray.d = glm::normalize(vec_from_eye_to_sample);

		for (auto obj_ptr : world.scene) {
			hit |= obj_ptr->hit(ray, sr);
		}

		if (hit && sr.material) {
			// use sr.t & sr.normal to get the shading of the sample
			sr.ray = ray;
			Colour clr = sr.material->shade(sr, world);

			r += clr.r;
			g += clr.g;
			b += clr.b;
		} else {
			r += world.background[0];
			g += world.background[1];
			b += world.background[2];
		}
		if (count_hit_calls) {
			world.bbox_hit_calls += sr.bbox_hit_calls;
			world.shape_hit_calls += sr.shape_hit_calls;
		}
	}
	r = r / world.sampler->getNumSamples();
	g = g / world.sampler->getNumSamples();
	b = b / world.sampler->getNumSamples();
	world.image[pixel.x + pixel.y*world.width] = Colour{ r,g,b };
}

void Pinhole::renderScene_multithreaded(World& world) const {
	Timer<float> timer;
	timer.start();

	world.image = std::vector<Colour>(world.width * world.height);

	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) {
		printf("std::thread::hardware_concurrency() returned 0. (Supposed to return the number of threads that can be created.)\n\
				Setting num_threads to 1, but if the program doesn't run correctly, consider changing multithread to false in the main function.\n");
		num_threads = 1;
	} 
	if (num_threads == 12) num_threads = 8; // save some threads so my pc doesnt run like doodoo
	unsigned int slabs_x = 8;
	unsigned int slabs_y = 6;
	
	unsigned int num_slabs = slabs_x * slabs_y;
	float slabs_per_thread = 1.0f*num_slabs / num_threads;

	unsigned int width_per_slab = (unsigned int)(world.width / slabs_x);
	unsigned int height_per_slab = (unsigned int)(world.height / slabs_y);

	printf("num_threads: %u \n"
			"slabs per thread: %f\n"
			"slab size(w,h): (%u,%u)\n", num_threads, slabs_per_thread, width_per_slab, height_per_slab);

	std::vector<Slab> slabs;

	for (unsigned int i = 0; i < slabs_y; i++) {
		for (unsigned int j = 0; j < slabs_x; j++) {
			Pixel start{width_per_slab*j, height_per_slab*i};
			Pixel end{width_per_slab*(j+1), height_per_slab*(i+1)};
			if ((j + 1) >= num_threads)
				end.x = (unsigned int)world.width;
			if ((i + 1) >= slabs_per_thread)
				end.y = (unsigned int)world.height;
			slabs.push_back(Slab{ start,end });
		}
	}

	unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::default_random_engine{seed};
	std::shuffle(std::begin(slabs), std::end(slabs), rng);

	std::vector<std::vector<Slab>> thread_slabs(num_threads);
	for (unsigned int i = 0; i < num_slabs; i++) {
		thread_slabs[i % num_threads].push_back(slabs[i]);
	}

	std::vector<std::thread> threads;
	std::vector<unsigned int> slabs_completed(num_threads, 0);

	float t = timer.elapsed();
	printf("slab preparation took: %f\n", t);
	printf("Threads being dispatched.\n");

	for (unsigned int i = 0; i < num_threads; i++) {
		threads.push_back(std::thread([this, i, &slabs_completed, &num_slabs](World& w, std::vector<Slab>& slabs) {
			for (auto& slab : slabs) {
				for (unsigned int row = slab.start.y; row < slab.end.y; row++) {
					for (unsigned int col = slab.start.x; col < slab.end.x; col++) {
						Pixel pixel{ col, row };
						renderPixel(w, pixel, false);
					}
				}
				/*slabs_completed[i] += 1;
				unsigned int total = 0;
				for (auto el : slabs_completed)
					total += el;
				printf("Roughly %u/%u slabs completed\n", total, num_slabs);*/
			}
		}, std::ref(world), std::ref(thread_slabs[i])));
	}

	for (unsigned int i = 0; i < num_threads; i++) {
		threads[i].join();
	}
}

void FishEye::renderScene(World& world) const {
	// from Ray Tracing From the Ground Up
	Colour L;
	int hres = (int)world.width;
	int vres = (int)world.height;
	float s = 1.0f; // not sure what this is supposed to be
	atlas::math::Ray<atlas::math::Vector> ray;
	atlas::math::Point sp;
	atlas::math::Point pp;
	float r_squared;
	
	ray.o = mEye;
	
	for (int r = 0; r < vres; r++) {
		for (int c = 0; c < hres; c++) {
			L = {0,0,0};
			
			ShadeRec sr;
			sr.t = std::numeric_limits<float>::max();
			
			for (int j = 0; j < world.sampler->getNumSamples(); j++) {
				sp = world.sampler->sampleUnitSquare();
				pp.x = s * (c - 0.5f * hres + sp.x);
				pp.y = s * (r - 0.5f * vres + sp.y);
				ray.d = ray_direction(pp, hres, vres, s, r_squared);
				
				if (r_squared <= 1.0) {
					// trace ray to see what it hits and colour L appropriately
					bool hit = false;
					for (auto obj_ptr : world.scene) {
						hit |= obj_ptr->hit(ray, sr);
					}
					if (hit) {
						// use sr.t & sr.normal to get the shading of the sample
						sr.ray = ray;
						Colour clr = sr.material->shade(sr, world);
						
						L += clr;
					} else {
						L += world.background;
					}
				}
			}
			
			L /= world.sampler->getNumSamples();
			world.image.push_back(Colour{L.r,L.g,L.b});
		}
	}
}

void FishEye::renderScene_multithreaded(World& world) const {
	(void)world;
	printf("Multithreaded rendering not implemented for fish eye camera. Use pinhole instead.\n");
}

atlas::math::Vector FishEye::ray_direction(atlas::math::Point& pp, int hres,
						int vres, float s, float& r_squared) const {
	atlas::math::Point pn{2.0f / (s*hres) * pp.x,
						  2.0f / (s*vres) * pp.y,
						  0};
	r_squared = pn.x * pn.x + pn.y * pn.y;
	
	if (r_squared <= 1.0) {
		float pi_on_180 = 0.017453292519943295f;
		float r = std::sqrt(r_squared);
		float psi = r * psi_max * pi_on_180;
		float sin_psi = sin(psi);
		float cos_psi = cos(psi);
		float sin_alpha = pn.y / r;
		float cos_alpha = pn.x / r;
		atlas::math::Vector dir = sin_psi*cos_alpha*mU +
								  sin_psi*sin_alpha*mV -
								  cos_psi*mW;
		return dir;
	} else {
		return atlas::math::Vector{0,0,0};
	}
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
    mNumSamples{numSamples}, mNumSets{numSets}, mCount{0}, mJump{0}
{
    mSamples.reserve(mNumSets * mNumSamples);
    setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
    return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
    mShuffledIndeces.reserve(mNumSamples * mNumSets);
    std::vector<int> indices;

    std::random_device d;
    std::mt19937 generator(d());

    for (int j = 0; j < mNumSamples; ++j)
    {
        indices.push_back(j);
    }

    for (int p = 0; p < mNumSets; ++p)
    {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int j = 0; j < mNumSamples; ++j)
        {
            mShuffledIndeces.push_back(indices[j]);
        }
    }
}

atlas::math::Point Sampler::sampleUnitSquare()
{
    if (mCount % mNumSamples == 0)
    {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

atlas::math::Point Sampler::sampleHemisphere() {
	if (mCount % mNumSamples == 0) {
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return hemisphere_samples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

void Sampler::map_samples_to_hemisphere(const float exp) {
	int size = (int)mSamples.size();
	hemisphere_samples.reserve(mNumSamples * mNumSets);

	for (int j = 0; j < size; j++) {
		float cos_phi = cos(2.0f * glm::pi<float>() * mSamples[j].x);
		float sin_phi = sin(2.0f * glm::pi<float>() * mSamples[j].x);
		float cos_theta = pow((1.0f - mSamples[j].y), 1.0f / (exp + 1.0f));
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		float pu = sin_theta * cos_phi;
		float pv = sin_theta * sin_phi;
		float pw = cos_theta;
		hemisphere_samples.push_back(atlas::math::Point(pu, pv, pw));
	}
}

// ***** Plane function members *****
Plane::Plane(atlas::math::Point p0, atlas::math::Point p1, atlas::math::Point p2, Colour colour) :
	normal_{-1.0f*glm::normalize(glm::cross((p1-p0),(p2-p0)))},
	p_{p0}
{
	mColour = colour;
}

Plane::Plane(atlas::math::Vector normal, atlas::math::Point p, Colour colour) :
	normal_{-1.0f*glm::normalize(normal)},
	p_{p}
{
	mColour = colour;
}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
                 ShadeRec& sr) {
	sr.shape_hit_calls += 1;
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
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const {
	// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
	float denom = glm::dot(normal_, ray.d);
	const float kEpsilon{0.01f};
	//printf("%f\n", denom);
	if (abs(denom) > kEpsilon) {
		atlas::math::Vector p_o{p_ - ray.o};
		float t = glm::dot(p_o, normal_) / denom;
		//printf("%f %f %f\n", p_o[0], p_o[1], p_o[2]);
		//printf("%f\n", t);
		if (atlas::core::geq(t, kEpsilon)) {
			tMin = t;
			return true;
		}
	}
	return false;
}

// ***** Triangle function members *****
Triangle::Triangle(atlas::math::Vector v0, atlas::math::Vector v1, atlas::math::Vector v2, Colour colour) :
	v0_{v0},
	v1_{v1},
	v2_{v2},
	normal_{-1.0f*glm::normalize(glm::cross((v1-v0),(v2-v0)))}
{
	mColour = colour;
}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
                 ShadeRec& sr) {
	sr.shape_hit_calls += 1;
	float t{};
    bool intersect{intersectRay(ray, t)};
	
    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
		sr.normal = normal_;
        sr.color = mColour;
        sr.t     = t;
		sr.material = mMaterial;
    }
    return intersect;
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const {
	// from ray tracing from the ground up
	double a = v0_[0] - v1_[0], b = v0_[0] - v2_[0], c = ray.d[0], d = v0_[0] - ray.o[0];
	double e = v0_[1] - v1_[1], f = v0_[1] - v2_[1], g = ray.d[1], h = v0_[1] - ray.o[1];
	double i = v0_[2] - v1_[2], j = v0_[2] - v2_[2], k = ray.d[2], l = v0_[2] - ray.o[2];
	
	double m = f*k - g*j, n = h*k - g*l, p = f*l - h*j;
	double q = g*i - e*k, s = e*j - f*i;
	
	double inv_denom = 1.0 / (a*m + b*q + c*s);
	double e1 = d*m - b*n - c*p;
	double beta = e1*inv_denom;
	
	if (beta < 0.0)
		return false;
	
	double r = e*l - h * i;
	double e2 = a*n + d*q + c*r;
	double gamma = e2*inv_denom;
	
	if (gamma < 0.0)
		return false;
	
	if (beta + gamma > 1.0)
		return false;
	
	double e3 = a*p - b*r + d*s;
	float t = (float)(e3*inv_denom);
	const float kEpsilon{ 0.01f };
	
	if (atlas::core::geq(t, kEpsilon)) {
		tMin = t;
		return true;
	}
		
	return false;
}

// ***** Box function members *****
Box::Box(atlas::math::Point p0, atlas::math::Point p1, Colour clr) :
	x0{p0.x}, x1{p1.x}, y0{p0.y}, y1{p1.y}, z0{p0.z}, z1{p1.z}
{
	mColour = clr;
}

atlas::math::Point Box::get_normal(const int face_hit) const {
	// from Ray Tracing from the Ground Up
	switch (face_hit) {
		case 0: return atlas::math::Point{-1,0,0}; // -x face
		case 1: return atlas::math::Point{0,-1,0}; // -y face
		case 2: return atlas::math::Point{0,0,-1}; // -z face
		case 3: return atlas::math::Point{1,0,0}; // +x face
		case 4: return atlas::math::Point{0,1,0}; // +y face
		default: return atlas::math::Point{0,0,1}; // +z face
	}
}

bool Box::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
                 ShadeRec& sr) {
	sr.shape_hit_calls += 1;
	float t{};
    bool intersect{intersectRay(ray, t)};
	
    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.color = mColour;
        sr.t     = t;
		// set the normal based on which face it hit
		sr.material = mMaterial;
    }
    return intersect;
}

bool Box::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const {
	// from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
	// couldn't get the code from Ray Tracing From the Ground Up to work
	float tmin = (x0 - ray.o.x) / ray.d.x;
	float tmax = (x1 - ray.o.x) / ray.d.x;
	
	if (tmin > tmax) {
		float temp = tmin;
		tmin = tmax;
		tmax = temp;
	}
	
	float tymin = (y0 - ray.o.y) / ray.d.y;
	float tymax = (y1 - ray.o.y) / ray.d.y;
	
	if (tymin > tymax) {
		float temp = tymin;
		tymin = tymax;
		tymax = temp;
	}
	
	if ((tmin > tymax) || (tymin > tmax)) return false;
	
	if (tymin > tmin) tmin = tymin;
	
	if (tymax < tmax) tmax = tymax;
	
	float tzmin = (z0 - ray.o.z) / ray.d.z;
	float tzmax = (z1 - ray.o.z) / ray.d.z;
	
	if (tzmin > tzmax) {
		float temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
	}
	
	if ((tmin > tzmax) || (tzmin > tmax)) return false;
	
	if (tzmin > tmin) tmin = tzmin;
	
	if (tzmax < tmax) tmax = tzmax;
	
	tMin = tmin;
	
	return true;
}

/*
bool Box::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                      float& tMin) const {
	const float kEpsilon{0.01f};
	// from Ray Tracing from the Ground Up
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
	
	int face_in, face_out;
	
	if (tx_min > ty_min) {
		t0 = tx_min;
		face_in = (a >= 0.0) ? 0 : 3;
	} else {
		t0 = ty_min;
		face_in = (b >= 0.0) ? 1 : 4;
	}
	
	if (tz_min > 0) {
		t0 = tz_min;
		face_in = (c >= 0.0) ? 2 : 5;
	}
	
	if (tx_max < ty_max) {
		t1 = tx_max;
		face_out = (a >= 0.0) ? 3 : 0;
	} else {
		t1 = ty_max;
		face_out = (b >= 0.0) ? 4 : 1;
	}
	
	if (tz_max < t1) {
		t1 = tz_max;
		face_out = (c >= 0.0) ? 5 : 2;
	}
	
	//printf("t0:%f, t1:%f\n", t0, t1);
	//int temp;
	//std::cin >> temp;
	
	if (t0 < t1 && t1 > kEpsilon) {
		printf("hit box\n");
		if (t0 > kEpsilon) {
			tMin = t0;
			// sr.normal = get_normal(face_in);
		} else {
			tMin = t1;
			// sr.normal = get_normal(face_out);
		}
		return true;
	} else {
		return false;
	}
}
*/

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
    mCentre{center}, mRadius{radius}, mRadiusSqr{radius * radius}
{
	glm::mat4 scale = glm::scale(glm::mat4(1.0f), { radius, radius, radius });
	glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);
	modelMat = translation * scale;
	inv_modelMat = glm::inverse(modelMat);
}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
                 ShadeRec& sr)
{
	sr.shape_hit_calls += 1;
    float t{};
    bool intersect{intersectRay(ray, t)};

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
		sr.normal = glm::normalize(ray(t) - mCentre);
        sr.color = mColour;
        sr.t     = t;
		sr.material = mMaterial;
		sr.inv_modelMat = inv_modelMat;
    }

    return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
                          float& tMin) const
{
    const auto tmp{ray.o - mCentre};
    const auto a{glm::dot(ray.d, ray.d)};
    const auto b{2.0f * glm::dot(ray.d, tmp)};
    const auto c{glm::dot(tmp, tmp) - mRadiusSqr};
    const auto disc{(b * b) - (4.0f * a * c)};

    if (atlas::core::geq(disc, 0.0f))
    {
        const float kEpsilon{0.01f};
        const float e{std::sqrt(disc)};
        const float denom{2.0f * a};

        // Look at the negative root first
        float t = (-b - e) / denom;
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }

        // Now the positive root
        t = (-b + e);
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }
    }

    return false;
}

// ***** Matte function members *****
Colour Matte::shade(ShadeRec& sr, World& world) {
	Colour sumColour{0,0,0};
	atlas::math::Point hitPoint { sr.ray(sr.t) };
	if (world.ambient) {
		Colour lc = world.ambient->L(sr, world);
		sumColour.r += sr.color.r * lc.r;
		sumColour.g += sr.color.g * lc.g;
		sumColour.b += sr.color.b * lc.b;
	}
	for (auto brdf : getBRDFs()) {
		for (auto const &l : world.lights) {
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
			atlas::math::Vector toLight{shadow_ray.d};
			Colour lc = l->L(sr, world);
			Colour brdf_clr = brdf->fn(sr, atlas::math::Vector{}, toLight);
			float ndotwi = glm::max(glm::dot(toLight, sr.normal), 0.0f);
			lc = lc * brdf_clr * ndotwi * l->G(sr) / l->pdf(sr);

			sumColour.r += glm::max(lc.r, 0.0f);
			sumColour.g += glm::max(lc.g, 0.0f);
			sumColour.b += glm::max(lc.b, 0.0f);
		}
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

Colour Phong::shade(ShadeRec& sr, World& world) {
	atlas::math::Vector wo = -1.0f*sr.ray.d;
	Colour sumColour{0,0,0};
	atlas::math::Point hitPoint { sr.ray(sr.t) };
	
	if (world.ambient) {
		sumColour += sr.color * world.ambient->L(sr, world);
	}
	for (auto const &l : world.lights) {
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
		
		atlas::math::Vector wi = l->getDirection(hitPoint);
		float ndotwi = glm::max(glm::dot(sr.normal,wi), 0.0f);
		
		if (ndotwi > 0.0f) {
			for (auto brdf : getBRDFs()) {
				Colour lc = brdf->fn(sr, wo, wi) * l->L(sr, world) * ndotwi * l->G(sr) / l->pdf(sr);
				sumColour.r += glm::max(lc.r, 0.0f);
				sumColour.g += glm::max(lc.g, 0.0f);
				sumColour.b += glm::max(lc.b, 0.0f);
			}
		}
	}
	
	int largest = 0;
	if (sumColour.g > sumColour[largest]) largest = 1;
	if (sumColour.b > sumColour[largest]) largest = 2;
	
	if (sumColour[largest] > 1.0f) {
		sumColour = sumColour / sumColour[largest];
	}
	return sumColour;
};

Colour Reflective::shade(ShadeRec& sr, World& world) {
	Colour L = Phong::shade(sr, world);

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;
	Colour fr = reflective_brdf->sample_f(sr, wo, wi);
	atlas::math::Point hitPoint{ sr.ray(sr.t) };
	atlas::math::Ray<atlas::math::Vector> reflected_ray(hitPoint, wi);
	ShadeRec reflect_sr;
	reflect_sr.depth = sr.depth + 1;
	reflect_sr.t = std::numeric_limits<float>::max();

	bool hit = false;
	Colour reflected_clr(0.0f);

	if (reflect_sr.depth <= world.max_depth) {
		for (auto obj_ptr2 : world.scene) {
			hit |= obj_ptr2->hit(reflected_ray, reflect_sr);
		}

		if (hit) {
			reflect_sr.ray = reflected_ray;
			reflected_clr = reflect_sr.material->shade(reflect_sr, world);
		}
	}

	L += fr * reflected_clr * glm::dot(sr.normal, wi);

	int largest = 0;
	if (L.g > L[largest]) largest = 1;
	if (L.b > L[largest]) largest = 2;

	if (L[largest] > 1.0f) {
		L = L / L[largest];
	}
	return L;
};

Colour GlossyReflector::shade(ShadeRec& sr, World& world) {
	Colour L = Phong::shade(sr, world);
	atlas::math::Vector wo(-sr.ray.d);
	atlas::math::Vector wi;
	float pdf;
	Colour fr(glossy_specular_brdf->sample_f(sr, wo, wi, pdf));
	atlas::math::Ray<atlas::math::Vector> reflected_ray(sr.ray(sr.t), wi);

	ShadeRec reflect_sr;
	reflect_sr.depth = sr.depth + 1;
	reflect_sr.t = std::numeric_limits<float>::max();

	bool hit = false;
	std::shared_ptr<Shape> temp_obj;
	Colour reflected_clr(0.0f);

	if (reflect_sr.depth <= world.max_depth) {
		for (auto obj_ptr2 : world.scene) {
			float temp_t = reflect_sr.t;
			hit |= obj_ptr2->hit(reflected_ray, reflect_sr);
			if (reflect_sr.t != temp_t) {
				temp_obj = obj_ptr2;
			}
		}

		if (hit) {
			reflect_sr.ray = reflected_ray;
			reflected_clr = reflect_sr.material->shade(reflect_sr, world);
		}
	}

	L += fr * reflected_clr * glm::dot(sr.normal, wi) / pdf;

	int largest = 0;
	if (L.g > L[largest]) largest = 1;
	if (L.b > L[largest]) largest = 2;

	if (L[largest] > 1.0f) {
		L = L / L[largest];
	}
	return L;
}

Colour Transparent::shade(ShadeRec& sr, World& world) {
	Colour L = Phong::shade(sr, world);

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;
	Colour fr = reflective_brdf->sample_f(sr, wo, wi);

	atlas::math::Ray<atlas::math::Vector> reflected_ray;
	reflected_ray.o = sr.ray(sr.t);
	reflected_ray.d = wi;

	ShadeRec reflect_sr;
	reflect_sr.depth = sr.depth + 1;
	reflect_sr.t = std::numeric_limits<float>::max();

	if (specular_btdf->tir(sr)) {
		L += trace_ray(reflect_sr, world, reflected_ray);
	} else {
		atlas::math::Vector wt;
		Colour ft = specular_btdf->sample_f(sr, wo, wt);

		atlas::math::Ray<atlas::math::Vector> transmitted_ray;
		transmitted_ray.o = sr.ray(sr.t);
		transmitted_ray.d = wt;
		ShadeRec transmitted_sr;
		transmitted_sr.depth = sr.depth + 1;
		transmitted_sr.t = std::numeric_limits<float>::max();

		L += fr * trace_ray(reflect_sr, world, reflected_ray) * fabs(glm::dot(reflect_sr.normal, wi));
		L += ft * trace_ray(transmitted_sr, world, transmitted_ray) * fabs(glm::dot(transmitted_sr.normal, wt));
	}

	return L;
}

Colour Transparent::trace_ray(ShadeRec& sr, World& world, atlas::math::Ray<atlas::math::Vector>& ray) {
	bool hit = false;
	if (sr.depth <= world.max_depth) {
		for (auto obj_ptr2 : world.scene) {
			hit |= obj_ptr2->hit(ray, sr);
		}

		if (hit) {
			sr.ray = ray;
			return sr.material->shade(sr, world);
		}
	}
	return world.background;
}

Colour Emissive::shade(ShadeRec& sr, World& world) {
	(void)sr;
	(void)world;
	if (glm::dot(-sr.normal, sr.ray.d) > 0.0f) {
		return ls * ce;
	} else {
		return Colour(0.0f);
	}
}

Colour Emissive::get_Le(ShadeRec& sr) const {
	(void)sr;
	return ls * ce;
}

// ***** Light function members *****
Colour Light::L(ShadeRec& sr, World& world) {
	(void)sr;
	(void)world;
	return mColour * mRadiance;
}

void Light::scaleRadiance(float b) {
	mRadiance = mRadiance * b;
}

void Light::setColour(Colour const& c) {
	mColour = c;
}

// ***** Directional function members *****
atlas::math::Vector Directional::getDirection(atlas::math::Point& hitPoint) {
	(void)hitPoint;
	return dir;
}

Colour Directional::L(ShadeRec& sr, World& world) {
	(void)sr;
	(void)world;
	return mColour * mRadiance;
}

// ***** Ambient function members *****
atlas::math::Vector Ambient::getDirection(atlas::math::Point& hitPoint) {
	(void)hitPoint;
	return atlas::math::Vector{0,0,0};
}

Colour Ambient::L(ShadeRec& sr, World& world) {
	(void)sr;
	(void)world;
	return mColour * mRadiance;
}

atlas::math::Vector AmbientOccluder::getDirection(atlas::math::Point& hitPoint) {
	(void)hitPoint;
	auto sp = getHemisphereSample();
	return sp.x * u + sp.y * v + sp.z * w;
}

// ***** AmbientOccluder function members *****
Colour AmbientOccluder::L(ShadeRec& sr, World& world) {
	w = sr.normal;
	v = glm::normalize(glm::cross(w, atlas::math::Vector(0.0072f, 1.0f, 0.0034f)));
	u = glm::cross(v, w);

	atlas::math::Point hitPoint{ sr.ray(sr.t) };

	atlas::math::Ray<atlas::math::Vector> shadow_ray;
	shadow_ray.o = hitPoint;
	shadow_ray.d = getDirection(hitPoint);
	bool shadow_hit = false;
	ShadeRec shadow_sr;
	shadow_sr.t = std::numeric_limits<float>::max();
	for (auto obj_ptr2 : world.scene) {
		if (obj_ptr2->hit(shadow_ray, shadow_sr)) {
			shadow_hit = true;
			break;
		}
	}
	if (shadow_hit) {
		return min_amount * mRadiance * mColour;
	} else {
		return mRadiance * mColour;
	}
}

// ***** PointLight function members *****
atlas::math::Vector PointLight::getDirection(atlas::math::Point& hitPoint) {
	return glm::normalize(p - hitPoint);
}

Colour PointLight::L(ShadeRec& sr, World& world) {
	(void)sr;
	(void)world;
	return mColour * mRadiance;
}

// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{numSamples, numSets}
{
    generateSamples();
}

void Regular::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{(q + 0.5f) / n, (p + 0.5f) / n, 0.0f});
            }
        }
    }
}

// ***** Random function members *****
Random::Random(int numSamples, int numSets) : Sampler{numSamples, numSets}
{
    generateSamples();
}

void Random::generateSamples()
{
    atlas::math::Random<float> engine;
    for (int p = 0; p < mNumSets; ++p)
    {
        for (int q = 0; q < mNumSamples; ++q)
        {
            mSamples.push_back(atlas::math::Point{
                engine.getRandomOne(), engine.getRandomOne(), 0.0f});
        }
    }
}

// ***** Random function members *****
MultiJittered::MultiJittered(int numSamples, int numSets) : Sampler{numSamples, numSets}
{
    generateSamples();
}

void MultiJittered::generateSamples()
{
	// modified from Ray Tracing From the Ground Up
    atlas::math::Random<float> engine;
	
    // num_samples needs to be a perfect square
			
	int n = (int)sqrt((float)mNumSamples);
	float subcell_width = 1.0f / ((float) mNumSamples);
	
	// fill the samples array with dummy points to allow us to use the [ ] notation when we set the 
	// initial patterns
	
	for (int j = 0; j < mNumSamples * mNumSets; j++)
		mSamples.push_back(atlas::math::Point{0,0,0});
		
	// distribute points in the initial patterns
	
	for (int p = 0; p < mNumSets; p++) 
		for (int i = 0; i < n; i++)		
			for (int j = 0; j < n; j++) {
				mSamples[i * n + j + p * mNumSamples].x = (i * n + j) * subcell_width + engine.getRandomOne() * subcell_width;
				mSamples[i * n + j + p * mNumSamples].y = (j * n + i) * subcell_width + engine.getRandomOne() * subcell_width;
			}
	
	// shuffle x coordinates
	for (int p = 0; p < mNumSets; p++) 
		for (int i = 0; i < n; i++)		
			for (int j = 0; j < n; j++) {
				int k = (int)((engine.getRandomOne()*(n-1-j)) + j);
				float t = mSamples[i * n + j + p * mNumSamples].x;
				mSamples[i * n + j + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
				mSamples[i * n + k + p * mNumSamples].x = t;
			}
	// shuffle y coordinates
	
	for (int p = 0; p < mNumSets; p++)
		for (int i = 0; i < n; i++)		
			for (int j = 0; j < n; j++) {
				int k = (int)((engine.getRandomOne()*(n-1-j)) + j);
				float t = mSamples[j * n + i + p * mNumSamples].y;
				mSamples[j * n + i + p * mNumSamples].y = mSamples[k * n + i + p * mNumSamples].y;
				mSamples[k * n + i + p * mNumSamples].y = t;
		}
}

// ******* World building methods *******

void World::build_world_pinhole_matte_point(bool multithread) {
	// pinhole
	// 1 plane
	// 3 spheres
	// 1 triangle
	// 1 pointlight
	// 1 matte material	
	std::shared_ptr<Material> matte = std::make_shared<Matte>();
	
	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{0,-1,0}, atlas::math::Point{0,-200,0}, Colour{0,0.75f,0.75f});
	p1->setMaterial(matte);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{0,0,100}, 200.0f);
	s1->setMaterial(matte);
	s1->setColour(Colour{1,0,0});
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{200,-50,250}, 150.0f);
	s2->setMaterial(matte);
	s2->setColour(Colour{0,0,1});
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{-300,0,-450}, 200.0f);
	s3->setMaterial(matte);
	s3->setColour(Colour{0,1,0});
	scene.push_back(s3);
	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{100,100,150}, atlas::math::Vector{400,100,-250}, atlas::math::Vector{700,100,50}, Colour{0.75,0,0.9});
	t1->setMaterial(matte);
	scene.push_back(t1);
	
	//lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 0.3f));
	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{0,1000,0}, Colour{1,1,1}, 1.0f));
	
	ambient = std::make_shared<Ambient>(Colour{1,1,1},0.1f);
	
	Pinhole cam{200};
	cam.setEye({400,300,-1000});
	cam.setLookAt({0,0,0});
	cam.setUpVector({0,1,0});
	cam.computeUVW();
	
	if (multithread)
		cam.renderScene_multithreaded(*this);
	else
		cam.renderScene(*this);
}

void World::build_world_pinhole_matte_directional(bool multithread) {
	// pinhole
	// 1 plane
	// 3 spheres
	// 1 triangle
	// 1 pointlight
	// 1 matte material	
	std::shared_ptr<Material> matte = std::make_shared<Matte>();
	
	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{0,-1,0}, atlas::math::Point{0,-200,0}, Colour{0,0.75f,0.75f});
	p1->setMaterial(matte);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{0,0,100}, 200.0f);
	s1->setMaterial(matte);
	s1->setColour(Colour{1,0,0});
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{200,-50,250}, 150.0f);
	s2->setMaterial(matte);
	s2->setColour(Colour{0,0,1});
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{-300,0,-450}, 200.0f);
	s3->setMaterial(matte);
	s3->setColour(Colour{0,1,0});
	scene.push_back(s3);
	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{100,100,150}, atlas::math::Vector{400,100,-250}, atlas::math::Vector{700,100,50}, Colour{0.75,0,0.9});
	t1->setMaterial(matte);
	scene.push_back(t1);
	
	lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 1.0f));
	//lights.push_back(std::make_shared<PointLight>(atlas::math::Point{0,1000,0}, Colour{1,1,1}, 1.0f));
	
	ambient = std::make_shared<Ambient>(Colour{1,1,1},0.1f);
	
	Pinhole cam{200};
	cam.setEye({400,300,-1000});
	cam.setLookAt({0,0,0});
	cam.setUpVector({0,1,0});
	cam.computeUVW();
	
	if (multithread)
		cam.renderScene_multithreaded(*this);
	else
		cam.renderScene(*this);
}

void World::build_world_fisheye_matte(bool multithread) {
	// fisheye @ 200fov
	// 1 plane
	// 3 spheres
	// 1 triangle
	// 1 pointlight
	// 1 matte material	
	std::shared_ptr<Material> matte = std::make_shared<Matte>();
	
	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{0,-1,0}, atlas::math::Point{0,-200,0}, Colour{0,0.75f,0.75f});
	p1->setMaterial(matte);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{0,0,100}, 200.0f);
	s1->setMaterial(matte);
	s1->setColour(Colour{1,0,0});
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{200,-50,250}, 150.0f);
	s2->setMaterial(matte);
	s2->setColour(Colour{0,0,1});
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{-300,0,-450}, 200.0f);
	s3->setMaterial(matte);
	s3->setColour(Colour{0,1,0});
	scene.push_back(s3);
	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{100,100,150}, atlas::math::Vector{400,100,-250}, atlas::math::Vector{700,100,50}, Colour{0.75,0,0.9});
	t1->setMaterial(matte);
	scene.push_back(t1);
	
	//lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 0.3f));
	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{0,1000,0}, Colour{1,1,1}, 1.0f));
	
	ambient = std::make_shared<Ambient>(Colour{1,1,1},0.3f);
	
	FishEye cam{200};
	cam.setEye({400,300,-200});
	cam.setLookAt({0,0,0});
	cam.setUpVector({0,1,0});
	cam.computeUVW();
	
	if (multithread)
		cam.renderScene_multithreaded(*this);
	else
		cam.renderScene(*this);
}

void World::build_world_pinhole_specular(bool multithread) {
	// pinhole
	// 1 plane
	// 3 spheres
	// 1 triangle
	// 1 pointlight
	// 1 phong material
	
	std::shared_ptr<Material> mat = std::make_shared<Phong>();
	
	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{0,-1,0}, atlas::math::Point{0,-200,0}, Colour{0,0.75f,0.75f});
	p1->setMaterial(mat);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{0,0,100}, 200.0f);
	s1->setMaterial(mat);
	s1->setColour(Colour{1,0,0});
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{200,-50,250}, 150.0f);
	s2->setMaterial(mat);
	s2->setColour(Colour{0,0,1});
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{-300,0,-450}, 200.0f);
	s3->setMaterial(mat);
	s3->setColour(Colour{0,1,0});
	scene.push_back(s3);
	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{100,100,150}, atlas::math::Vector{400,100,-250}, atlas::math::Vector{700,100,50}, Colour{0.75,0,0.9});
	t1->setMaterial(mat);
	scene.push_back(t1);
	
	//lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 0.3f));
	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{0,1000,-400}, Colour{1,1,1}, 1.0f));
	
	ambient = std::make_shared<Ambient>(Colour{1,1,1},0.2f);
	
	Pinhole cam{200};
	cam.setEye({400,300,-1000});
	cam.setLookAt({0,0,0});
	cam.setUpVector({0,1,0});
	cam.computeUVW();
	
	if (multithread)
		cam.renderScene_multithreaded(*this);
	else
		cam.renderScene(*this);
}

void World::build_world_regular_grid() {


	int num_spheres = 10000;
	float volume = 0.5f / num_spheres;
	float radius = std::pow(0.75f * volume / 3.14159f, 0.333333f);

	std::shared_ptr<Grid> grid_ptr = std::make_shared<Grid>();

	std::shared_ptr<Material> matte = std::make_shared<Matte>();

	printf("radius: %f\n", radius);

	/*for (int j = 0; j < num_spheres; j++) {
		//atlas::math::Point p = atlas::math::Point(1.0f - 2.0f * rand_f(), 1.0f - 2.0f * rand_f(), 1.0f - 2.0f * rand_f());
		atlas::math::Point p = atlas::math::Point(rand_f(), rand_f(), rand_f());
		//printf("%s\n", vec3_to_string(p).c_str());
		std::shared_ptr<Shape> sphere_ptr = std::make_shared<Sphere>(p, radius);
		sphere_ptr->setColour(Colour(rand_f(), rand_f(), rand_f()));
		sphere_ptr->setMaterial(matte);
		grid_ptr->add_object(sphere_ptr);
	}*/
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point(0.0f, 0.0f, 0.0f), 50.5f);
	s1->setColour(Colour(1.0f, 0.0f, 0.0f));
	s1->setMaterial(matte);
	grid_ptr->add_object(s1);

	grid_ptr->setup_cells();

	scene.push_back(grid_ptr);

	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,5,0 }, Colour{ 1,1,1 }, 1.0f));

	ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);


	Pinhole cam{ 2 };
	cam.setEye({ 0,0,-5 });
	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();

	cam.renderScene(*this);
}

void World::build_world_mirror_reflection(bool ao, bool area_instead_of_point) {
	std::shared_ptr<Material> phong = std::make_shared<Phong>();
	std::shared_ptr<Material> matte = std::make_shared<Matte>();
	std::shared_ptr<Material> reflective = std::make_shared<Reflective>();
	std::shared_ptr<Material> emissive = std::make_shared<Emissive>(Colour(1.0f),40.0f);

	std::string texture_filename("land_ocean_ice_cloud_1024.ppm");
	texture_filename = ShaderPath + texture_filename;
	std::shared_ptr<Image> image_ptr = std::make_shared<Image>(texture_filename);
	std::shared_ptr<Mapping> mapping = std::make_shared<SphericalMapping>();
	std::shared_ptr<Texture> image_tex = std::make_shared<ImageTexture>(image_ptr, mapping);
	std::shared_ptr<Material> earth_material = std::make_shared<SV_Matte>(image_tex);

	Colour clr1(0.0f, 0.0f, 0.0f);
	Colour clr2(1.0f, 1.0f, 1.0f);
	std::shared_ptr<Texture> plane_checker = std::make_shared<PlaneChecker>(60.0f, clr1, clr2);
	std::shared_ptr<Material> plane_checker_mat = std::make_shared<SV_Matte>(plane_checker);

	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{ 0,-1,0 }, atlas::math::Point{ 0,-200,0 }, Colour{ 0,0.75f,0.75f });
	p1->setMaterial(plane_checker_mat);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{ -900,0,-400 }, 200.0f);
	s1->setMaterial(phong);
	s1->setColour(Colour{ 1,0,0 });
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{ 700,-50,-350 }, 150.0f);
	s2->setMaterial(phong);
	s2->setColour(Colour{ 0,0,1 });
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{ -600,0,-900 }, 125.0f);
	s3->setMaterial(matte);
	s3->setColour(Colour{ 0,1,0 });
	scene.push_back(s3);
	std::shared_ptr<Shape> s4 = std::make_shared<Sphere>(atlas::math::Point{ 175,200,-300 }, 300.0f);
	s4->setMaterial(earth_material);
	scene.push_back(s4);
	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{ 100,400,150 }, atlas::math::Vector{ 400,400,-250 }, atlas::math::Vector{ 700,400,50 }, Colour{ 0.75,0,0.9 });
	t1->setMaterial(matte);
	scene.push_back(t1);
	std::shared_ptr<Shape> t2 = std::make_shared<Triangle>(atlas::math::Vector{1200,-800,300}, atlas::math::Vector{500,1400,1200}, atlas::math::Vector{1400,400,-300}, Colour{ 1.0,1.0,1.0 });
	t2->setMaterial(reflective);
	scene.push_back(t2);
	std::shared_ptr<Shape> r1 = std::make_shared<Rectangle>(atlas::math::Vector{ -1700,1400,-800 }, atlas::math::Vector{ 2000,0,2000 }, atlas::math::Vector{ 0,-2000,400 }, Colour{ 1.0,1.0,1.0 });
	r1->setMaterial(reflective);
	scene.push_back(r1);

	//lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 0.3f));
	
	if (area_instead_of_point) {
		std::shared_ptr<Shape> rect_light = std::make_shared<Rectangle>(atlas::math::Vector{ -300,1000,-700 }, atlas::math::Vector{ 200,0,0 }, atlas::math::Vector{ 0,100,200 }, Colour{ 1.0,1.0,1.0 });
		rect_light->setMaterial(emissive);
		scene.push_back(rect_light);
		lights.push_back(std::make_shared<AreaLight>(rect_light, emissive, std::make_shared<Random>(64, 83)));

		//std::shared_ptr<Shape> r3 = std::make_shared<Rectangle>(atlas::math::Vector{ 500,600,-600 }, atlas::math::Vector{ 100,-40,0 }, atlas::math::Vector{ 0,0,100 }, Colour{ 1.0,1.0,1.0 });
		//r3->setMaterial(emissive);
		//scene.push_back(r3);
		//lights.push_back(std::make_shared<AreaLight>(r3, emissive, std::make_shared<Random>(64, 83)));
	}
	else
		lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,3000,-400 }, Colour{ 1,1,1 }, 1.0f));

	if (ao) {
		ambient = std::make_shared<AmbientOccluder>(Colour(1.0f), 0.2f, Colour(0.12f));
		ambient->set_sampler(std::make_shared<Random>(64, 83));
	}
	else
		ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);

	build_bvh();

	//ambient = nullptr;

	Pinhole cam{ 400 };
	cam.setEye({ 400,600,-1200 });
	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();

	cam.renderScene(*this);
}

void World::build_world_multithread() {
	std::shared_ptr<Material> phong = std::make_shared<Phong>();
	std::shared_ptr<Material> matte = std::make_shared<Matte>();
	std::shared_ptr<Material> reflective = std::make_shared<Reflective>();
	std::shared_ptr<Material> glossyReflector = std::make_shared<GlossyReflector>(std::make_shared<MultiJittered>(64, 83), 1.0f);
	std::shared_ptr<Material> emissive = std::make_shared<Emissive>(Colour(1.0f), 40.0f);

	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{ 0,-1,0 }, atlas::math::Point{ 0,-200,0 }, Colour{ 0,0.75f,0.75f });
	p1->setMaterial(matte);
	scene.push_back(p1);
	std::shared_ptr<Shape> s1 = std::make_shared<Sphere>(atlas::math::Point{ 0,0,100 }, 200.0f);
	s1->setMaterial(phong);
	s1->setColour(Colour{ 1,0,0 });
	scene.push_back(s1);
	std::shared_ptr<Shape> s2 = std::make_shared<Sphere>(atlas::math::Point{ 200,-50,250 }, 150.0f);
	s2->setMaterial(phong);
	s2->setColour(Colour{ 0,0,1 });
	scene.push_back(s2);
	std::shared_ptr<Shape> s3 = std::make_shared<Sphere>(atlas::math::Point{ -300,0,-450 }, 200.0f);
	s3->setMaterial(phong);
	s3->setColour(Colour{ 0,1,0 });
	scene.push_back(s3);

	std::vector<atlas::math::Point> clump_centers = { 
		{900,1200,1000}, {-900,500,2000}, {0,1600,500}, {400,1500,4000},
		{-2000,1500,1200}, {2000,1600,1000}, {-3000,1500,1000}, {200,500,400},
		{-4000,1500,1000}, {-5000,1200,600}, {-6000,1600,1200}
	};
	std::vector<float> clump_scale = { 
		100.0f, 100.0f, 150.0f, 200.0f,
		100.0f, 100.0f, 200.0f, 80.0f,
		100.0f, 100.0f, 100.0f
	};
	std::vector<Colour> clump_colours = { 
		{0.9f,0.3f,0.1f}, {0.2f,0.7f,0.2f}, {0.2f,0.2f,0.8f}, {0.5f,0.1f,0.7f},
		{0.8f,0.0f,0.0f}, {0.0f,0.9f,0.0f}, {0.0f,0.0f,0.9f}, {1.0f,1.0f,1.0f},
		{0.8f,0.8f,0.1f}, {0.0f,1.0f,1.0f}, {0.3f,1.0f,0.0f}
	};
	std::vector<atlas::math::Point> clump_offsets = {
		atlas::math::Point(0.0f,0.0f,0.0f),
		atlas::math::Point(3.0f,0.0f,0.0f),  atlas::math::Point(-3.0f, 0.0f, 0.0f),
		atlas::math::Point(-2.0f, 2.0f, 0.0f), atlas::math::Point(2.0f, 2.0f, 0.0f),
		atlas::math::Point(2.0f, -2.0f, 0.0f), atlas::math::Point(-2.0f, -2.0f, 0.0f),
		atlas::math::Point(0.0f, 3.0f, 0.0f), atlas::math::Point(0.0f, -3.0f, 0.0f)
	};
	int num_clumps = (int)clump_centers.size();
	int num_shapes_in_clump = (int)clump_offsets.size();

	for (int i = 0; i < num_clumps; i++) {
		for (int j = 0; j < num_shapes_in_clump; j++) {
			std::shared_ptr<Shape> shape = std::make_shared<Sphere>(clump_centers[i] + (clump_scale[i] * clump_offsets[j]), clump_scale[i]);
			shape->setMaterial(phong);
			shape->setColour(clump_colours[i]);
			scene.push_back(shape);
		}
	}

	std::shared_ptr<Shape> t1 = std::make_shared<Triangle>(atlas::math::Vector{ 100,100,150 }, atlas::math::Vector{ 400,100,-250 }, atlas::math::Vector{ 700,100,50 }, Colour{ 0.75,0,0.9 });
	t1->setMaterial(matte);
	scene.push_back(t1);
	std::shared_ptr<Shape> r1 = std::make_shared<Rectangle>(atlas::math::Point{ -600,700,-100 }, atlas::math::Vector{ 0,-300,-500 }, atlas::math::Vector{ 400,0,0 }, Colour{ 0.6f,0.6f,0.1f });
	r1->setMaterial(matte);
	scene.push_back(r1);

	build_bvh();

	//lights.push_back(std::make_shared<Directional>(atlas::math::Vector{0.25f,1.0f,0.25f}, Colour{1,1,1}, 0.3f));
	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,3000,-400 }, Colour{ 1,1,1 }, 1.0f));

	ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);

	//ambient = nullptr;

	Pinhole cam{ 200 };
	cam.setEye({ 400,300,-1000 });
	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();

	cam.renderScene_multithreaded(*this);
}

void World::build_world_mesh(std::string mesh_filename, std::string mesh_fileRoot) {
	std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
	mesh->build_mesh(mesh_filename, mesh_fileRoot);

	/*float scale = 55.0f;
	Pinhole cam{ 5 };
	cam.setEye({ 0,0,30 });*/
	float scale = 700.0f;
	Pinhole cam{ 150 };
	cam.setEye({ 700,700,800 });

	mesh->modelMat = glm::scale(mesh->modelMat, glm::vec3(scale,scale,scale));
	mesh->inv_modelMat = glm::inverse(mesh->modelMat);

	std::shared_ptr<Material> matte = std::make_shared<Matte>();

	std::vector<Colour> clrs = {
		{ 1.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.0f }, { 0.5f, 0.5f, 0.0f },
		{ 0.0f, 1.0f, 0.0f }, { 0.0f, 0.75f, 0.25f }, { 0.0f, 0.5f, 0.5f },
		{ 0.0f, 0.0f, 1.0f }, { 0.25f, 0.0f, 0.75f }, { 0.5f, 0.0f, 0.5f }
	};

	int num_clrs = (int)clrs.size();
	int num_faces = (int)mesh->faces.size();
	for (int i = 0; i < num_faces; i++) {
		std::shared_ptr<Shape> tri = std::make_shared<FlatTriangle>(mesh, i, clrs[i % num_clrs], matte);
		scene.push_back(tri);
	}

	std::shared_ptr<Shape> p1 = std::make_shared<Plane>(atlas::math::Vector{ 0,-1,0 }, atlas::math::Point{ 0,-4500,0 }, Colour{ 0,0.75f,0.75f });
	p1->setMaterial(matte);
	scene.push_back(p1);

	lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,3000,1000 }, Colour{ 1,1,1 }, 1.0f));
	ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);

	build_bvh();

	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();
	//cam.renderScene(*this);
	cam.renderScene_multithreaded(*this);
}

void World::build_test() {
	std::string texture_filename("./../land_ocean_ice_cloud_1024.ppm");
	std::shared_ptr<Image> image_ptr = std::make_shared<Image>(texture_filename);
	std::shared_ptr<Mapping> mapping = std::make_shared<SphericalMapping>();
	std::shared_ptr<Texture> image_tex = std::make_shared<ImageTexture>(image_ptr, mapping);
	std::shared_ptr<Material> sv_matte = std::make_shared<SV_Matte>(image_tex);

	std::shared_ptr<Shape> s4 = std::make_shared<Sphere>(atlas::math::Point{ 300,0,0 }, 400.0f);
	s4->setMaterial(sv_matte);
	scene.push_back(s4);
	build_bvh();

	bool area_instead_of_point = true;
	bool ao = true;

	if (area_instead_of_point) {
		std::shared_ptr<Material> emissive = std::make_shared<Emissive>(Colour(1.0f), 40.0f);
		std::shared_ptr<Shape> r2 = std::make_shared<Rectangle>(atlas::math::Vector{ 0,1200,0 }, atlas::math::Vector{ 500,0,0 }, atlas::math::Vector{ 0,150,500 }, Colour{ 1.0,1.0,1.0 });
		r2->setMaterial(emissive);
		scene.push_back(r2);
		lights.push_back(std::make_shared<AreaLight>(r2, emissive, std::make_shared<Random>(64, 83)));
	} else
		lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,3000,-400 }, Colour{ 1,1,1 }, 1.0f));

	if (ao) {
		ambient = std::make_shared<AmbientOccluder>(Colour(1.0f), 0.2f, Colour(0.12f));
		ambient->set_sampler(std::make_shared<Random>(64, 83));
	} else
		ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);

	Pinhole cam{ 200 };
	cam.setEye({ 400,300,-1000 });
	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();

	cam.renderScene(*this);
}

void World::build_bvh() {
	Timer<float> timer;
	timer.start();
	
	std::vector<std::shared_ptr<Shape>> shapes_not_in_bvh;
	std::vector<std::shared_ptr<BVHnode>> nodes;

	// start by creating a bvh leaf node for each shape in the scene
	for (auto& shape : scene) {
		if (shape->hasBBox()) {
			// planes dont have a bbox and wont go into a bvh node
			nodes.push_back(std::make_shared<BVHnode>(shape));
		} else {
			shapes_not_in_bvh.push_back(shape);
		}
	}

	int num_nodes = (int)nodes.size();
	// now find the two closest nodes and put them into a new bvh node
	// repeat until only one node left
	while (num_nodes > 1) {
		float closest = -1.0f;
		int n1_i = 0;
		int n2_i = 0;
		for (int i = 0; i < num_nodes; i++) {
			for (int j = i + 1; j < num_nodes; j++) {
				float dist = nodes[i]->distance(nodes[j]);
				if (closest == -1.0f || dist < closest) {
					closest = dist;
					n1_i = i;
					n2_i = j;
				}
			}
		}

		std::shared_ptr<BVHnode> new_node = std::make_shared<BVHnode>(nodes[n1_i], nodes[n2_i]);
		nodes.erase(nodes.begin() + n2_i);
		nodes.erase(nodes.begin() + n1_i);
		nodes.push_back(new_node);

		num_nodes = (int)nodes.size();
	}

	scene.clear();
	scene.push_back(std::static_pointer_cast<Shape>(nodes[0]));
	for (auto& shape : shapes_not_in_bvh)
		scene.push_back(shape);

	float t = timer.elapsed();
	printf("bvh tree building took: %f\n", t);
}

void World::build_world_bvh(int num_clumps) {
	/*
	100 clumps, 16 samples, random w/ same seed on laptop

	with bvh:
	elapsed: 24.000694
	BBox hit calls: 635153716
	Shape hit calls: 27642780
	662796496

	elapsed: 22.557024
	BBox hit calls: 635151114
	Shape hit calls: 27642775
	662793889

	elapsed: 21.771664
	BBox hit calls: 635152420
	Shape hit calls: 27642804
	662795224


	without bvh:

	elapsed: 504.000244
	BBox hit calls: 0
	Shape hit calls: 1515098112


	200 clumps 16 samples random w/ same seed on laptop

	with bvh singlethreading

	elapsed: 59.536659
	BBox hit calls: 1223118344
	Shape hit calls: 65518823

	with bvh mutlithreading (4 threads)
	elapsed: 56.276257

	(with 1 slab per thread)
	elapsed: 60.093975

	
	200 clumps 16 samples random w/ same seed on desktop

	with bvh singlethreading

	elapsed: 38.210220
	BBox hit calls: 1223116862
	Shape hit calls: 65518484


	with bvh multithreading (1 thread) 24 slabs
	elapsed: 78.647552

	with bvh multi (1 thread) 1 slab
	elapsed: 38.363369

	with bvh multi (8 threads) 24 slabs (3 per thread)
	elapsed: 29.744591

	with bvh multi (8 threads) 8 slabs (1 per thread)
	elapsed: 27.479179




	mirror reflection w/ 512 samples
	
	with bvh:
	total elapsed: 291.705292
	BBox hit calls: 2562728382
	Shape hit calls: 875714041
	(total): 3.4 billion (3438442423)
	w/o bvh:
	total elapsed: 375.767120
	BBox hit calls: 0
	Shape hit calls: 4096000000
	(total): 4.09 billion
	*/
	std::shared_ptr<Material> phong = std::make_shared<Phong>();

	float minx = -3500;
	float maxx = 3500;
	float miny = -3500;
	float maxy = 3500;
	float minz = 1000;
	float maxz = 5000;

	float min_radius = 50.0f;
	float max_radius = 200.0f;

	//srand((unsigned int)time(0)); // for calls to rand_f()

	std::vector<atlas::math::Point> clump_offsets = {
		atlas::math::Point(0.0f,0.0f,0.0f),
		atlas::math::Point(3.0f,0.0f,0.0f),  atlas::math::Point(-3.0f, 0.0f, 0.0f),
		atlas::math::Point(-2.0f, 2.0f, 0.0f), atlas::math::Point(2.0f, 2.0f, 0.0f),
		atlas::math::Point(2.0f, -2.0f, 0.0f), atlas::math::Point(-2.0f, -2.0f, 0.0f),
		atlas::math::Point(0.0f, 3.0f, 0.0f), atlas::math::Point(0.0f, -3.0f, 0.0f)
	};

	int num_offsets = (int)clump_offsets.size();

	for (int i = 0; i < num_clumps; i++) {
		float x = rand_f(minx, maxx);
		float y = rand_f(miny, maxy);
		float z = rand_f(minz, maxz);
		if (z > 4000) {
			x = x * 4;
			y = y * 4;
		} else if (z > 3000) {
			x = x * 3;
			y = y * 3;
		} else if (z > 2000) {
			x = x * 2;
			y = y * 2;
		}
		atlas::math::Point c(x, y, z);
		float r = rand_f(min_radius, max_radius);
		Colour clr(rand_f(), rand_f(), rand_f());
		for (int j = 0; j < num_offsets; j++) {
			std::shared_ptr<Shape> shape = std::make_shared<Sphere>(c + (r * clump_offsets[j]), r);
			shape->setMaterial(phong);
			shape->setColour(clr);
			scene.push_back(shape);
		}
	}

	build_bvh();

	//lights.push_back(std::make_shared<PointLight>(atlas::math::Point{ 0,3000,-400 }, Colour{ 1,1,1 }, 1.0f));
	lights.push_back(std::make_shared<Directional>(atlas::math::Vector{ 0.15f,1.0f,-0.25f }, Colour{ 1,1,1 }, 0.8f));
	lights.push_back(std::make_shared<Directional>(atlas::math::Vector{ -0.15f,-1.0f,-0.25f }, Colour{ 1,1,1 }, 0.8f));

	ambient = std::make_shared<Ambient>(Colour(1.0f), 0.2f);

	//ambient = nullptr;

	Pinhole cam{ 200 };
	cam.setEye({ 0,0,-1000 });
	cam.setLookAt({ 0,0,0 });
	cam.setUpVector({ 0,1,0 });
	cam.computeUVW();

	//cam.renderScene(*this);
	cam.renderScene_multithreaded(*this);
}

// ******* Driver Code *******

int main()
{
	/*std::string texture_filename("./../land_ocean_ice_cloud_1024.ppm");
	std::shared_ptr<Image> image_ptr = std::make_shared<Image>(texture_filename);

	std::vector<Colour> image;
	std::vector<std::vector<Colour>> pixels = image_ptr->pixels;
	for (std::vector<Colour>& row : pixels) {
		for (Colour& pix : row) {
			image.push_back(pix);
		}
	}
	int width = image_ptr->hres;
	int height = image_ptr->vres;

	saveToFile("./test.bmp", width, height, image);*/

	//std::shared_ptr<Sampler> rand = std::make_shared<Random>(8, 100);
	std::shared_ptr<Sampler> mj = std::make_shared<MultiJittered>(512, 100);
	float t = 0.0f;

	printf("Starting first image with %d samples.\n", mj->getNumSamples());
	printf("This render includes shadows, ambient occlusion, area lights, mirror reflection, regular texture mapping (earth on sphere), procedural texture (checkerboard plane) and uses a BVH.\n");
	printf("This render takes roughly 700 seconds on my desktop pc.\n\n");

	bool ambientOcclusion = true;
	bool area_instead_of_point = true;
	(void)ambientOcclusion;
	(void)area_instead_of_point;
	
	World main_world = World();
	main_world.sampler = mj;
	main_world.width = 1500;
	main_world.height = 1500;
	main_world.background = Colour{0,0,0};
	main_world.max_depth = 4;

	Timer<float> timer;
	timer.start();
	main_world.build_world_mirror_reflection(ambientOcclusion, area_instead_of_point);
	//main_world.build_world_multithread(); // 55.9 50x50 | 18.2 200x200
	//main_world.build_world_regular_grid();
	//main_world.build_world_mesh("cube.obj", ShaderPath);
	//main_world.build_world_bvh(200);
	//main_world.build_test();
	t = timer.elapsed();
	printf("total elapsed: %f\n", t);

	fmt::print("BBox hit calls: {}\nShape hit calls: {}\n", main_world.bbox_hit_calls, main_world.shape_hit_calls);

	saveToFile("./render_main.bmp", main_world.width, main_world.height, main_world.image);
	
	printf("Done first image.\n\n");



	mj = std::make_shared<MultiJittered>(128, 100);

	printf("Starting second image with %d samples.\n", mj->getNumSamples());
	printf("This scene includes a cube mesh with point and ambient lighting. A BVH and multithreading are used to speed up the execution.\n");
	printf("Calls to bbox hit and shape hit are not counted while multithreading is active.\n");
	printf("This scene takes roughly 15 seconds to render on my desktop pc.\n\n");

	World mesh_world = World();
	mesh_world.sampler = mj;
	mesh_world.width = 1000;
	mesh_world.height = 1000;
	mesh_world.background = Colour{ 0,0,0 };
	mesh_world.max_depth = 4;

	Timer<float> timer2;
	timer2.start();
	mesh_world.build_world_mesh("cube.obj", ShaderPath);
	t = timer2.elapsed();
	printf("total elapsed: %f\n", t);

	saveToFile("./render_mesh.bmp", mesh_world.width, mesh_world.height, mesh_world.image);

	printf("Done second image.\n\n");



	mj = std::make_shared<MultiJittered>(64, 100);

	printf("Starting first image with %d samples.\n", mj->getNumSamples());
	printf("This scene includes 1800 spheres scattered around the scene and is meant to showcase the benefits of BVH and multithreading.\n");
	printf("Calls to bbox hit and shape hit are not counted while multithreading is active, but a comparison with and without the BVH can be seen in the analysis pdf.\n");
	printf("This scene takes roughly 31 seconds to render on my desktop pc.\n\n");

	World bvh_world = World();
	bvh_world.sampler = mj;
	bvh_world.width = 1000;
	bvh_world.height = 1000;
	bvh_world.background = Colour{ 0,0,0 };
	bvh_world.max_depth = 4;

	Timer<float> timer3;
	timer3.start();
	bvh_world.build_world_bvh(200);
	t = timer3.elapsed();
	printf("total elapsed: %f\n", t);

	saveToFile("./render_bvh_multithreaded.bmp", bvh_world.width, bvh_world.height, bvh_world.image);

	printf("Done third image.\n\n");
	
    return 0;
}

void saveToFile(std::string const& filename,
                std::size_t width,
                std::size_t height,
                std::vector<Colour> const& image)
{
    std::vector<unsigned char> data(image.size() * 3);

    for (std::size_t i{0}, k{0}; i < image.size(); ++i, k += 3)
    {
        Colour pixel = image[i];
        data[k + 0]  = static_cast<unsigned char>(pixel.r * 255);
        data[k + 1]  = static_cast<unsigned char>(pixel.g * 255);
        data[k + 2]  = static_cast<unsigned char>(pixel.b * 255);
    }

    stbi_write_bmp(filename.c_str(),
                   static_cast<int>(width),
                   static_cast<int>(height),
                   3,
                   data.data());
}
