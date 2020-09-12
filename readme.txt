Patrick Holland
V00878409
CSC 305 April 2020
Assignment 4


Included files:
main.cpp
assignment.hpp
CMakeLists.txt
paths.hpp.in

cube.obj
land_ocean_ice_cloud_1024.ppm

render.bmp
render_mesh2.bmp
render_bvh_multithread2.bmp

analysis.pdf
readme.txt


Notes:
Program execution is setup to render those 3 images as separate scenes.
The first image uses 512 samples and takes roughly 700 seconds on my PC.
Ambient occlusion suffers with lower samples, but you can lower it to 128 or 256 if you want to speed it up.
In this case, refer to the supplied render.bmp file to see what it looks like with 512 samples.
The other two scenes render quite fast and shouldn't be an issue.

Implemented features (11 points):
Medium Parallelization (2 points):
    Uses the number of threads that the OS returns from std::thread::hardware_concurrency()
	(This is 12 on my desktop and 4 on my laptop)
    Currently have it hardcoded to break the image up into 48 slabs and split them randomly among the threads.
    Refer to analysis.pdf to see the speed up statistics.
BVH (2 points):
    Builds a BVH from the leafs up.
    Combines nodes based on distance between bbox centerpoints.
    Results in a massive speed up.
    Used in all 3 renders.
    Refer to analysis.pdf to see the speed up statistics.
Mirror Reflection (1 point):
    Can be seen clearly in a rectangle and a triangle within render.bmp
Regular Texture Mapping (1 point):
    Mapped a map of the earth onto a sphere in render.bmp
Procedural Texture (1 point):
    Checkerboard plane in render.bmp
Shadows (1 point):
    Shown in every scene.
Ambient Occlusion (1 point):
    Shown only in render.bmp
Area Lights (1 point):
    Shown only in render.bmp
    It works to have multiple area lights in the scene, but I prefered the look with just the one.
Mesh (1 point):
    Shown in render_mesh.bmp
    Doesn't load material files in because the package that atlas uses doesn't support it currently.
	(Emailed Mauricio about it)
    Gave the triangles different colours so you could see how the cube has been created by the triangles.

This is a total of 11 points just in case I lose a mark for something :).




render.bmp (output as render_main.bmp by the program so it doesn't overwrite the included one)
    BVH
    Mirror Reflection
    Regular Texture Mapping
    Procedural Texture
    Shadows
    Ambient Occlusion
    Area Light

render_mesh.bmp
    Multithreading
    BVH
    Shadows
    Mesh
    
render_bvh.bmp
    Multithreading
    BVH
    Shadows