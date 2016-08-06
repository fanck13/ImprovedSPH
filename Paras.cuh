#include "device_launch_parameters.h"
#include "cuda_runtime.h"

struct Paras
{
	float dt;

	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float zmin;
	float zmax;

	float mass;
	float h;
	float restDens;
	float k;
	float mu;
	float3 gravity;

	int3 gridSize;
	float3 worldOrigin;
	float3 cellSize;

	float globalDamping;
	float particleRadius;


	float spring;
	float damping;
	float shear;
	float attraction;
	float boundaryDamping;

	int threshold;

	float soundSpeed;

	float beta;
	float gama;

	float zdamp;
};

