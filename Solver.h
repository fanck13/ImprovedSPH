#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Paras.cuh"
#include <iostream>
#include <fstream>
using namespace std;

class Solver
{
public:
	Solver(unsigned int _count);
	~Solver();

private:
	unsigned int count;
	float3* hpos;
	float3* hvel;
	float3* dpos;
	float3* dvel;

	float* hdens;

	float3* dspos;
	float3* dsvel;
	float* ddens;
	float3* dforce;
	float* dpress;
	unsigned int *dindex;
	unsigned int *dhash;
	unsigned int *dcellStart;
	unsigned int *dcellEnd;

	size_t size1;
	size_t size3;
	size_t gridNum;

	Paras pa;

	float dt;

	int *dneighbourNum; //neighbour the number of particles nearby
	int *hneighbourNum;

private:
	void InitParticles();
public:
	void Update();

	float3* GetPos();

	float3* GetVel();

	float* GetDens();
	float CalcTimeStep();

	void SetBoundary(float x);

	int* GetNeighbourNum();
};

