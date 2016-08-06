#include "Solver.h"

extern "C"
{
	void HandleError(cudaError_t status, char* message);

	void SetParas(Paras *p);

	void CalHash(unsigned int* index,  unsigned int* hash, 
				 float3* pos,  unsigned int count);

	void SortParticles(unsigned int *hash, unsigned int *index, 
					   unsigned int count);

	void ReorderDataAndFindCellStart(unsigned int* cellstart,
									 unsigned int* cellend,
									 float3* spos,float3* svel,
									 unsigned int* hash,
									 unsigned int* index,
									 float3* pos, float3* vel,
									 unsigned int count,
									 unsigned int gridNum);

	void CalcNeighNumber(unsigned int* cellstart,unsigned int* cellend, 
		                 float3* spos, unsigned int count, int* pnum);

	void CalcDensity(float* dens, unsigned int* cellstart, 
					 unsigned int* cellend, float3 *spos, 
					 unsigned int count, int* pnum);

	void CalcPressure(float* press, float* dens, 
					  unsigned int count);

	void CalcForce(float3* force, float3* spos, float3* svel, float3* vel, 
				   float* press, float* dens, unsigned int* index, 
				   unsigned int* cellstart, unsigned int* cellend, 
				   unsigned int count, int* pnum);

	void UpdateVelocityAndPosition(float3* pos, float3* vel, 
								   float3* force, unsigned int count, 
								   float dt);

	void HandleBoundary(float3* pos,  float3* vel, unsigned int count);

	void HandleBoundary2(float3* pos, float3* vel, unsigned int count);

}

inline float operator*(float3 a, float3 b)
{
	return (a.x*b.x + a.y*b.y + a.z*b.z);
}


#define CHECK(ptr, message)  {if(ptr==nullptr)\
								{cerr<<message<<endl;exit(1);}}


Solver::Solver(unsigned int _count) :count(_count)
{
	size1 = count*sizeof(float);
	size3 = count*sizeof(float3);
	gridNum = 50 * 30 * 40;


	////////set parameters//////////////
	pa.xmin = 0.0f;
	pa.xmax = 50.0f;
	pa.ymin = 0.0f;
	pa.ymax = 30.0f;
	pa.zmin = 0.0f;
	pa.zmax = 40.0f;

	pa.mass = 1.0f;
	pa.dt = 0.00001f;
	pa.h = 1.1f;
	pa.k = 1500.0f;

	pa.restDens = 2.5f;
	pa.mu = 0.1f;
	pa.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);
	pa.cellSize = make_float3(1.0f, 1.0f, 1.0f);
	pa.gridSize =make_int3(50,30, 40);
	pa.gravity = make_float3(0.0f, 980.f, 0.0f);

	pa.zdamp = 1.0f;
	//
	pa.spring = 0.5f;
	pa.damping = 0.08f;   //碰撞阻尼

	pa.shear = 2.0f;

	pa.attraction = 3.5f;
	pa.boundaryDamping = 0.5f;

	pa.particleRadius = 1.0f*pa.h;

	pa.threshold = 7;           //粒子的周围粒子个数的多少阈值

	//
	pa.soundSpeed = 2.1f;

	pa.gama = 2;
	pa.beta = 200.0f*9.8f*20.0f / (pa.restDens*pa.gama);

	////////allocate memory//////
	
	hpos=(float3*)malloc(size3);
	CHECK(hpos, "Failed to allocate memory of hpos!");
	hvel = (float3*)malloc(size3);
	CHECK(hvel, "Failed to allocate memory of hvel!");
	hdens = (float*)malloc(size1);
	CHECK(hdens, "Failed to allocate memory of hdens!");
	hneighbourNum = (int*)malloc(sizeof(int)*count);
	CHECK(hneighbourNum, "Failed to allocate memory of hneughbour");


	HandleError(cudaMalloc((void**) &dpos, size3), "Failed to allocate memory of dpos!");
	HandleError(cudaMalloc((void**) &dvel, size3), "Failed to allocate memory of dvel!");
	HandleError(cudaMalloc((void**)&dspos, size3), "Failed to allocate memory of dspos!");
	HandleError(cudaMalloc((void**)&dsvel, size3), "Failed to allocate memory of dsvel!");
	HandleError(cudaMalloc((void**) &ddens, size1), "Failed to allocate memory of ddens!");
	HandleError(cudaMalloc((void**) &dforce, size3), "Failed to allocate memory of dforce!");
	HandleError(cudaMalloc((void**) &dpress, size1), "Failed to allocate memory of dpress!");
	HandleError(cudaMalloc((void**)&dindex, count*sizeof(unsigned int)), "Failed to allocate memory of dindex");
	HandleError(cudaMalloc((void**)&dhash, count*sizeof(unsigned int)), "Failed to allocate memory of dhash");
	HandleError(cudaMalloc((void**)&dcellStart, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellstart");
	HandleError(cudaMalloc((void**)&dcellEnd, gridNum*sizeof(unsigned int)), "Failed to allocate memory of dcellend");

	HandleError(cudaMalloc((void**)&dneighbourNum, count*sizeof(int)), "Failed to allocate memory of dcellend");
	HandleError(cudaMemset(dneighbourNum, 0, sizeof(int)*count), "Failed to memset neighbour");

	HandleError(cudaMemcpy(hneighbourNum, dneighbourNum, sizeof(int)*count, cudaMemcpyDeviceToHost), "Failed to copy device to host in update!");

	/*ofstream oo("oo.txt");

	for (int i = 0; i < count; i++)
	{
		oo << hneighbourNum[i] << endl;
	}*/

	InitParticles();

	HandleError(cudaMemcpy(dpos, hpos, size3, cudaMemcpyHostToDevice), "Failed to copy memory of hpos!");
	HandleError(cudaMemcpy(ddens, hdens, size1, cudaMemcpyHostToDevice), "Failed to copy memory of hpos!");
	HandleError(cudaMemset(dvel, 0, size3), "Failed to memset dvel!");
	HandleError(cudaMemset(dsvel, 0, size3), "Failed to memset dsvel!"); 
	HandleError(cudaMemset(dspos, 0, size3), "Failed to memset dspos!"); 
	HandleError(cudaMemset(ddens, 0, size1), "Failed to memset ddens!");
	HandleError(cudaMemset(dforce, 0, size3), "Failed to memset dforce!");
	HandleError(cudaMemset(dpress, 0, size1), "Failed to memset dpress!");
}


Solver::~Solver()
{
	free(hpos);
	free(hvel);
	free(hdens);

	HandleError(cudaFree(dpos), "Failed to free dpos!");
	HandleError(cudaFree(dvel), "Failed to free dvel!");
	HandleError(cudaFree(ddens), "Failed to free ddens!");
	HandleError(cudaFree(dforce), "Failed to free dforce!");
	HandleError(cudaFree(dpress), "Failed to free dpress!");
	HandleError(cudaFree(dhash), "Failed to free dhash!");
	HandleError(cudaFree(dindex), "Failed to free dindex!");
	HandleError(cudaFree(dcellStart), "Failed to free dcellStart!");
	HandleError(cudaFree(dcellEnd), "Failed to free dcellEnd!");

	HandleError(cudaFree(dspos), "Failed to free dspos!");
	HandleError(cudaFree(dsvel), "Failed to free dsvel!");
}


void Solver::InitParticles()
{
	int id = 0;
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				id = k * 32 * 32 + j * 32 + i;
				hpos[id].x = i*0.9f+0.5f;
				hpos[id].y = k*0.9f+0.5f;
				hpos[id].z = j*0.9f+0.5f;

				hdens[id] = pa.restDens;
			}
		}
	}
}

float Solver::CalcTimeStep()
{
	float maxval = 0.0f;
	for (int i = 0; i < count; i++)
	{
		float temp = sqrtf(hvel[i] * hvel[i]);
		if (maxval < temp)
		{
			maxval = temp;
		}
	}
	/*pa.soundSpeed = maxval/0.5f;
	cout << pa.soundSpeed << endl;
	system("pause");*/
	return /*0.4f**/2.0f*(pa.h / maxval);
}

void Solver::Update()
{
	SetParas(&pa);
	//cout << "SetParas" << endl;
	CalHash(dindex, dhash, dpos, count);
	//cout << "CalHash" << endl;
	SortParticles(dhash, dindex, count);
	//cout << "SortParticles" << endl;
	ReorderDataAndFindCellStart(dcellStart, dcellEnd, dspos, dsvel, 
		                        dhash, dindex, dpos, dvel, count, gridNum);
	//cout << "ReorderDataAndFindCellStart" << endl;
	CalcNeighNumber(dcellStart, dcellEnd, dspos, count, dneighbourNum);
	CalcDensity(ddens, dcellStart, dcellEnd, dspos, count, dneighbourNum);

	HandleError(cudaMemcpy(hneighbourNum, 
		                   dneighbourNum, 
						   sizeof(int)*count, 
		                   cudaMemcpyDeviceToHost), 
				"Failed to copy device to host in update!");

	//cout << "CalcDensity" << endl;
	CalcPressure(dpress, ddens, count);
	//cout << "CalcPressure" << endl;
	CalcForce(dforce, dspos, dsvel, dvel, dpress, ddens, dindex, 
		       dcellStart, dcellEnd, count, dneighbourNum);
	//cout << "CalcForce" << endl;


	HandleError(cudaMemcpy(hvel, dvel, size3, cudaMemcpyDeviceToHost), 
		        "Failed to copy device to host in update!");
	dt = CalcTimeStep();
	cout << dt << endl;
	UpdateVelocityAndPosition(dpos, dvel, dforce, count, dt);
	//cout << "UpdateVelocityAndPosition" << endl;
	HandleBoundary(dpos, dvel, count);
	HandleBoundary2(dpos, dvel, count);
	//cout << "HandleBoundary2" << endl;
	HandleError(cudaMemcpy(hpos, dpos, size3, cudaMemcpyDeviceToHost), 
		        "Failed to copy device to host in update!");
	HandleError(cudaMemcpy(hvel, dvel, size3, cudaMemcpyDeviceToHost),
		        "Failed to copy device to host in update!");
	HandleError(cudaMemcpy(hdens, ddens, size1, cudaMemcpyDeviceToHost),
		"Failed to copy device to host in update!");
	//cout << "HandleError" << endl;
}


float3* Solver::GetPos()
{
	return hpos;
}

void Solver::SetBoundary(float x)
{
	pa.xmin = x;
}

int* Solver::GetNeighbourNum()
{
	return hneighbourNum;
}

float3* Solver::GetVel()
{
	return hvel;
}

float* Solver::GetDens()
{
	return hdens;
}