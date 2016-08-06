#include "Solver.cuh"
#include <iostream>
using namespace std;
#include "thrust\sort.h"
#include "thrust\device_ptr.h"
#include "thrust\for_each.h"
//#include "thrust\iterator\zip_iterator.h"

#define getLastCudaError(msg)      \
		__getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, 
                               const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, 
			"%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

extern "C"
{

	void HandleError(cudaError_t status, char* message)
	{
		if (cudaSuccess == status)
		{
			return;
		}

		//cout << message << endl;
		//exit(1);
	}

	void SetParas(Paras *p)
	{
		HandleError(cudaMemcpyToSymbol(para, p, sizeof(Paras)), 
			        "Failed to copy Symbol!");
		getLastCudaError("kernel execute failed!");
	}

	void CalHash(unsigned int* index, unsigned int* hash, 
		         float3* pos, unsigned int count)
	{
		cudaCalHash <<<32, 512 >>>(index, hash, pos, count);

		getLastCudaError("kernel execute failed!");
	}
	void SortParticles(unsigned int *hash, unsigned int *index, 
		               unsigned int count)
	{
		thrust::sort_by_key(thrust::device_ptr<unsigned int>(hash),
			thrust::device_ptr<unsigned int>(hash + count),
			thrust::device_ptr<unsigned int>(index));
		getLastCudaError("kernel execute failed!");
	}

	void ReorderDataAndFindCellStart(unsigned int* cellstart,
                                     unsigned int* cellend,
									 float3* spos,float3* svel,
									 unsigned int* hash,unsigned int* index,
									 float3* pos, float3* vel,
									 unsigned int count, unsigned int gridNum)
	{
		cudaMemset(cellstart, 0xffffffff, gridNum*sizeof(unsigned int));
		unsigned int memsize = sizeof(unsigned int)*(512 + 1);
		cudaReorderDataAndFindCellStart <<<32, 512, memsize>>>(cellstart, cellend, spos,
			svel, hash, index, pos, vel, count);

		getLastCudaError("kernel execute failed!");
	}

	void CalcNeighNumber(unsigned int* cellstart,
		                 unsigned int* cellend, float3* spos,
		                 unsigned int count, int* pnum)
	{
		cudaCalcNeighNumber <<<64, 256 >>>(cellstart, cellend, 
			                                spos, count, pnum);

		getLastCudaError("kernel execute failed!");
	}

	void CalcDensity(float* dens, unsigned int* cellstart, unsigned int* cellend, float3 *spos, unsigned int count, int* pnum)
	{
		cudaCalcDensity<<<64, 256>>>(dens, cellstart, cellend, spos, count, pnum);

		getLastCudaError("kernel execute failed!");
	}

	void CalcPressure(float* press, float* dens, unsigned int count)
	{
		cudaCalcPressure <<<32, 512 >>>(press, dens, count);

		getLastCudaError("kernel execute failed!");
	}

	void CalcForce(float3* force, float3* spos, float3* svel, 
		           float3* vel, float* press, float* dens, 
				   unsigned int* index, unsigned int* cellstart, 
				   unsigned int* cellend, unsigned int count, int* pnum)
	{
		cudaCalcForce <<<32, 512 >>>(force, spos, svel, vel, press, dens, 
			                         index, cellstart, cellend, count, pnum);
		//(float3* force, float3* spos, float3* svel, float3* vel, float* press, 
		//float* dens, int* index, int* cellstart, int* cellend, int count)

		getLastCudaError("kernel execute failed!");
	}

	void UpdateVelocityAndPosition(float3* pos, float3* vel, float3* force, unsigned int count, float dt)
	{
		cudaUpdateVelocityAndPosition <<<32, 512 >>>(pos, vel, force, count, dt);

		getLastCudaError("kernel execute failed!");
	}

	void HandleBoundary(float3* pos, float3* vel, unsigned int count)
	{
		cudaHandleBoundary <<<32, 512 >>>(pos, vel, count);

		getLastCudaError("kernel execute failed!");
	}

	void HandleBoundary2(float3* pos, float3* vel, unsigned int count)
	{
		cudaHandleBoundary2 <<<32, 512 >>>(pos, vel, count);

		getLastCudaError("kernel execute failed!");
	}
}