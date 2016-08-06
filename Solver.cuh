/*this file */
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Paras.cuh"
#include "math_functions.h"


#define PI 3.141592657f

#define SQR(x)					((x) * (x))
#define CUBE(x)					((x) * (x) * (x))
#define POW6(x)					(CUBE(x) * CUBE(x))
#define POW9(x)					(POW6(x) * CUBE(x))

#define BOUNDARY  0          //设置一个很大的值用来证明该粒子位于边界处，free surface

__constant__ Paras para;

inline __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __device__ float operator*(float3 a, float3 b)
{
	return (a.x*b.x + a.y*b.y + a.z*b.z);
}

inline __device__ float3 operator*(float a, float3 b)
{
	return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ float kernel(float3 r, float h)
{
	return 315.0f / (64.0f * PI * POW9(h)) * CUBE(SQR(h) - r*r);
}

__device__ float3 kernel_grident(float3 r, float h)
{
	return -945.0f / (32.0f * PI * POW9(h)) * SQR(SQR(h) - r*r) * r;
}


__device__ float laplacian_kernel(float3 r, float h)
{
	return 45.0f / (PI * POW6(h)) * (h - sqrtf(r.x*r.x + r.y*r.y + r.z*r.z));
}

__device__ int3 cudaCalcGridPos(float3 pos)
{
	int3 gridPos;
	gridPos.x = floor((pos.x - para.worldOrigin.x) / para.cellSize.x);
	
	gridPos.y = floor((pos.y - para.worldOrigin.y) / para.cellSize.y);

	gridPos.z = floor((pos.z - para.worldOrigin.z) / para.cellSize.z);

	return gridPos;
}

__device__ int cudaCalcGridHash(int3 gridPos)
{
	return (gridPos.z * para.gridSize.x + gridPos.y * para.gridSize.x * para.gridSize.z + gridPos.x);
}

/*__device__ unsigned int cudaCalcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (50 - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (30 - 1);
	gridPos.z = gridPos.z & (40 - 1);
	return __umul24(__umul24(gridPos.y, 40), 50) + __umul24(gridPos.z, 50) + gridPos.x;
}*/

__global__ void cudaCalHash(unsigned int* index, unsigned int* hash, float3* pos, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x; 
	if (tid >= count)
	{
		return;
	}

	int3 gridPos = cudaCalcGridPos(pos[tid]);
	unsigned int ash = cudaCalcGridHash(gridPos);

	hash[tid] = ash;
	index[tid] = tid;
}

__global__ void cudaReorderDataAndFindCellStart(unsigned int *cellstart,
												unsigned int *cellend,
												float3* spos,
												float3* svel, 
												unsigned int* hash,
												unsigned int* index,
												float3* pos, 
												float3* vel, 
												unsigned int count)
{
	extern __shared__ int sharedHash[];
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int _hash;
	if (tid < count)
	{
		_hash = hash[tid];

		sharedHash[threadIdx.x + 1] = _hash;
		if (tid > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = hash[tid - 1];
		}

	}
	__syncthreads();

	if (tid < count)
	{
		if (tid == 0 || _hash != sharedHash[threadIdx.x])
		{
			cellstart[_hash] = tid;
			if (tid > 0)
			{
				cellend[sharedHash[threadIdx.x]] = tid;
			}
		}
		if (tid == (count - 1))
		{
			cellend[_hash] = tid + 1;
		}

		int sortedIndex = index[tid];
		float3 _pos = pos[sortedIndex];
		float3 _vel = vel[sortedIndex];

		spos[tid] = _pos;
		svel[tid] = _vel;
	}
}

__device__ float length(float3 a)
{
	return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

/*
计算支集半径内粒子的数目
时间：2016-07-22
*/
__device__ int cudaAddNeighNumber(unsigned int tid, int3 gridPos,
								  float3 pos, unsigned int* cellstart,
								  unsigned int* cellend, float3* spos)
{
	int num = 0;
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 _pos = spos[j];
			float3 deltapos = pos - _pos;
			if (length(deltapos) <= para.h)
			{
				num = num + 1;
			}
		}
	}

	return num;
}


/*
计算支集半径内粒子的数目
时间：2016-07-22
*/
__global__ void cudaCalcNeighNumber(unsigned int* cellstart,
									unsigned int* cellend, float3* spos,
									unsigned int count, int* pnum)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= count)
	{
		return;
	}
	int num = 0;
	float3 _pos = spos[tid];

	int3 gridPos = cudaCalcGridPos(_pos);

	int scale = para.h + 1;

	if ((_pos.x <= para.xmax - para.h)
		&& (_pos.x >= para.xmin + para.h)
		&& (_pos.y <= para.ymax - para.h)
		&& (_pos.y >= para.ymin + para.h)
		&& (_pos.z <= para.zmax - para.h)
		&& (_pos.z >= para.zmin + para.h))
	{
		for (int z = -scale; z <= scale; z++)
		{
			for (int y = -scale; y <= scale; y++)
			{
				for (int x = -scale; x <= scale; x++)
				{
					int3 neighbour = gridPos + make_int3(x, y, z);

					if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
					{
						num += cudaAddNeighNumber(tid, neighbour, _pos, cellstart, cellend, spos);
					}
				}
			}
		}
		pnum[tid] = num;

	}
	else
	{
		pnum[tid] = BOUNDARY;
	}
}

__device__ float cudaAddDensity(unsigned int tid, int3 gridPos, 
	                            float3 pos, unsigned int* cellstart, 
								unsigned int* cellend, float3* spos/*,
								int *num*/)
{

	float _dens = 0.0f;
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 _pos = spos[j];
			float3 deltapos = pos - _pos;
			if (length(deltapos) <= para.h)
			{
				_dens += para.mass*kernel(deltapos, para.h);
				//(*num) = (*num) + 1;
			}
		}
	}

	return _dens;
}




__global__ void cudaCalcDensity(float* dens, unsigned int* cellstart, 
	                            unsigned int* cellend, float3* spos, 
								unsigned int count, int* pnum)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= count)
	{
		return;
	}
	float3 _pos = spos[tid];

	float _dens = 0.0f;
	int3 gridPos = cudaCalcGridPos(_pos);

	int scale = para.h + 1;

	if (pnum[tid]>=para.threshold)
	{
		for (int z = -scale; z <= scale; z++)
		{
			for (int y = -scale; y <= scale; y++)
			{
				for (int x = -scale; x <= scale; x++)
				{
					int3 neighbour = gridPos + make_int3(x, y, z);

					if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
					{
						_dens += cudaAddDensity(tid, neighbour, _pos, cellstart, cellend, spos/*, &neiparnum*/);
					}
				}
			}
		}

		dens[tid] = _dens;
		
	}
	else
	{
		dens[tid] = para.restDens;
	}
}

//__global__ void cudaCalcPressure(float* press, float* dens, unsigned int count)
//{
//	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
//
//	if (tid >= count)
//	{
//		return;
//	}
//
//	press[tid] = para.k*(dens[tid] - para.restDens);
//}

//__global__ void cudaCalcPressure(float* press, float* dens, unsigned int count)
//{
//	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
//
//	if (tid >= count)
//	{
//		return;
//	}
//
//	press[tid] = para.k*(dens[tid] - para.restDens);
//}

//__global__ void cudaCalcPressure(float* press, float* dens, unsigned int count)
//{
//	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
//
//	if (tid >= count)
//	{
//		return;
//	}
//
//	press[tid] = para.soundSpeed*para.soundSpeed*dens[tid];
//}

__global__ void cudaCalcPressure(float* press, float* dens, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= count)
	{
		return;
	}

	press[tid] = para.beta*(powf(dens[tid] / para.restDens, para.gama) - 1);
}

__device__ float3 cudaAddForce(unsigned int tid, int3 gridPos, float3 pos, float3 vel, 
	float press, float* pre, unsigned int* cellstart, unsigned int* cellend, float3* spos, float3* svel, float* dens)
{
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	unsigned int _hash = cudaCalcGridHash(gridPos);
	unsigned int startIndex = cellstart[_hash];

	if (startIndex != 0xffffffff)
	{
		unsigned int endIndex = cellend[_hash];
		for (int j = startIndex; j != endIndex; j++)
		{
			float3 _pos = spos[j];
			float3 dis = pos - _pos;
			if (length(dis) <= para.h)
			{

				float3 _vel = svel[j];
				float _dens = dens[j];
				float _press = pre[j];
				float3 deltavel = _vel - vel;


				force += para.mu*para.mass*deltavel / _dens*laplacian_kernel(dis, para.h)*0.5;
				force += -para.mass*(press + _press) / (2.0f * _dens)*kernel_grident(dis, para.h)*0.2f;
			}
		}
	}

	return force;
}

__device__ float3 collideSpheres(float3 posA, float3 posB, float3 velA, float3 velB,
	float radiusA, float radiusB, float attraction)
{
	// calculate relative position
	float3 relPos = (posB - posA);

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		//float3 tanVel = relVel - (relVel*norm * norm);

		// spring force
		force = -para.spring*(collideDist - dist) * norm;

		// dashpot (damping) force
		force += para.damping*relVel;          //domainate

		// tangential shear force
		//force += para.shear*tanVel;

		// attraction
		force += attraction*relPos;
	}

	return force;
}

__device__ float3 cudaCollideCell(int3 gridPos, unsigned int index,
	float3 pos, float3 vel, float3 *oldPos,
	float3 *oldVel, unsigned int *cellStart,
	unsigned int *cellEnd)
{
	int gridHash = cudaCalcGridHash(gridPos);

	// get start of bucket for this cell
	unsigned int startIndex = cellStart[gridHash];

	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		unsigned int endIndex = cellEnd[gridHash];

		for (unsigned int j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = oldPos[j];
				float3 vel2 = oldVel[j];

				// collide two spheres
				force += collideSpheres(pos, pos2, vel, vel2, para.particleRadius, 
					                    para.particleRadius, para.attraction);
			}
		}
	}

	return force;
}

__global__ void cudaCalcForce(float3* force, float3* spos, float3* svel, 
	                          float3* vel, float* press, float* dens, 
							  unsigned int* index, unsigned int* cellstart, 
							  unsigned int* cellend, unsigned int count, 
							  int *pnum)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	float3 _pos = spos[tid];
	float3 _vel = svel[tid];
	float _press = press[tid];

	int3 gridPos = cudaCalcGridPos(_pos);

	float3 _force = make_float3(0.0f, 0.0f, 0.0f);

	int scale = para.h + 1;

	for (int z = -scale; z <= scale; z++)
	{
		for (int y = -scale; y <= scale; y++)
		{
			for (int x = -scale; x <= scale; x++)
			{
				int3 neighbour = gridPos + make_int3(x, y, z);
				if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.z >= 0)
				{
					if (pnum[tid] >= para.threshold)
					{
						_force += cudaAddForce(tid, neighbour, _pos, _vel, _press, press, cellstart, cellend, spos, svel, dens);
					}
					else
					{
						_force += para.damping*cudaCollideCell(neighbour, tid, _pos, _vel,
							spos, svel, cellstart, cellend);
					}
				}
			}
		}
	}


	force[tid] = _force - para.gravity;

	unsigned int originalIndex = index[tid];
	vel[originalIndex] = make_float3(_vel.x + _force.x, _vel.y + _force.y, _vel.z + _force.z);
}

__global__ void cudaUpdateVelocityAndPosition(float3* pos, float3* vel, float3* force, unsigned int count, float dt)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	vel[tid] += force[tid] * para.dt;
	pos[tid] += vel[tid] * para.dt;
}

__global__ void cudaHandleBoundary(float3* pos, float3* vel, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	float x = pos[tid].x;
	float y = pos[tid].y;
	float z = pos[tid].z;

	if ( x > para.xmax-2.5f)
	{
		pos[tid].x = para.xmax - (x - (para.xmax - 2.5f));
		vel[tid].x = -para.boundaryDamping*vel[tid].x;
	}

	if ( x < para.xmin+0.5f)
	{
		pos[tid].x = para.xmin + para.xmin + 0.5f - x;
		vel[tid].x = -para.boundaryDamping*vel[tid].x;
	}

	if ( y > para.ymax - 3.5f)
	{
		pos[tid].y = para.ymax - (y - (para.ymax - 3.5f));
		vel[tid].y = -para.boundaryDamping*vel[tid].y;
	}

	if (y < para.ymin + 0.5f)
	{
		pos[tid].y = para.ymin + para.ymin + 0.5f - y;
		vel[tid].y = -para.zdamp*para.boundaryDamping*vel[tid].y;
	}

	if (z > para.zmax - 2.5f)
	{
		pos[tid].z = para.zmax - (z - (para.zmax - 2.5f));
		vel[tid].z = -para.boundaryDamping*vel[tid].z;
	}

	if (z < para.zmin + 0.5f)
	{
		pos[tid].z = 0.2f*(para.zmin + para.zmin + 0.5f - z);
		vel[tid].z = -para.boundaryDamping*vel[tid].z;
	}
}

__global__ void cudaHandleBoundary2(float3* pos, float3* vel, unsigned int count)
{
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= count)
	{
		return;
	}

	float x = pos[tid].x;
	float y = pos[tid].y;
	float z = pos[tid].z;

	if (x > para.xmax-2.5)
	{
		pos[tid].x = para.xmax - 2.5f; 
		vel[tid].x = -para.boundaryDamping*vel[tid].x;
	}

	if (x < para.xmin+0.5)
	{
		pos[tid].x = para.xmin + 0.5f;
		vel[tid].x = -para.boundaryDamping*vel[tid].x;
	}

	if (y > para.ymax-3.5f)
	{
		pos[tid].y = para.ymax - 3.5f;
		vel[tid].y = -para.boundaryDamping*vel[tid].y;
	}

	if (y < para.ymin+0.5f)
	{
		pos[tid].y = para.ymin + 0.5f;
		vel[tid].y = -para.zdamp*para.boundaryDamping*vel[tid].y;
	}

	if (z > para.zmax-2.5f)
	{
		pos[tid].z = para.zmax  -2.5f;
		vel[tid].z = -para.boundaryDamping*vel[tid].z;
	}

	if (z < para.zmin+0.5f)
	{
		pos[tid].z = para.zmin + 0.5f; 
		vel[tid].z = -para.boundaryDamping*vel[tid].z;
	}
}

