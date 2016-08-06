#pragma once
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include "GL/freeglut.h"

#pragma comment(lib, "freeglut.lib")

ofstream out("data.txt");

struct GLvector
{
	GLfloat fX;
	GLfloat fY;
	GLfloat fZ;
};

struct Node
{
	float value[8];
	GLvector pos;
};

extern const GLfloat a2fVertexOffset[8][3];
extern const GLint a2iEdgeConnection[12][2];
extern const GLfloat a2fEdgeDirection[12][3];
//a2iTetrahedronEdgeConnection lists the index of the endpoint vertices for each of the 6 edges of the tetrahedron
extern const GLint a2iTetrahedronEdgeConnection[6][2];

//a2iTetrahedronEdgeConnection lists the index of verticies from a cube 
// that made up each of the six tetrahedrons within the cube
extern const GLint a2iTetrahedronsInACube[6][4];
extern GLenum    ePolygonMode;
extern GLint aiTetrahedronEdgeFlags[16];
extern GLint a2iTetrahedronTriangles[16][7];
extern GLint aiCubeEdgeFlags[256];
extern GLint a2iTriangleConnectionTable[256][16];


class MCSurface
{
public:
	MCSurface(float* pos, float* value, GLvector corner1, GLvector corner2, float cellsize, float threshold, int count)
	{
		this->pos = pos;
		this->value = value;
		this->corner1 = corner1;
		this->corner2 = corner2;
		this->threshold = threshold;
		this->count = count;

		iDataSetSize[0] = static_cast<int>((corner2.fX - corner1.fX) / cellsize);
		iDataSetSize[1] = static_cast<int>((corner2.fY - corner1.fY) / cellsize);
		iDataSetSize[2] = static_cast<int>((corner2.fZ - corner1.fZ) / cellsize);

		fStepSize.fX = cellsize;// (corner2.fX - corner1.fX) / iDataSetSize[0];
		fStepSize.fY = cellsize;// (corner2.fY - corner1.fY) / iDataSetSize[1];
		fStepSize.fZ = cellsize;// (corner2.fZ - corner1.fZ) / iDataSetSize[2];

		normal1.fX = 0.0f;
		normal1.fY = 0.0f;
		normal1.fZ = 0.0f;

		normal2.fX = 0.0f;
		normal2.fY = 0.0f;
		normal2.fZ = 0.0f;

		grid = new Node[iDataSetSize[0] * iDataSetSize[1] * iDataSetSize[2]];

		if (nullptr == grid)
		{
			cerr << "Failed to allocate memory!" << endl;
			exit(1);
		}

		GLint iX, iY, iZ;
		for (iX = 0; iX < iDataSetSize[0]; iX++)
		{
			for (iY = 0; iY < iDataSetSize[1]; iY++)
			{
				for (iZ = 0; iZ < iDataSetSize[2]; iZ++)
				{
					grid[iX + iZ*iDataSetSize[0] + iY*iDataSetSize[0] * iDataSetSize[2]].pos.fX = iX*fStepSize.fX;// +corner1.fX;
					grid[iX + iZ*iDataSetSize[0] + iY*iDataSetSize[0] * iDataSetSize[2]].pos.fY = iY*fStepSize.fY;// +corner1.fY;
					grid[iX + iZ*iDataSetSize[0] + iY*iDataSetSize[0] * iDataSetSize[2]].pos.fZ = iZ*fStepSize.fZ;// +corner1.fZ;
					//vMarchCube(iX*fStepSize.fX, iY*fStepSize.fY, iZ*fStepSize.fY, fStepSize);
					for (int i = 0; i < 8; i++)
					{
						grid[iX + iZ*iDataSetSize[0] + iY*iDataSetSize[0] * iDataSetSize[2]].value[i] = 0.0f;
					}
				}
			}
		}

		int Xindex = 0;
		int Yindex = 0;
		int Zindex = 0;
		int index = 0;

		int iVertex;

		for (int i = 0; i < count; i++)
		{
			Xindex=static_cast<int>((pos[i * 3])/fStepSize.fX);
			Yindex=static_cast<int>((pos[i * 3 + 1])/fStepSize.fY);
			Zindex=static_cast<int>((pos[i * 3 + 2])/fStepSize.fZ);

			//srand((int))

			int neighbour = 3;
			for (int innerx = -neighbour; innerx <= neighbour; innerx++)
			{
				for (int innery = -neighbour; innery <= neighbour; innery++)
				{
					for (int innerz = -neighbour; innerz <= neighbour; innerz++)
					{
						if ((Xindex + innerx) < iDataSetSize[0] && (Xindex + innerx)>=0 &&
							(Yindex + innery) < iDataSetSize[1] && (Yindex + innery)>=0 &&
							(Zindex + innerz) < iDataSetSize[2] && (Zindex + innerz)>=0)
						{
							index = (Xindex + innerx) + (Zindex + innerz)*iDataSetSize[0] + (Yindex + innery)*iDataSetSize[0] * iDataSetSize[2];
							//cout << index << endl;
							Node &current = grid[index];
							for (iVertex = 0; iVertex < 8; iVertex++)
							{
								current.value[iVertex] +=
									fSample(current.pos.fX + a2fVertexOffset[iVertex][0] * fStepSize.fX,
									current.pos.fY + a2fVertexOffset[iVertex][1] * fStepSize.fY,
									current.pos.fZ + a2fVertexOffset[iVertex][2] * fStepSize.fZ, pos[i*3], pos[i*3+1], pos[i*3+2]);

								//cout << grid[index].value[iVertex] << endl;
							}
						}
					}
				}
			}
		}

	}

	~MCSurface()
	{
		delete[] grid;
	}

	GLvoid vMarchingCubes(ofstream& out)
	{
		GLint iX, iY, iZ, index;
		for (iX = 0; iX < iDataSetSize[0]; iX++)
			for (iY = 0; iY < iDataSetSize[1]; iY++)
				for (iZ = 0; iZ < iDataSetSize[2]; iZ++)
				{
					index = iZ + iY*iDataSetSize[2] + iX*iDataSetSize[1] * iDataSetSize[2];
					Node &current = grid[index];
					vMarchCube(current, fStepSize, out);
				}
	}

	GLvoid vMarchCube(Node node, GLvector fScale, ofstream& out)
	{
		extern GLint aiCubeEdgeFlags[256];
		extern GLint a2iTriangleConnectionTable[256][16];

		GLint iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
		GLfloat fOffset;
		GLvector sColor;
		GLfloat afCubeValue[8];
		GLvector asEdgeVertex[12];
		GLvector asEdgeNorm[12];

		for (iVertex = 0; iVertex < 8; iVertex++)
		{
			afCubeValue[iVertex] = node.value[iVertex];
		}
		iFlagIndex = 0;
		for (iVertexTest = 0; iVertexTest < 8; iVertexTest++)
		{
			if (afCubeValue[iVertexTest] <= threshold)
				iFlagIndex |= 1 << iVertexTest;
		}

		iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

		if (iEdgeFlags == 0)
		{
			return;
		}

		for (iEdge = 0; iEdge < 12; iEdge++)
		{
			if (iEdgeFlags & (1 << iEdge))
			{
				fOffset = fGetOffset(afCubeValue[a2iEdgeConnection[iEdge][0]],
					afCubeValue[a2iEdgeConnection[iEdge][1]], threshold);

				asEdgeVertex[iEdge].fX = node.pos.fX + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][0] + fOffset * a2fEdgeDirection[iEdge][0]) * fScale.fX;
				asEdgeVertex[iEdge].fY = node.pos.fY + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][1] + fOffset * a2fEdgeDirection[iEdge][1]) * fScale.fY;
				asEdgeVertex[iEdge].fZ = node.pos.fZ + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][2] + fOffset * a2fEdgeDirection[iEdge][2]) * fScale.fZ;

				//vGetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ);
			}
		}
		for (iTriangle = 0; iTriangle < 5; iTriangle++)
		{
			if (a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
				break;
			for (iCorner = 0; iCorner < 3; iCorner++)
			{
				iVertex = a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle + iCorner];
				//GLvector vcn; //current normal
				//vcn.fX = (asEdgeNorm[iVertex].fX + normal1.fX + normal2.fX) / 3.0f;
				//vcn.fY = (asEdgeNorm[iVertex].fY + normal1.fY + normal2.fY) / 3.0f;
				//vcn.fZ = (asEdgeNorm[iVertex].fZ + normal1.fZ + normal2.fZ) / 3.0f;

				//normal1.fX = normal2.fX;
				//normal1.fY = normal2.fY;
				//normal1.fZ = normal2.fZ;

				//normal2.fX = vcn.fX;
				//normal2.fY = vcn.fY;
				//normal2.fZ = vcn.fZ;
				//vGetColor(sColor, asEdgeVertex[iVertex], asEdgeNorm[iVertex]);
				//glColor3f(sColor.fX, sColor.fY, sColor.fZ);
				//glNormal3f(asEdgeNorm[iVertex].fX, asEdgeNorm[iVertex].fY, asEdgeNorm[iVertex].fZ);
				//glNormal3f(vcn.fX, vcn.fY, vcn.fZ);
				glVertex3f(asEdgeVertex[iVertex].fX, asEdgeVertex[iVertex].fY, asEdgeVertex[iVertex].fZ);

				//£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿£¿
				//out.setf(ios::fixed);
				//out <<setprecision(6) << "<" << asEdgeVertex[iVertex].fX << ", " << asEdgeVertex[iVertex].fY << ", " << asEdgeVertex[iVertex].fZ << ">, ";
				//out << setprecision(6) << "<" << vcn.fX << ", " << vcn.fY << ", " << vcn.fZ << ">" << (iCorner == 2 ? "" : ", ");

				out << asEdgeVertex[iVertex].fX << "," << asEdgeVertex[iVertex].fY << "," << asEdgeVertex[iVertex].fZ << " ";
			}
			out << endl;
		}
	}

	GLfloat fSample(GLfloat fX, GLfloat fY, GLfloat fZ, float posx, float posy, float posz)
	{
		GLfloat fResult = 0.0f;
		GLfloat fDx, fDy, fDz;
		//for (int i = 0; i < POINTCOUNT; i++)
		//{
		fDx = fX - posx;//sSourcePoint[i].fX;
		fDy = fY - posy;// sSourcePoint[i].fY;
		fDz = fZ - posz;// sSourcePoint[i].fZ;
		//cout << fDx << " " << fDy << " " << fDz << endl;
		float distance = fDx*fDx + fDy*fDy + fDz*fDz;
		if (distance < 2.0f)
		{
			fResult += 0.5f / (distance);
		}
		//}

		//cout << fResult << endl;

		return fResult;
	}

	GLfloat fGetOffset(GLfloat fValue1, GLfloat fValue2, GLfloat fValueDesired)
	{
		GLfloat fDelta = fValue2 - fValue1;

		if (fDelta == 0.0)
		{
			return 0.5f;
		}
		return (fValueDesired - fValue1) / fDelta;
	}

	GLvoid vGetColor(GLvector &rfColor, GLvector &rfPosition, GLvector &rfNormal)
	{
		GLfloat fX = rfNormal.fX;
		GLfloat fY = rfNormal.fY;
		GLfloat fZ = rfNormal.fZ;
		rfColor.fX = (fX > 0.0f ? fX : 0.0f) + (fY < 0.0f ? -0.5f*fY : 0.0f) + (fZ < 0.0f ? -0.5f*fZ : 0.0f);
		rfColor.fY = (fY > 0.0f ? fY : 0.0f) + (fZ < 0.0f ? -0.5f*fZ : 0.0f) + (fX < 0.0f ? -0.5f*fX : 0.0f);
		rfColor.fZ = (fZ > 0.0f ? fZ : 0.0f) + (fX < 0.0f ? -0.5f*fX : 0.0f) + (fY < 0.0f ? -0.5f*fY : 0.0f);
	}

	GLvoid vNormalizeVector(GLvector &rfVectorResult, GLvector &rfVectorSource)
	{
		GLfloat fOldLength;
		GLfloat fScale;

		fOldLength = sqrtf((rfVectorSource.fX * rfVectorSource.fX) +
			(rfVectorSource.fY * rfVectorSource.fY) +
			(rfVectorSource.fZ * rfVectorSource.fZ));

		if (fOldLength < 0.000001 && fOldLength>-0.000001f)
		{
			rfVectorResult.fX = rfVectorSource.fX;
			rfVectorResult.fY = rfVectorSource.fY;
			rfVectorResult.fZ = rfVectorSource.fZ;
		}
		else
		{
			fScale = 1.0f / fOldLength;
			rfVectorResult.fX = rfVectorSource.fX*fScale;
			rfVectorResult.fY = rfVectorSource.fY*fScale;
			rfVectorResult.fZ = rfVectorSource.fZ*fScale;
		}
	}

	GLvoid vGetNormal(GLvector &rfNormal, GLfloat fX, GLfloat fY, GLfloat fZ)
	{
		/*rfNormal.fX = fSample(fX - 0.01, fY, fZ) - fSample(fX + 0.01, fY, fZ);
		rfNormal.fY = fSample(fX, fY - 0.01, fZ) - fSample(fX, fY + 0.01, fZ);
		rfNormal.fZ = fSample(fX, fY, fZ - 0.01) - fSample(fX, fY, fZ + 0.01);*/
		int Xindex, Yindex, Zindex;
		Xindex = static_cast<int>(fX / fStepSize.fX);
		Yindex = static_cast<int>(fY / fStepSize.fY);
		Zindex = static_cast<int>(fZ / fStepSize.fZ);

		int index = index = Xindex + Zindex*iDataSetSize[0] + Yindex*iDataSetSize[0] * iDataSetSize[2];

		int left = index - 1;
		int front = index - iDataSetSize[0];

		rfNormal.fX = -(grid[left].value[3] - grid[index].value[2])/2.0f;

		rfNormal.fY = -(grid[index].value[3] - grid[index].value[7]);
		
		rfNormal.fZ = -(grid[front].value[3] - grid[index].value[0])/2.0f;

		vNormalizeVector(rfNormal, rfNormal);
	}

	//GLfloat fSample2(GLfloat fX, GLfloat fY, GLfloat fZ)
	//{
	//	GLfloat fResult = 0.0f;
	//	GLfloat fDx, fDy, fDz;

	//	//for (int i = 0; i < POINTCOUNT; i++)
	//	//{
	//	fDx = fX - posx;//sSourcePoint[i].fX;
	//	fDy = fY - posy;// sSourcePoint[i].fY;
	//	fDz = fZ - posz;// sSourcePoint[i].fZ;
	//	//cout << fDx << " " << fDy << " " << fDz << endl;
	//	float distance = fDx*fDx + fDy*fDy + fDz*fDz;
	//	if (distance < 1.0f)
	//	{
	//		fResult += 0.5f / (distance);
	//	}
	//	//}

	//	//cout << fResult << endl;

	//	return fResult;
	//}


private:
	GLvector corner1;
	GLvector corner2;
	float* value;
	float* pos;
	float threshold;
	GLint iDataSetSize[3];
	GLvector fStepSize;
	int count;
	float cellsize;
	Node* grid;

	GLvector normal1;
	GLvector normal2;

};
