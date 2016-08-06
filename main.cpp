/* */
//#include <GL.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <algorithm>
using namespace std;

#include "GL\freeglut.h"
#include "FreeImage.h"

#include "constant.h"
#include "Solver.h"
#include "MarchingCubes.h"

#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "FreeImage.lib")

ofstream outpn("particleNum.txt");            //add in 20160721 to recorder the neighbournumber

ofstream outVol;
ofstream outVar;

const int particleCount = 16 * 1024;

const int PicNum = 2000;

struct Vec3f
{
	float x;
	float y;
	float z;
};

Solver solver(particleCount);          //粒子数量

int cycle = 0;                     //查看模拟的帧数

bool isSaved = false;              //是否保存图片
bool isLight = false;              //是否打开光照，打开光照后，以球星（sphere）来表示每个粒子，否则以点（point）来表示每个粒子
bool isfps = true;                 //是否显示实时计算频率,在标题栏
bool isShowFps = true;             //在窗口里显示计算频率
bool isSurface = false;

bool bounFlag = false;

float xboun = 0.0f;

GLfloat light_position[] = { 25.0f, 15.0f, -50.0f, 0.0f };


chrono::high_resolution_clock::time_point st;          //每帧的开始时间：strat
chrono::high_resolution_clock::time_point en;          //每帧的结束时间：end

ofstream fout;

/*recorder the number of neighbour particles
 *date:20160721
*/
void WriteNeighbour(int* pnum)
{
	outpn << "///////////////////////////////" << endl;
	for (int i = 0; i < particleCount; i++)
	{
		outpn << pnum[i] << endl;
	}
}

/*
计算粒子密度的方差
时间：2016-07-25
*/
float GetVariance(float* arg, int n)
{
	float sum = 0.0f;

	for (int i = 0; i < n; i++)
	{
		sum += arg[i];
	}

	float variance = 0;
	float mean = sum / n;
	for (int i = 0; i < n; i++)
	{
		variance += ((arg[i] - mean)*(arg[i] - mean));
	}

	return variance;
}

/*
计算流体的体积
时间：2016-07-25
*/
float GetVolume(float* density, int n)
{
	float volume = 0.0f;
	float mass = 1.0f;
	for (int index = 0; index < n; index++)
	{
		volume += (mass / density[index]);
	}
	return volume;
}

/*
作用：用于保存OpenGL画出的图像
imgpath：保存图片的路径和名字
*/
bool SaveImage(char *imgpath)
{
	unsigned char *mpixels = new unsigned char[800 * 600 * 4];
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE, mpixels);
	glReadBuffer(GL_BACK);
	for (int i = 0; i<(int)800 * 600 * 4; i += 4)
	{

		mpixels[i] ^= mpixels[i + 2] ^= mpixels[i] ^= mpixels[i + 2];
	}
	FIBITMAP* bitmap = FreeImage_Allocate(800, 600, 32, 8, 8, 8);

	for (unsigned int y = 0; y<FreeImage_GetHeight(bitmap); y++)
	{
		BYTE *bits = FreeImage_GetScanLine(bitmap, y);
		for (unsigned int x = 0; x<FreeImage_GetWidth(bitmap); x++)
		{
			bits[0] = mpixels[(y * 800 + x) * 4 + 0];
			bits[1] = mpixels[(y * 800 + x) * 4 + 1];
			bits[2] = mpixels[(y * 800 + x) * 4 + 2];
			bits[3] = 255;
			bits += 4;
		}
	}
	bool bSuccess = FreeImage_Save(FIF_PNG, bitmap, imgpath, PNG_DEFAULT);
	FreeImage_Unload(bitmap);

	return bSuccess;

}

void RenderBitmapString(float x, float y, void *font,char *string) 
{
	char *c;
	glRasterPos2f(x, y);
	for (c = string; *c != '\0'; c++) 
	{
		glColor3f(1.0f, 1.0f, 1.0f);
		glutBitmapCharacter(font, *c);
	}
}

void init(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glShadeModel(GL_SMOOTH);

	outVol.open("volume.txt", ios::out);
	if (outVol.bad())
	{
		cerr << "Failed to open the file!" << endl;
		exit(1);
	}

	outVar.open("variance.txt", ios::out);
	if (outVar.bad())
	{
		cerr << "Failed to open the file!" << endl;
		exit(1);
	}
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	gluLookAt(0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	//画出粒子所在的长方体
	if (isLight)
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
	}
	glPushMatrix();
	glTranslatef(0.0f, -1.5f, 0.0f);
	glScalef(50.0f, 27.0f, 40.0f);
	glColor3f(1.0f, 1.0f, 1.0f);
	glutWireCube(1.0f);
	glPopMatrix();

	if (isLight)
	{
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHTING);
	}

	solver.Update();
	cycle++;
	float3* pos = solver.GetPos();   //获取粒子的位置

	float3* vel = solver.GetVel();

	float* dens = solver.GetDens();

	if (bounFlag)
	{
		xboun = xboun - 0.05f;
		if (xboun <= 0.5f)
		{
			bounFlag = false;
		}
	}
	else
	{
		xboun = xboun + 0.05f;

		if (xboun >= 19.5f)
		{
			bounFlag = true;
		}
	}
	//solver.SetBoundary(xboun);

	outVol << GetVolume(dens, particleCount) << endl;
	outVar << GetVariance(dens, particleCount) << endl;

	/*int* pnum = solver.GetNeighbourNum();
	WriteNeighbour(pnum);*/

	if (isSurface)
	{
		GLvector corner1;
		corner1.fX = 0.0f;
		corner1.fY = 0.0f;
		corner1.fZ = 0.0f;

		GLvector corner2;
		corner2.fX = 50.0f;
		corner2.fY = 30.0f;
		corner2.fZ = 40.0f;

		MCSurface surface((float*)pos, nullptr, corner1, 
			        corner2, 0.5f, 1.5f, particleCount);

		char filename[20];
		sprintf_s(filename, "%d.txt", cycle);
		fout.open(filename, ios::out);
		glPushMatrix();
		glTranslatef(-25, -15, -20);
		glBegin(GL_TRIANGLES);
		surface.vMarchingCubes(fout);
		glEnd();
		glPopMatrix();
		fout.close();
	}
	else
	{

		//判断是否开启光照，而作不同的选择
		if (isLight)
		{
			glLightfv(GL_LIGHT0, GL_POSITION, light_position);
			for (int pindex = 0; pindex < 16 * 1024; pindex++)
			{
				glPushMatrix();
				glTranslatef(pos[pindex].x - 25.0f, pos[pindex].y - 15.0f, pos[pindex].z - 20.0f);
				glutSolidSphere(0.5, 10, 10);
				glPopMatrix();
			}
		}
		else
		{
			glBegin(GL_POINTS);
			{
				for (int pindex = 0; pindex < particleCount; pindex++)
				{
					float norm = sqrtf(vel[pindex].x*vel[pindex].x + vel[pindex].y*vel[pindex].y + vel[pindex].z*vel[pindex].z);
					glColor3f(abs(vel[pindex].x) / norm, abs(vel[pindex].y/norm), abs(vel[pindex].z) / norm);
					glVertex3f(pos[pindex].x - 25.0f, pos[pindex].y - 15.0f, pos[pindex].z - 20.0f);

				}
				glColor3f(1.0f, 1.0f, 1.0f);
			}
			glEnd();
		}
	}

	//判断是否需要保存图片
	if (isSaved)
	{
		char fname[20];
		sprintf_s(fname, "%d.bmp", cycle);
		if (!SaveImage(fname))
		{
			cout << "Failed to save picture!" << endl;
		}
	}

	char text[32];

	//判断是否需要输出计算频率（在窗口的标题栏）
	if (isfps || isShowFps)
	{
		en = chrono::high_resolution_clock::now();
		chrono::duration<float> dura = en - st;
		st = en;	
		sprintf_s(text, "The framerate is %f: ", 1.0 / dura.count());
	}
	if (isfps)
	{
		glutSetWindowTitle(text);
	}

	if (isShowFps)
	{
		RenderBitmapString(-45.0f, -34.0f, (void*)GLUT_BITMAP_8_BY_13, text);
	}
	cout << cycle << endl;
	glutSwapBuffers();

	if (cycle == PicNum)
	{
		exit(0);
	}
}

void reshape(int width, int height)
{
	glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, static_cast<float>(width) / (height = (height == 0 ? 1 : height)), 0.01, 1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'L':
	case 'l':
		if (isLight)
		{
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHTING);
		}
		else
		{
			glEnable(GL_LIGHT0);
			glEnable(GL_LIGHTING);
		}
		isLight = !isLight;
		break;
	case 'S':
	case 's':
		isSaved = !isSaved;
		break;
	case 27:
		exit(0);
		break;
	case 'f':
	case 'F':
		isfps = !isfps;
		glutSetWindowTitle("SPH");
		break;
	case 'W':
	case 'w':
		isShowFps = !isShowFps;
		break;

	case 'U':
	case 'u':
		isSurface = !isSurface;
		break;
	default:
		break;
	}

	glutPostRedisplay();
}

void special(int key, int x, int y)
{
	switch (key)
	{
	default:
		break;
	}

	glutPostRedisplay();
}

void processMenuEvents(int option)
{
	switch (option)
	{
	case 1:
		isSaved = !isSaved;
		break;
	case 2:
		if (isLight)
		{
			glDisable(GL_LIGHT0);
			glDisable(GL_LIGHTING);
			glutChangeToMenuEntry(2, "Light On", 2);

		}
		else
		{
			glEnable(GL_LIGHT0);
			glEnable(GL_LIGHTING);
			glutChangeToMenuEntry(2, "Light Off", 2);
		}
		isLight = !isLight;
		break;
	case 3:
		exit(0);
		break;

	case 4:
		isfps = !isfps;
		glutSetWindowTitle("SPH");
		break; 
	case 5:
		isShowFps = !isShowFps;
		break;
	case 6:
		isSurface = !isSurface;
		break;
	default:
		break;
	}
}

void createGLUTMenus(int& menu)
{
	menu = glutCreateMenu(processMenuEvents);

	glutAddMenuEntry("Save Picture",1);
	if (isLight)
	{
		glutAddMenuEntry("Light off", 2);
	}
	else
	{
		glutAddMenuEntry("Light on", 2);
	}
	glutAddMenuEntry("Exit",3);
	glutAddMenuEntry("FPS1",4);
	glutAddMenuEntry("FPS2", 5);
	glutAddMenuEntry("Surface", 6);

	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void mouse(int button, int state, int x, int y)
{
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	case GLUT_MIDDLE_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	case GLUT_RIGHT_BUTTON:
		if (GLUT_DOWN == state)
		{
		}
		else if (GLUT_UP == state)
		{
		}
		break;


	default:
		break;
	}

	glutPostRedisplay();
}

void motion(int x, int y)
{
}

void passivemotion(int x, int y)
{
}


int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(800, 600);
	glutCreateWindow(argv[0]);
	FreeImage_Initialise(true);
	init();

	st = chrono::high_resolution_clock::now();

	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);

	int menu;
	createGLUTMenus(menu);

	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passivemotion);

	glutMainLoop();

	glutDetachMenu(GLUT_RIGHT_BUTTON);
	glutDestroyMenu(menu);

	outVol.close();
	outVar.close();

	return 0;
}

