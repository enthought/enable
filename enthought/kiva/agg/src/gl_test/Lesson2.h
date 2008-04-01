#ifndef LESSON2_H
#define LESSON2_H

#include <windows.h>		// Header File For Windows

#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include <gl\glut.h>



// Implement your own version of this function
int DrawGLScene(GLvoid);


// Functions implemented in Lesson2.cpp
GLvoid ReSizeGLScene(GLsizei width, GLsizei height);
int InitGL(GLvoid);
GLvoid KillGLWindow(GLvoid);
BOOL CreateGLWindow(char* title, int width, int height, int bits, bool fullscreenflag);
int DefaultDrawGLScene(GLvoid);

int WINAPI WinMain(	HINSTANCE	hInstance,
					HINSTANCE	hPrevInstance,
					LPSTR		lpCmdLine,
					int			nCmdShow);



#endif /* LESSON2_H */


