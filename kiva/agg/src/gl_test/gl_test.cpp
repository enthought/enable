

#include "Lesson2.h"
#include "gl_graphics_context.h"
using namespace kiva;

#define WIDTH 640
#define HEIGHT 480

int OrigDrawGLScene(GLvoid)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
	glTranslatef(-1.5f,0.0f,-6.0f);						// Move Left 1.5 Units And Into The Screen 6.0
	glBegin(GL_TRIANGLES);								// Drawing Using Triangles
		glVertex3f( 0.0f, 1.0f, 0.0f);					// Top
		glVertex3f(-1.0f,-1.0f, 0.0f);					// Bottom Left
		glVertex3f( 1.0f,-1.0f, 0.0f);					// Bottom Right
	glEnd();											// Finished Drawing The Triangle
	glTranslatef(3.0f,0.0f,0.0f);						// Move Right 3 Units
	glBegin(GL_QUADS);									// Draw A Quad
		glVertex3f(-1.0f, 1.0f, 0.0f);					// Top Left
		glVertex3f( 1.0f, 1.0f, 0.0f);					// Top Right
		glVertex3f( 1.0f,-1.0f, 0.0f);					// Bottom Right
		glVertex3f(-1.0f,-1.0f, 0.0f);					// Bottom Left
	glEnd();											// Done Drawing The Quad
	return TRUE;										// Keep Going

}


int KivaDrawGLScene(GLvoid)
{
	
	gl_graphics_context gc(WIDTH, HEIGHT);
	gc.gl_init();
	
	// XXX: Verify antialiasing from python
	//gc.set_antialias(1);

	gc.set_fill_color(agg24::rgba(1.0, 0.0, 0.0));
	gc.set_stroke_color(agg24::rgba(0.0, 1.0, 0.0));
	gc.set_line_width(1.0);
	gc.move_to(100.0, 100.0);
	gc.line_to(100.0, 200.0);
	gc.line_to(200.0, 200.0);
	gc.close_path();
	gc.draw_path(FILL_STROKE);

	gc.begin_path();
	gc.line_to(50, 50);
	gc.line_to(75, 75);
	gc.line_to(275, 75);
	gc.line_to(275, 50);
	gc.close_path();
	gc.draw_path(FILL_STROKE);
	
	return TRUE;
}


int DrawGLScene(GLvoid)
{
	//return OrigDrawGLScene();

	return KivaDrawGLScene();
}