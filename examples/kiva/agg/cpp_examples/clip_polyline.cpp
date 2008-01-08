#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>

#include "agg_color_rgba8.h"
#include "agg_pixfmt_rgb24.h"

#include "agg_rendering_buffer.h"

#include "agg_scanline_u.h"
#include "agg_scanline_bin.h"
#include "agg_renderer_scanline.h"

#include "agg_rasterizer_scanline_aa.h"

#include "agg_renderer_outline_aa.h"
#include "agg_rasterizer_outline_aa.h"

#include "agg_renderer_primitives.h"
#include "agg_rasterizer_outline.h"

#include "agg_renderer_base.h"
#include "agg_path_storage.h"
#include "agg_conv_stroke.h"
#include "agg_conv_clip_polyline.h"
#include "agg_conv_dash.h"

enum cap_type { CAP_ROUND, CAP_BUTT, CAP_SQUARE };
enum join_type { JOIN_ROUND, JOIN_MITER, JOIN_BEVEL };


/* Signal Generators */
agg::path_storage sin_path(int depth, int width)
{
	agg::path_storage path;
	for (double y=0;y<depth;y++)
	{
		double x = sin(y/depth*6.28) * width/2.0 + width/2.0;
		if (y==0)
			path.move_to(x, y);
		else
			path.line_to(x, y);
	}

	return path;
}

agg::path_storage rand_path(int depth, int width)
{
	agg::path_storage path;
	path.move_to(0, 0);
	for (double y=0;y<depth;y++)
	{
		double x = (((float)rand())/RAND_MAX * width);
		if (y==0)
			path.move_to(x, y);
		else
			path.line_to(x, y);
	}

	return path;
}

/* Helper Function for saving files */
bool write_ppm(const unsigned char* buf, 
               unsigned width, 
               unsigned height, 
               const char* file_name)
{
    FILE* fd = fopen(file_name, "wb");
    if(fd)
    {
        fprintf(fd, "P6 %d %d 255 ", width, height);
        fwrite(buf, 1, width * height * 3, fd);
        fclose(fd);
        return true;
    }
    return false;
}

typedef agg::renderer_base<agg::pixfmt_bgr24> bgr24_renderer_type;

class stroke_path_example
{
public:
	bool alias;
	double width;
	cap_type cap;
	join_type join;
	agg::rgba8 color;
	double* dash;
	int Ndash;
	double phase;

	int img_width;
	int img_height;
	agg::rgba8 background_color;

	unsigned char* mem_buffer;
	agg::rendering_buffer buffer;
	agg::pixfmt_bgr24 pixf;
	typedef bgr24_renderer_type renderer_type;
	renderer_type renderer_base;

	stroke_path_example::stroke_path_example(int img_width=500, int img_height=500, 
		                bool alias=0,double width=1.0, 
						cap_type cap=CAP_ROUND, join_type join=JOIN_BEVEL, 
						double* dash=NULL, int Ndash=0, double phase=0):
						img_width(img_width), img_height(img_height),
						alias(alias), width(width), cap(cap), join(join),
						dash(dash), Ndash(Ndash), phase(phase),
						pixf(buffer), renderer_base(pixf)
	{
		
		mem_buffer = (unsigned char*) malloc(img_width*img_height*3);	
		buffer.attach(mem_buffer, img_width, img_height, img_width*3);
		renderer_base.clip_box(0,0, img_width, img_height);
		

		this->color = agg::rgba(0, 0, 0, 1);
		this->background_color = agg::rgba(1, 1, 1, 1);
		this->clear();
	}

	~stroke_path_example()
	{
		free(mem_buffer);
	}

	void clear()
	{
		this->renderer_base.clear(background_color);
	}


	template<class PathType>
	void stroke_path(PathType& path)
	{
		// short circuit for transparent or 0 width lines
		if (this->color.a == 0 || this->width == 0.0)
			return;
		
		// handle dash/no-dash selection.
		if(this->Ndash==0)
		{
			// no dash -- don't do anything.
			this->stroke_path2(path);
		}
		else
		{
			// Set up the dashed path.
			agg::conv_dash<PathType> dash_path(path);
			for (int i=0; i<this->Ndash;i+=2)
			{
				dash_path.add_dash(this->dash[i],this->dash[i+1]);				
			}
			dash_path.dash_start(this->phase);

			this->stroke_path2(dash_path);
		}
	}

	template<class PathType>
	void stroke_path2(PathType& path)
	{
		if (this->alias)
		{  
			if ( this->width <= 1.0)
			{
				// ignore cap and join type here.
				this->stroke_path_outline(path);
			}
			else if ( this->alias &&
					  this->width <=10.0 && 
					  (this->cap == CAP_ROUND || this->cap == CAP_BUTT) &&
					  this->join == JOIN_MITER )
			{
				// fix me: how to force this to be aliased???
				this->stroke_path_outline_aa(path);
			}
			else
			{
				// fix me: This appears to be anti-aliased still.
				typedef agg::renderer_scanline_bin_solid<renderer_type> renderer_bin_type;
				renderer_bin_type renderer(this->renderer_base);
			    agg::scanline_bin scanline;

				this->stroke_path_scanline_aa(path, renderer, scanline);
			}
		}
		else // anti-aliased
		{ 
			if ( (this->cap == CAP_ROUND || this->cap == CAP_BUTT) &&
				 this->join == JOIN_MITER )
			{
				this->stroke_path_outline_aa(path);
			}
			else
			{
				typedef agg::renderer_scanline_aa_solid<renderer_type> renderer_aa_type;
				renderer_aa_type renderer(this->renderer_base);
				agg::scanline_u8 scanline;

				this->stroke_path_scanline_aa(path, renderer, scanline);
			}
		}

	}

	template<class PathType>
	void stroke_path_outline(PathType& path)
	{
		typedef agg::renderer_primitives<renderer_type> primitives_renderer_type;
		typedef agg::rasterizer_outline<primitives_renderer_type> rasterizer_type;
		
		primitives_renderer_type primitives_renderer(this->renderer_base);
		primitives_renderer.line_color(this->color);
		rasterizer_type rasterizer(primitives_renderer);		
		rasterizer.add_path(path);
	}

	template<class PathType>
	void stroke_path_outline_aa(PathType& path)
	{
		// fix me: How do you render aliased lines with this?
		
		// rasterizer_outline_aa algorithm only works for
		// CAP_ROUND or CAP_BUTT.  It also only works for JOIN_MITER
		
		typedef agg::renderer_outline_aa<renderer_type> outline_renderer_type;
		typedef agg::rasterizer_outline_aa<outline_renderer_type> rasterizer_type;
			
		// fix me: scale width by ctm
		agg::line_profile_aa profile(this->width, agg::gamma_none());
			
	    outline_renderer_type renderer(this->renderer_base, profile);
		renderer.color(this->color); 
		rasterizer_type rasterizer(renderer);
		
        if (this->cap == CAP_ROUND)
        {
	        rasterizer.round_cap(true);
        }
        else if (this->cap == CAP_BUTT)
        {    //default behavior
        }

		// fix me: not sure about the setting for this...
		rasterizer.accurate_join(false);

		rasterizer.add_path(path);
	}

	template<class PathType, class RendererType, class ScanlineType>
	void stroke_path_scanline_aa(PathType& path, RendererType& renderer,
								 ScanlineType& scanline)
	{	
		agg::rasterizer_scanline_aa<> rasterizer;		
    
		agg::conv_stroke<PathType> stroked_path(path);
		// fix me: scale width by ctm
		stroked_path.width(this->width);
		rasterizer.add_path(stroked_path);
		
		renderer.color(this->color);
		agg::render_scanlines(rasterizer, scanline, renderer);
	}


	void save(char* file_name)
	{
		write_ppm(this->mem_buffer, this->img_width, this->img_height, file_name);
	}
};


int main()
{
	time_t t1, t2;
	bool save_output = 0;
	int M = 500;
	int N = 500;
	//agg::path_storage raw_path = sin_path(M,N);
	agg::path_storage raw_path = rand_path(M,N);

	double dash[] = {10,10};
	int Ndash = 2;
	double phase = 0.0;

	stroke_path_example ex(M,N);

	// aliased outline
	ex.clear();
	ex.alias = 1;
	ex.width = 1.0;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("outline aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("outline_aliased.ppm");

	// aliased dashed outline
	ex.clear();
	ex.alias = 1;
	ex.Ndash = Ndash;
	ex.dash = dash;
	ex.phase = phase;
	ex.width = 1.0;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("outline dash aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("outline_dash_aliased.ppm");

	// aliased outline_aa
	ex.clear();
	ex.alias = 1;
	ex.Ndash = 0;
	ex.width = 2.0;
	ex.cap = CAP_BUTT;
	ex.join = JOIN_MITER;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("outline_aa aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("outline_aa_aliased.ppm");

	// aliased scanline_aa
	// width must be > 1 for aliased drawing to get to scanline renderer.
	// JOIN_BEVEL is the other attribute used here to get to the scanline renderer.
	ex.clear();
	ex.alias = 1;
	ex.Ndash = 0;
	ex.width = 2.0;
	ex.cap = CAP_ROUND;
	ex.join = JOIN_BEVEL;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("scanline_aa aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("scanline_aa_aliased.ppm");

	// anti-aliased outline_aa
	ex.clear();
	ex.alias = 0;
	ex.Ndash = 0;
	ex.width = 1.0;
	ex.cap = CAP_BUTT;
	ex.join = JOIN_MITER;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("outline_aa anti-aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("outline_aa.ppm");

    // anti-aliased scanline_aa
	// JOIN_BEVEL is the attribute used here to get to the scanline renderer.
	ex.clear();
	ex.alias = 0;
	ex.Ndash = 0;
	ex.width = 1.0;
	ex.cap = CAP_BUTT;
	ex.join = JOIN_BEVEL;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("scanline_aa anti-aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("scanline_aa.ppm");

	// anti-aliased dashed outline
	ex.clear();
	ex.alias = 0;
	ex.Ndash = Ndash;
	ex.dash = dash;
	ex.phase = phase;
	ex.cap = CAP_BUTT;
	ex.join = JOIN_MITER;
	ex.width = 1.0;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("outline dash anti-aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("outline_dash.ppm");

	// anti-aliased dashed outline
	ex.clear();
	ex.alias = 0;
	ex.Ndash = Ndash;
	ex.dash = dash;
	ex.phase = phase;
	ex.cap = CAP_BUTT;
	ex.join = JOIN_BEVEL;
	ex.width = 1.0;
	t1 = clock();
	ex.stroke_path(raw_path);
	t2 = clock();
	printf("scanline dash anti-aliased (sec): %f\n", ((float)(t2-t1))/CLOCKS_PER_SEC);
	if (save_output)
		ex.save("scanline_dash.ppm");

	return 0;
}