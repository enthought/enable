// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include "agg_trans_affine.h"
#include "kiva_graphics_context.h"
#include "kiva_compiled_path.h"
#include "kiva_font_type.h"
#include "kiva_rect.h"

#include <iostream>

// ripped from Agg
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

agg24::rgba black(0.0, 0.0, 0.0);
agg24::rgba white(1.0, 1.0, 1.0);
agg24::rgba lightgray(0.2, 0.2, 0.2);
agg24::rgba red(1.0, 0.0, 0.0);
agg24::rgba green(0.0, 1.0, 0.0);
agg24::rgba blue(0.0, 0.0, 1.0);
agg24::rgba niceblue(0.411, 0.584, 0.843);


typedef  agg24::pixfmt_rgb24                     AGG_PIX_TYPE;
typedef  kiva::graphics_context<AGG_PIX_TYPE>  GC_TYPE;
typedef  std::vector<kiva::rect_type>          rect_list_type;




void draw_sub_image(GC_TYPE &gc_img, unsigned int width, unsigned int height)
{
    gc_img.clear(white);
    gc_img.set_alpha(0.4);
    gc_img.set_fill_color(green);
    gc_img.rect(0, 0, width, height);
    gc_img.fill_path();
    
    gc_img.set_alpha(1.0);
	gc_img.set_stroke_color(red);
    gc_img.move_to(0.0,0.0);
	gc_img.line_to(width,height);
    gc_img.stroke_path();
    gc_img.set_stroke_color(blue);
    gc_img.move_to(0.0, height);
    gc_img.line_to(width, 0.0);
	gc_img.stroke_path();

    //show_clip_rects(gc_img);

}


void test_arc_to2(GC_TYPE &gc, double x2, double y2, double radiusstep=25.0)
{
    gc.set_stroke_color(lightgray);
    //gc.move_to(100, 0);
    //gc.line_to(0, 0);
    gc.move_to(0, 0);
    gc.line_to(100, 0);
    gc.line_to(x2, y2);
    gc.stroke_path();
    gc.set_stroke_color(black);

    int numradii = 7;
    for (int i=0; i<numradii; i++)
    {
        //gc.move_to(100, 0);
        //gc.arc_to(0, 0, x2, y2, i*radiusstep+20.0);
        gc.move_to(0, 0);
        gc.arc_to(100, 0, x2, y2, i*radiusstep+20.0);
    }
    gc.stroke_path();
}


void test_arc_curve(GC_TYPE &gc)
{
    gc.path.save_ctm();
    gc.path.translate_ctm(50.0, 50.0);
    gc.path.rotate_ctm(3.14159/8);
    gc.set_stroke_color(blue);
    gc.rect(0.5, 0.5, 210, 210);
    gc.stroke_path();
    gc.set_stroke_color(black);
    gc.set_line_width(1);
    gc.move_to(50.5,25.5);
    gc.arc(50.5, 50.5, 50.0, 0.0, 3.14/2, false);
    gc.move_to(100.5, 50.5);
    gc.arc(100.5, 50.5, 50.0, 0.0, -3.14/2*0.8, false);
    gc.stroke_path();
    //gc.move_to(150.5, 50.5);
    //gc.arc(150.5, 50.5, 50.0, 0.0, 3.14/2*0.6);
    //gc.line_to(120.5, 100.5);
    //gc.move_to(50.5, 50.5);
    //gc.rect(50.0, 50.0, 50.0, 50.0);
    //gc.close_path();

    gc.path.restore_ctm();
    gc.path.save_ctm();
    gc.path.translate_ctm(250.5, 50.5);
//    gc.path.rotate_ctm(3.14159/8);
//    gc.path.scale_ctm(1.0, 0.5);
    gc.set_stroke_color(blue);
    gc.rect(0.5, 0.5, 250.0, 250.0);
    gc.stroke_path();
    gc.set_stroke_color(red);
    gc.move_to(100.0, 100.0);
    gc.line_to(100.0, 150.0);
//    gc.arc_to(150.0, 200.0, 3.14159/2);
    gc.arc_to(100.0, 200.0, 150.0, 200.0, 50.0);
    gc.line_to(200.0, 200.0);
    gc.close_path();
    gc.stroke_path();
    gc.path.restore_ctm();

}


void test_arc_to(GC_TYPE &gc)
{
    kiva::compiled_path axes;
    axes.move_to(0.5, 50.5);
    axes.line_to(100.5, 50.5);
    axes.move_to(50.5, 0.5);
    axes.line_to(50.5, 100.5);
    
    kiva::compiled_path box;
    box.move_to(0.5, 0.5);
    box.line_to(100.5, 0.5);
    box.line_to(100.5, 100.5);
    box.line_to(0.5, 100.5);
    box.close_polygon();

    kiva::compiled_path arc;
    arc.move_to(10, 10);
    arc.line_to(20, 10);
    arc.arc_to(40, 10, 40, 30, 20.0);
    arc.line_to(40, 40);

    kiva::compiled_path whole_shebang;
    whole_shebang.save_ctm();
    whole_shebang.add_path(axes);
    whole_shebang.add_path(box);
    whole_shebang.translate_ctm(0.0, 50.5);
    whole_shebang.add_path(arc);

    whole_shebang.translate_ctm(50.5, 50.5);
    whole_shebang.rotate_ctm(-agg24::pi/2);
    whole_shebang.add_path(arc);
    whole_shebang.rotate_ctm(agg24::pi/2);

    whole_shebang.translate_ctm(50.5, -50.5);
    whole_shebang.rotate_ctm(-agg24::pi);
    whole_shebang.add_path(arc);
    whole_shebang.rotate_ctm(agg24::pi);

    whole_shebang.translate_ctm(-50.5, -50.5);
    whole_shebang.rotate_ctm(-3*agg24::pi/2);
    whole_shebang.add_path(arc);
    whole_shebang.restore_ctm();

    gc.set_stroke_color(red);
    gc.set_line_width(1.0);

    gc.path.save_ctm();
    gc.translate_ctm(50.5, 300.5);
    gc.add_path(whole_shebang);
    gc.stroke_path();

    gc.translate_ctm(130.5, 50.0);
    gc.path.save_ctm();
    gc.rotate_ctm(-agg24::pi/6);
    gc.add_path(whole_shebang);
    gc.set_stroke_color(blue);
    gc.stroke_path();
    gc.path.restore_ctm();

    gc.translate_ctm(130.5, 0.0);
    gc.path.save_ctm();
    gc.rotate_ctm(-agg24::pi/3);
    gc.scale_ctm(1.0, 2.0);
    gc.add_path(whole_shebang);
    gc.stroke_path();
    gc.path.restore_ctm();
    gc.path.restore_ctm();

    gc.path.save_ctm();
    gc.translate_ctm(150.5, 20.5);
    test_arc_to2(gc, 70.5, 96.5);
    gc.translate_ctm(300.5, 0);
    test_arc_to2(gc, 160.5, 76.5, 50.0);
    gc.path.restore_ctm();

    gc.path.translate_ctm(120.5, 100.5);
    gc.scale_ctm(-1.0, 1.0);
    test_arc_to2(gc, 70.5, 96.5);
    gc.path.translate_ctm(-300.5, 100.5);
    gc.scale_ctm(0.75, -1.0);
    test_arc_to2(gc, 160.5, 76.5, 50.0);

}

void test_simple_clip_stack(GC_TYPE &gc)
{
    gc.clear(white);
    //gc.clip_to_rect(40.0, 40.0, gc.width(), gc.height());
    //gc.clip_to_rect(0.0, 0.0, 60.0, 60.0);
    gc.clip_to_rect(100.0, 100.0, 1.0, 1.0);
    gc.rect(0.0, 0.0, gc.width(), gc.height());
    gc.set_fill_color(red);
    gc.fill_path();
}

void test_clip_stack(GC_TYPE &gc)
{
    // General idea:
    //    1. set a multi-clipped window; draw an image;
    //    2. add another clipping rect; draw image;
    //    3. restore the clip state; draw image.
    //
    // Assume the GC is 640x480
    double sub_windows[] = { 10.5, 250, 200, 200,
                             220.5, 250, 200, 200,
                             430.5, 250, 200, 200,
                             10.5, 10, 200, 200,
                             220.5, 10, 200, 200,
                             430.5, 10, 200, 200 };
    // draw the sub-image frames
    gc.set_line_width(2);
    gc.set_stroke_color(black);
    gc.rects(sub_windows, 6);
    gc.stroke_path();

    unsigned char imgbuf[200*200*4];
    GC_TYPE img(imgbuf, 200, 200, -200*4);

    kiva::rect_list_type main_rects;
    main_rects.push_back(kiva::rect_type(40.5, 30.5, 120, 50));
    main_rects.push_back(kiva::rect_type(40.5, 120.5, 120, 50));
    kiva::rect_list_type disjoint_rects;
    disjoint_rects.push_back(kiva::rect_type(60.5, 115.5, 80, 15));
    disjoint_rects.push_back(kiva::rect_type(60.5, 70.5, 80, 15));
    kiva::rect_type vert_rect(60.5, 10.5, 55, 180);

    // draw the full image
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[0]);
    img.clear();

    // first clip
    img.clip_to_rects(main_rects);
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[4]);
    img.save_state();

    // second clip
    img.clear();
    img.clip_to_rects(main_rects);
    img.clip_to_rect(vert_rect);
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[8]);

    // pop back to first clip
    img.restore_state();
    img.clear();
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[12]);

    // adding a disjoint set of rects
    img.clear();
    img.save_state();
    img.clip_to_rects(main_rects);
    img.clip_to_rects(disjoint_rects);
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[16]);

    // pop back to first clip
    img.restore_state();
    img.clear();
    draw_sub_image(img, 200, 200);
    gc.draw_image((kiva::graphics_context_base*)(&img), &sub_windows[20]);

}

void show_clip_rects(GC_TYPE &gc)
{
    kiva::rect_list_type cliprects(gc.state.device_space_clip_rects);
    gc.save_state();
    gc.set_line_width(1);
    gc.set_stroke_color(niceblue);
    gc.rects(cliprects);
    gc.stroke_path();
    gc.restore_state();
}

void test_disjoint_union(GC_TYPE &gc)
{
    gc.save_state();
    gc.set_alpha(0.3);
    rect_list_type mclip_rects;
    rect_list_type actual_rects;
    actual_rects.push_back(kiva::rect_type(40.5,120.5,60,40));
    actual_rects.push_back(kiva::rect_type(80.5,100.5,60,40));
    actual_rects.push_back(kiva::rect_type(80.5,40.5,80,30));
    actual_rects.push_back(kiva::rect_type(100.5,20.5,30,70));
    actual_rects.push_back(kiva::rect_type(20.5,60.5,80,90));
    mclip_rects = kiva::disjoint_union(actual_rects);
//    for (kiva::rect_iterator it=mclip_rects.begin(); it != mclip_rects.end(); it++)
//    {
//        std::cout << it->x << " " << it->y << " " << it->w << " " << it->h << std::endl;
//    }
    //gc.clip_to_rects(mclip_rects);
    //kiva::test_disjoint_union();
    //draw_sub_image(gc, 200, 200);

    double color_delta = 0.0;
    for (kiva::rect_iterator it=actual_rects.begin(); it != actual_rects.end(); it++)
    {
        agg24::rgba tmpcolor(0.0, 0.4+color_delta, 0.0);
        gc.set_fill_color(tmpcolor);
        gc.rect(*it);
        gc.fill_path();
        color_delta += 0.5/actual_rects.size();
    }
    gc.set_alpha(1.0);
    gc.set_stroke_color(black);
    gc.set_line_width(1);
    //gc.clear_clip_path();
    gc.rects(mclip_rects);
    gc.rect(0, 0, 200, 200);
    gc.stroke_path();
    gc.restore_state();
}

void test_disjoint_intersect(GC_TYPE &gc)
{
    kiva::rect_list_type output_rects;
    kiva::rect_list_type input_rects;
    //kiva::rect_type rect1(atof(argv[0]),atof(argv[1]),atof(argv[2]),atof(argv[3]));
    //kiva::rect_type rect2(atof(argv[4]),atof(argv[5]),atof(argv[6]),atof(argv[7]));
    input_rects.push_back(kiva::rect_type(20.5,20.5,60,50));
    //input_rects.push_back(kiva::rect_type(40.5,40.5,60,10));
    input_rects.push_back(kiva::rect_type(60.5,80.5,35,60));
//    input_rects.push_back(kiva::rect_type(40.5,60.5,60,60));
    kiva::rect_type new_rect(40.5,60.5,60,60);
    output_rects = kiva::disjoint_intersect(input_rects, new_rect);

    gc.save_state();
    gc.set_alpha(1.0);
    gc.set_stroke_color(blue);
    gc.rects(input_rects);
    gc.rect(new_rect);
    gc.stroke_path();
    gc.set_alpha(0.4);
    gc.set_fill_color(red);
    gc.rects(output_rects);
    gc.fill_path();
    gc.restore_state();
}

void test_compiled_path(GC_TYPE &gc)
{
    // Compiled path test
    kiva::compiled_path mypath;
    mypath.begin_path();
    mypath.move_to(0.0, 0.0);
    mypath.line_to(20.0, 0.0);
    mypath.line_to(20.0,20.0);
    mypath.line_to(0.0, 20.0);
    mypath.line_to(0.0, 10.0);
    mypath.close_path();
    agg24::rgba tmpcolor(0.0, 0.0, 1.0);
    gc.set_stroke_color(tmpcolor);
    gc.add_path(mypath);
    gc.stroke_path();
    double points[] = {25, 25, 75, 25, 25, 75, 75, 75, 50, 50};
    gc.draw_path_at_points(points, 5, mypath, kiva::STROKE);

}

void test_handling_text(GC_TYPE &gc)
{
    char unicodeString[] = {230, 234, 223, 220, 0};

    // Text handling test.  Make sure timesi.ttf is in the current directory!
    kiva::font_type timesFont("times", 12, "regular");
    kiva::font_type copperFont("coprgtl", 12, "regular");
    kiva::font_type curlzFont("arial", 12, "regular");
    //kiva::font_type unicodeFont("uni_font", 12, "regular");

    gc.set_alpha(1.0);
    gc.set_text_drawing_mode(kiva::TEXT_FILL);

    gc.set_font(timesFont);
    //gc.set_font(unicodeFont);
    gc.translate_ctm(100.0, 100.0);
    gc.move_to(-5,0);
    gc.line_to(5,0);
    gc.move_to(0,5);
    gc.line_to(0,-5);
    gc.move_to(0,0);
    gc.stroke_path();
    //agg24::trans_affine_translation txtTrans(200.0,150.0);
    agg24::trans_affine_rotation txtRot(agg24::deg2rad(30));
    //txtTrans.premultiply(txtRot);
    //txtTrans *= txtRot;
    gc.set_text_matrix(txtRot);
    //gc.set_text_position(150.5, 350.5);
    //kiva::rect_type bbox(gc.get_text_extent("Hello"));
    //gc.get_text_matrix().transform(&bbox.x, &bbox.y);
    gc.show_text("Hello");
    gc.show_text("foobar");
    gc.show_text(unicodeString);
    //txtRot.invert();
    //gc.set_text_matrix(txtRot);
    //gc.show_text("inverted");
//    gc.rect(bbox);
//    gc.stroke_path();
/*
    gc.set_font(copperFont);
    gc.set_text_position(150.5, 250.5);
    kiva::rect_type bbox2(gc.get_text_extent("Hello"));
    gc.get_text_matrix().transform(&bbox2.x, &bbox2.y);
    gc.show_text("Hello");
//    gc.rect(bbox2);
//    gc.stroke_path();

    gc.set_font(curlzFont);
    gc.set_text_position(150.5, 150.5);
    kiva::rect_type bbox3(gc.get_text_extent("Hello"));
    gc.get_text_matrix().transform(&bbox3.x, &bbox3.y);
    gc.show_text("Hello");
//    gc.rect(bbox3);
//    gc.stroke_path();
*/
    //gc.set_stroke_color(red);
    //gc.show_text("blah");

    //gc.set_text_position(200.0, 100.0);
    //gc.show_text("Hello");
    //gc.show_text_translate("Hello", 10.0, 360.0);
    //gc.set_font_size(36);
    //gc.show_text_translate("Hello", 10.0, 190.0);
}


void gc_stress_test()
{
    int width = 100;
    int height = 100;
    int pixelsize = 3;
    unsigned char *buf = new unsigned char[width * height * pixelsize];

    bool tmp = true;
    for (int i=0; i<100000; i++)
    {
        if (i%1000 == 0)
        {
            printf("%d\n", i);
        }
        GC_TYPE gc(buf, width, height, -width*pixelsize);
        kiva::font_type f("arial");
        gc.set_font(f);
//        tmp = gc.show_text("hello");
        if (f.filename == "")
        {
            int q = 5;
        }

    }

    delete[] buf;
}

void brandon_draw_test(GC_TYPE &gc)
{
    gc.set_stroke_color(red);
    gc.set_line_width(1.0);
    gc.begin_path();
    gc.move_to(30.0, 10.0);
    gc.line_to(20.0, 10.0);
    gc.line_to(20.0, 90.0);
    gc.line_to(10.0, 10.0);
    gc.close_path();
    gc.stroke_path();
}

int main(int argc, char **argv)
{

	unsigned int width = 640;
	unsigned int height = 480;
    unsigned char pixelsize = 0;
    AGG_PIX_TYPE *msvc6_dummy = NULL;
    switch (kiva::agg_pix_to_kiva(msvc6_dummy))
    {
        case (kiva::pix_format_gray8) : pixelsize = 1; break;
        case (kiva::pix_format_rgb24) :
        case (kiva::pix_format_bgr24) : pixelsize = 3; break;
        case (kiva::pix_format_bgra32):
        case (kiva::pix_format_rgba32):
        case (kiva::pix_format_argb32):
        case (kiva::pix_format_abgr32): pixelsize = 4; break;
    }
    unsigned char *buf = new unsigned char[width * height * pixelsize];
    
	GC_TYPE gc((unsigned char*)buf, width, height, -width * pixelsize);
    gc.clear();

    //brandon_draw_test(gc);
    //gc_stress_test();
    //test_handling_text(gc);
    test_clip_stack(gc);
    //test_arc_curve(gc);
    //test_arc_to(gc);
    //test_clip_stack(gc);

    if (!write_ppm(buf, width, height, "dummy.ppm"))
	{
		printf("\nError writing file.\n");
	}

    delete[] buf;

    return 0;
}
