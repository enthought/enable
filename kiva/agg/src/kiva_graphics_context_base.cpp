// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifdef _WIN32 // Win32 threads
    #include <windows.h>
    static CRITICAL_SECTION gCriticalSection;

#else        // POSIX threads
#endif

#include <assert.h>

#include "utf8.h"

#include "agg_path_storage.h"
#include "kiva_exceptions.h"
#include "kiva_graphics_context_base.h"

using namespace kiva;


#ifdef KIVA_USE_FREETYPE
static font_engine_type gFontEngine;
#endif
#ifdef KIVA_USE_WIN32
static font_engine_type gFontEngine(hdc);
#endif
static font_manager_type gFontManager(gFontEngine);

font_engine_type* kiva::GlobalFontEngine()
{
    return &gFontEngine;
}

font_manager_type* kiva::GlobalFontManager()
{
    return &gFontManager;
}

void kiva::cleanup_font_threading_primitives()
{
#ifdef _WIN32
    DeleteCriticalSection(&gCriticalSection);
#else

#endif
}


// Create a static pool of font engines that get recycled.
//static FontEngineCache font_engine_cache = FontEngineCache();

graphics_context_base::graphics_context_base(unsigned char *data, int width,
                       int height, int stride, interpolation_e interp):
                       buf(),
                       _image_interpolation(interp)
{
   this->buf.attach(data, width, height, stride);
}

graphics_context_base::~graphics_context_base()
{
}


int graphics_context_base::width()
{
    return this->buf.width();
}


int graphics_context_base::height()
{
    return this->buf.height();
}


int graphics_context_base::stride()
{
    return this->buf.stride();
}


int graphics_context_base::bottom_up()
{
    return (this->stride() > 0 ? 0 : 1);
}


agg24::rendering_buffer& graphics_context_base::rendering_buffer()
{
    return this->buf;
}



kiva::interpolation_e graphics_context_base::get_image_interpolation()
{
    return this->_image_interpolation;
}


void graphics_context_base::set_image_interpolation(kiva::interpolation_e interpolation)
{
    this->_image_interpolation = interpolation;
}


//---------------------------------------------------------------
// set graphics_state values
//---------------------------------------------------------------


void graphics_context_base::set_stroke_color(agg24::rgba& value)
{
    this->state.line_color = value;
}


agg24::rgba& graphics_context_base::get_stroke_color()
{
    return this->state.line_color;
}


void graphics_context_base::set_line_width(double value)
{
    this->state.line_width = value;
}


void graphics_context_base::set_line_join(kiva::line_join_e value)
{
    this->state.line_join = value;
}


void graphics_context_base::set_line_cap(kiva::line_cap_e value)
{
    this->state.line_cap = value;
}

void graphics_context_base::set_line_dash(double* pattern, int n, double phase)
{
    this->state.line_dash = kiva::dash_type(phase, pattern, n);
}

void graphics_context_base::set_blend_mode(kiva::blend_mode_e value)
{
    this->state.blend_mode = value;
}

kiva::blend_mode_e graphics_context_base::get_blend_mode()
{
    return this->state.blend_mode;
}

void graphics_context_base::set_fill_color(agg24::rgba& value)
{
    this->state.fill_color = value;
}

agg24::rgba& graphics_context_base::get_fill_color()
{
    return this->state.fill_color;
}

void graphics_context_base::set_alpha(double value)
{
    // alpha should be between 0 and 1, so clamp:
    if (value < 0.0)
    {
        value = 0.0;
    }
    else if (value > 1.0)
    {
        value = 1.0;
    }
    this->state.alpha = value;
}


double graphics_context_base::get_alpha()
{
    return this->state.alpha;
}

void graphics_context_base::set_antialias(int value)
{
    this->state.should_antialias = value;
}


int graphics_context_base::get_antialias()
{
    return this->state.should_antialias;
}


void graphics_context_base::set_miter_limit(double value)
{
    this->state.miter_limit = value;
}


void graphics_context_base::set_flatness(double value)
{
    this->state.flatness = value;
}


//---------------------------------------------------------------
// text and font functions
//---------------------------------------------------------------


void graphics_context_base::set_text_position(double tx, double ty)
{
    double temp[6];
    this->text_matrix.store_to(temp);
    temp[4] = tx;
    temp[5] = ty;
    this->text_matrix.load_from(temp);
}


void graphics_context_base::get_text_position(double* tx, double* ty)
{
    double temp[6];
    agg24::trans_affine result = this->get_text_matrix();
    result.store_to(temp);
    *tx = temp[4];
    *ty = temp[5];
}

bool graphics_context_base::is_font_initialized()
{
    // This method is left in here just for legacy reasons.  Although
    // technically the font is *never* initialized now that all GCs
    // are sharing a single font_cache_manager, external users of the
    // class should be able to proceed as if the font were initialized.
    return true;
}

void graphics_context_base::set_text_matrix(agg24::trans_affine& value)
{
    this->text_matrix = value;
}


agg24::trans_affine graphics_context_base::get_text_matrix()
{
    return this->text_matrix;
}


void graphics_context_base::set_character_spacing(double value)
{
    this->state.character_spacing = value;
}


double graphics_context_base::get_character_spacing()
{
    return this->state.character_spacing;
}


void graphics_context_base::set_text_drawing_mode(kiva::text_draw_mode_e value)
{
    this->state.text_drawing_mode = value;
}




//---------------------------------------------------------------
// save/restore graphics state
//---------------------------------------------------------------


void graphics_context_base::save_state()
{
    this->state_stack.push(this->state);
    this->path.save_ctm();
}

//---------------------------------------------------------------
// coordinate transform matrix transforms
//---------------------------------------------------------------


void graphics_context_base::translate_ctm(double x, double y)
{
    this->path.translate_ctm(x, y);
}


void graphics_context_base::rotate_ctm(double angle)
{
    this->path.rotate_ctm(angle);
}


void graphics_context_base::scale_ctm(double sx, double sy)
{
    this->path.scale_ctm(sx, sy);
}


void graphics_context_base::concat_ctm(agg24::trans_affine& m)
{
    this->path.concat_ctm(m);
}


void graphics_context_base::set_ctm(agg24::trans_affine& m)
{
    this->path.set_ctm(m);
}


agg24::trans_affine graphics_context_base::get_ctm()
{
    return this->path.get_ctm();
}


void graphics_context_base::get_freetype_text_matrix(double* out)
{
    agg24::trans_affine result =  this->get_ctm();
    result.multiply(this->get_text_matrix());
    result.store_to(out);
    // freetype and agg transpose their matrix conventions
    double temp = out[1];
    out[1] = out[2];
    out[2] = temp;
}
//---------------------------------------------------------------
// Sending drawing data to a device
//---------------------------------------------------------------

void graphics_context_base::flush()
{
    // TODO-PZW: clarify this and other "not sure if anything is needed" functions
    // not sure if anything is needed.
}


void graphics_context_base::synchronize()
{
    // not sure if anything is needed.
}

//---------------------------------------------------------------
// Page Definitions
//---------------------------------------------------------------


void graphics_context_base::begin_page()
{
    // not sure if anything is needed.
}


void graphics_context_base::end_page()
{
    // not sure if anything is needed.
}

//---------------------------------------------------------------
// Path operations
//---------------------------------------------------------------


void graphics_context_base::begin_path()
{
    this->path.begin_path();
}


void graphics_context_base::move_to(double x, double y)
{
    this->path.move_to(x, y);
}


void graphics_context_base::line_to( double x, double y)
{
    this->path.line_to(x, y);
}


void graphics_context_base::curve_to(double cpx1, double cpy1,
              double cpx2, double cpy2,
              double x, double y)
{
    this->path.curve_to(cpx1, cpy1, cpx2, cpy2, x, y);
}


void graphics_context_base::quad_curve_to(double cpx, double cpy,
                   double x, double y)
{
    this->path.quad_curve_to(cpx, cpy, x, y);
}

void graphics_context_base::arc(double x, double y, double radius,
                                double start_angle, double end_angle,
                                bool cw)
{
    this->path.arc(x, y, radius, start_angle, end_angle, cw);
}

void graphics_context_base::arc_to(double x1, double y1, double x2, double y2,
                                   double radius)
{
    this->path.arc_to(x1, y1, x2, y2, radius);
}

void graphics_context_base::close_path()
{
    this->path.close_polygon();
}


void graphics_context_base::add_path(kiva::compiled_path& other_path)
{
    this->path.add_path(other_path);
}


void graphics_context_base::lines(double* pts, int Npts)
{
    this->path.lines(pts, Npts);
}

void graphics_context_base::line_set(double* start, int Nstart, double* end, int Nend)
{
    this->path.line_set(start, Nstart, end, Nend);
}

void graphics_context_base::rect(double x, double y, double sx, double sy)
{
    this->path.rect(x, y, sx, sy);
}

void graphics_context_base::rect(kiva::rect_type &rect)
{
    this->path.rect(rect);
}


void graphics_context_base::rects(double* all_rects, int Nrects)
{
    this->path.rects(all_rects,Nrects);
}

void graphics_context_base::rects(kiva::rect_list_type &rectlist)
{
    this->path.rects(rectlist);
}

kiva::compiled_path graphics_context_base::_get_path()
{
    return this->path;
}

kiva::rect_type graphics_context_base::_get_path_bounds()
{
    double xmin = 0., ymin = 0., xmax = 0., ymax = 0.;
    double x = 0., y = 0.;
    
    for (unsigned i = 0; i < this->path.total_vertices(); ++i)
    {
        this->path.vertex(i, &x, &y);
        
        if (i == 0)
        {
            xmin = xmax = x;
            ymin = ymax = y;
            continue;
        }
        
        if (x < xmin) xmin = x;
        else if (xmax < x) xmax = x;
        if (y < ymin) ymin = y;
        else if (ymax < y) ymax = y;
    }
    
    return kiva::rect_type(xmin, ymin, xmax-xmin, ymax-ymin);
}

agg24::path_storage graphics_context_base::boundary_path(agg24::trans_affine& affine_mtx)
{
    // Return the path that outlines the image in device space
    // This is used in _draw to specify the device area
    // that should be rendered.
    agg24::path_storage clip_path;
    double p0x = 0;
    double p0y = 0;
    double p1x = this->width();
    double p1y = 0;
    double p2x = this->width();
    double p2y = this->height();
    double p3x = 0;
    double p3y = this->height();

    affine_mtx.transform(&p0x, &p0y);
    affine_mtx.transform(&p1x, &p1y);
    affine_mtx.transform(&p2x, &p2y);
    affine_mtx.transform(&p3x, &p3y);

    clip_path.move_to(p0x, p0y);
    clip_path.line_to(p1x, p1y);
    clip_path.line_to(p2x, p2y);
    clip_path.line_to(p3x, p3y);
    clip_path.close_polygon();
    return clip_path;
}

/////////////////////////////////////////////////////////////////////////////
// Text methods
/////////////////////////////////////////////////////////////////////////////

bool graphics_context_base::set_font(kiva::font_type& font)
{
    // See if the font is the same; if it is, then do nothing:
    if (font == this->state.font)
    {
        return true;
    }

    this->state.font = font;

    // short-circuit: if the font didn't even load properly, then this
    // call can't succeed.
    if (!this->state.font.is_loaded())
    {
        return false;
    }
    else
    {
        return true;
    }
}


kiva::font_type& graphics_context_base::get_font()
{
    return this->state.font;
}


bool graphics_context_base::set_font_size(int size)
{
    // just make sure the font is loaded; don't check is_font_initialized
    if (!this->state.font.is_loaded())
    {
        return false;
    }
    else
    {
        this->state.font.size = size;
        return true;
    }
}


bool graphics_context_base::show_text_at_point(char *text,
                                               double tx, double ty)
{
    double oldx, oldy;

    this->get_text_position(&oldx, &oldy);
    this->set_text_position(tx, ty);
    bool retval = this->show_text(text);
    this->set_text_position(oldx, oldy);
    return retval;
}

kiva::rect_type graphics_context_base::get_text_extent(char *text)
{
    const agg24::glyph_cache *glyph = NULL;

    // Explicitly decode UTF8 bytes to 32-bit codepoints to feed into the
    // font API.
    size_t text_length = strlen(text);
    utf8::iterator<char*> p(text, text, text+text_length);
    utf8::iterator<char*> p_end(text+text_length, text, text+text_length);

    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2= 0.0;

    static font_manager_type *font_manager = GlobalFontManager();

    if (font_manager == NULL)
        return kiva::rect_type(0, 0, 0, 0);

    this->_grab_font_manager();

    //typedef agg24::glyph_raster_bin<agg24::rgba8> GlyphGeneratorType;
    //GlyphGeneratorType glyphGen(this->font_manager.glyph(*p)->data);

    for (; p!=p_end; ++p)
    {
        glyph = font_manager->glyph(*p);
        if (glyph == NULL)
        {
            continue;
        }
        font_manager->add_kerning(&x2, &y2);
        x1 = kiva::min(x1, glyph->bounds.x1);
        x2 += glyph->advance_x;
        y1 = kiva::min(y1, glyph->bounds.y1);
        y2 = kiva::max(y2, glyph->bounds.y2);
    }

    this->_release_font_manager();

    return kiva::rect_type(x1, y1, x2-x1, y2 - y1);
}


bool graphics_context_base::get_text_bbox_as_rect(char *text)
{
    return false;
}

int graphics_context_base::draw_image(kiva::graphics_context_base* img)
{
    double tmp[] = {0, 0, img->width(), img->height()};
    return this->draw_image(img, tmp);
}

void graphics_context_base::_grab_font_manager()
{
// Win32 threads
#ifdef _WIN32
    static bool critical_section_initialized = false;

    if (!critical_section_initialized)
    {
        // FIXME: We need to delete the CriticalSection object when the process
        // exits, but where should we put that code?
        InitializeCriticalSection(&gCriticalSection);
        critical_section_initialized = true;
    }

    EnterCriticalSection(&gCriticalSection);

// POSIX threads
#else

#endif  // _WIN32

    font_engine_type *font_engine = GlobalFontEngine();
    if (font_engine == NULL)
        return;

    font_type *font = &this->state.font;

#ifdef KIVA_USE_FREETYPE
    if (font->filename != "")
    {
        font_engine->load_font(font->filename.c_str(), font->face_index,
                               agg24::glyph_ren_agg_gray8);
    }
    else
    {
        font_engine->load_font(font->name.c_str(), font->face_index,
                               agg24::glyph_ren_agg_gray8);
    }
#endif

#ifdef KIVA_USE_WIN32
    font_engine->create_font(font->name,
                             agg24::glyph_ren_native_gray8,
                             font->size);
#endif

    font_engine->hinting(1);
    font_engine->resolution(72);

//    The following is a more laborious but more "correct" way of determining
//    the correct font size to use under Win32.  Unfortunately, it doesn't
//    work exactly right either; characters come out just a few pixels larger.
//    Thus, for the time being, we're going to punt and leave the hard-coded
//    adjustment factor.
//
//    this->font_engine.height(12.0);
//    this->font_engine.width(12.0);
//    kiva::rect_type tmp(this->get_text_extent("X"));
//    this->font_engine.height(font.size*12.0/tmp.h);
//    this->font_engine.width(font.size*12.0/tmp.w);
    font_engine->height(font->size);
    font_engine->width(font->size);

}

void graphics_context_base::_release_font_manager()
{

// Win32 thread-safe implementations of GlobalFontEngine and GlobalFontManager
#ifdef _WIN32
    LeaveCriticalSection(&gCriticalSection);

// POSIX thread-safe implementations of GlobalFontEngine and GlobalFontManager
#else

#endif  // _WIN32
}

//---------------------------------------------------------------------
// Gradient support
//---------------------------------------------------------------------
void graphics_context_base::linear_gradient(double x1, double y1,
                    double x2, double y2,
                    std::vector<kiva::gradient_stop> stops,
                    const char* spread_method,
                    const char* units)
{
    // not implemented
    throw kiva::not_implemented_error;
}

void graphics_context_base::radial_gradient(double cx, double cy, double r,
                    double fx, double fy,
                    std::vector<kiva::gradient_stop> stops,
                    const char* spread_method,
                    const char* units)
{
    // not implemented
    throw kiva::not_implemented_error;
}

