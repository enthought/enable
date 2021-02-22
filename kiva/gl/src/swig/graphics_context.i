// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

// typemaps for many enumerated types used in graphics_contexts
%include "constants.i"

// Language independent exception handler
%include exception.i

// handle kiva_gl::rect declarations
%include "rect.i"
//%apply kiva_gl::rect_type {kiva_gl::rect_type};

%include "agg_typemaps.i"
%apply (double* point_array, int point_count) {(double* pts, int Npts)};
%apply (double* point_array, int point_count) {(double* start, int Nstart)};
%apply (double* point_array, int point_count) {(double* end, int Nend)};
%apply (double* rect_array, int rect_count) {(double* rects, int Nrects)};
%apply (double* pt_x, double* pt_y) {(double* tx, double* ty)};
%apply (double* array6) {(double* out)};
%apply (double* dash_pattern, int n) { (double* pattern, int n)};
%apply (unsigned char *image_data, int width, int height, int stride) {
            (unsigned char *data, int width, int height, int stride) };
%apply (owned_pointer) { kiva_gl::graphics_context* };

// typemaps for double ary[]
%include "sequence_to_array.i"

%include "rgba_array.i"
%apply rgba_as_array {kiva_gl_agg::rgba&};
%{
    kiva_gl_agg::rgba _clear_color = kiva_gl_agg::rgba(1,1,1,1);
%}

%typemap(out) PyObject*
{
    $result = $1;
}

%{
#include "kiva_gl_graphics_context.h"
%}

namespace kiva_gl {

    %rename(GraphicsContextGL) gl_graphics_context;

    class gl_graphics_context : public graphics_context_base
    {
    public:
        gl_graphics_context(int width, int height,
                            kiva_gl::pix_format_e format=kiva_gl::pix_format_rgb24);

        ~gl_graphics_context();

        //---------------------------------------------------------------
        // GL-specific methods
        //---------------------------------------------------------------
        void gl_init();
        void gl_cleanup();
        void gl_render_path(kiva_gl::compiled_path *path, bool polygon=false, bool fill=false);
        void gl_render_points(double** points, bool polygon, bool fill,
                              kiva_gl::draw_mode_e mode = FILL);

        //---------------------------------------------------------------
        // GraphicsContextBase interface
        //---------------------------------------------------------------

        int bottom_up();
        int width();
        int height();
        int stride();

        void save_state();
        void restore_state();

        void flush();
        void synchronize();

        void begin_page();
        void end_page();

        void translate_ctm(double x, double y);
        void rotate_ctm(double angle);
        void scale_ctm(double sx, double sy);

        %feature("shadow") concat_ctm(kiva_gl_agg::trans_affine& m)
        %{
        def concat_ctm(self, m):
            if isinstance(m, tuple):
                _gl.GraphicsContextGL_concat_ctm(self, _AffineMatrix(*m))
            else:
                _gl.GraphicsContextGL_concat_ctm(self, m)
        %}
        void concat_ctm(kiva_gl_agg::trans_affine& m);

        %feature("shadow") set_ctm(kiva_gl_agg::trans_affine& m)
        %{
        def set_ctm(self, m):
            if isinstance(m, tuple):
                _gl.GraphicsContextGL_set_ctm(self, _AffineMatrix(*m))
            else:
                _gl.GraphicsContextGL_set_ctm(self, m)
        %}
        void set_ctm(kiva_gl_agg::trans_affine& m);

        %feature("shadow") get_ctm()
        %{
        def get_ctm(self):
            tmp = _gl.GraphicsContextGL_get_ctm(self)
            return (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        %}
        kiva_gl_agg::trans_affine get_ctm();

        %feature("shadow") format()
        %{
        def format(self):
            enum = _gl.GraphicsContextGL_format(self)
            return pix_format_enum_map[enum]
        %}
        kiva_gl::pix_format_e format();

        %feature("shadow") get_image_interpolation()
        %{
        def get_image_interpolation(self):
            enum = _gl.GraphicsContextGL_get_image_interpolation(self)
            return interp_enum_map[enum]
        %}
        interpolation_e get_image_interpolation();

        %feature("shadow") set_image_interpolation(kiva_gl::interpolation_e interpolation)
        %{
        def set_image_interpolation(self,interp):
            enum = interp_string_map[interp]
            _gl.GraphicsContextGL_set_image_interpolation(self, enum)
        %}
        void set_image_interpolation(kiva_gl::interpolation_e interpolation);

        %feature("shadow") set_stroke_color(kiva_gl_agg::rgba& rgba_in)
        %{
        def set_stroke_color(self, color):
            if is_array(color) and len(color) == 3:
                ary = color
                r, g, b = ary
                color = Rgba(r, g, b)
            elif is_array(color) and len(color) == 4:
                ary = color
                r, g, b, a = ary
                color = Rgba(r, g, b, a)
            _gl.GraphicsContextGL_set_stroke_color(self, color)
        %}
        void set_stroke_color(kiva_gl_agg::rgba& rgba_in);
        kiva_gl_agg::rgba& get_stroke_color();

        %feature("shadow") set_fill_color(kiva_gl_agg::rgba& rgba_in)
        %{
        def set_fill_color(self, color):
            if is_array(color) and len(color) == 3:
                ary = color
                r, g, b = ary
                color = Rgba(r, g, b)
            elif is_array(color) and len(color) == 4:
                ary = color
                r, g, b, a = ary
                color = Rgba(r, g, b, a)
            _gl.GraphicsContextGL_set_fill_color(self, color)
        %}
        void set_fill_color(kiva_gl_agg::rgba& rgba_in);
        kiva_gl_agg::rgba& get_fill_color();

        void set_alpha(double value);
        double get_alpha();
        void set_antialias(int value);
        int get_antialias();
        void set_miter_limit(double value);
        void set_flatness(double value);
        void set_line_width(double value);
        void set_line_join(kiva_gl::line_join_e value);
        void set_line_cap(kiva_gl::line_cap_e value);
        void set_line_dash(double* pattern, int n, double phase=0);
        void set_blend_mode(kiva_gl::blend_mode_e value);
        kiva_gl::blend_mode_e get_blend_mode();

        //---------------------------------------------------------------
        // Path manipulation
        //---------------------------------------------------------------

        void begin_path();
        void move_to(double x, double y);
        void line_to( double x, double y);
        void curve_to(double cpx1, double cpy1,
                      double cpx2, double cpy2,
                      double x, double y);
        void quad_curve_to(double cpx, double cpy, double x, double y);

        void arc(double x, double y, double radius, double start_angle,
                 double end_angle, bool cw=false);
        void arc_to(double x1, double y1, double x2, double y2, double radius);

        void close_path();
        void add_path(kiva_gl::compiled_path& other_path);
        void lines(double* pts, int Npts);
        void line_set(double* start, int Nstart, double* end, int Nend);
        void rect(kiva_gl::rect_type &rect);
        void rect(double x, double y, double sx, double sy);
        void rects(double* all_rects, int Nrects);
        compiled_path _get_path();

        //---------------------------------------------------------------
        // Clipping path manipulation
        //---------------------------------------------------------------

        void clip();
        void even_odd_clip();
        void clip_to_rect(double x, double y, double sx, double sy);
        void clip_to_rect(kiva_gl::rect_type &rect);
        void clip_to_rects(double* new_rects, int Nrects);
        void clip_to_rects(kiva_gl::rect_list_type &rects);
        kiva_gl::rect_type transform_clip_rectangle(const kiva_gl::rect_type &rect);
        void clear_clip_path();

        int get_num_clip_regions();
        kiva_gl::rect_type get_clip_region(unsigned int i);

        //---------------------------------------------------------------
        // Painting paths (drawing and filling contours)
        //---------------------------------------------------------------

        // Declare clear() to pass by reference so that the typemap applies,
        // even though it is pass by value in the actual C++ class
        void clear(kiva_gl_agg::rgba& value=_clear_color);

        void fill_path();
        void eof_fill_path();
        void stroke_path();
        // empty function; for some reason this is abstract in the base class
        inline void _stroke_path() { }

        void draw_path(draw_mode_e mode=FILL_STROKE);
        void draw_rect(double rect[4], draw_mode_e mode=FILL_STROKE);

        %feature("shadow") draw_marker_at_points(double* pts,int Npts, int size,
                                   kiva_gl::marker_e type = kiva_gl::marker_square)
        %{
        def draw_marker_at_points(self, pts, size, kiva_marker_type):
            marker = kiva_marker_to_agg.get(kiva_marker_type, None)
            if marker is None:
                success = 0
            else:
                args = (self, pts, int(size), marker)
                success = _gl.GraphicsContextGL_draw_marker_at_points(
                    self, pts, int(size), marker
                )
            return success
        %}
        int draw_marker_at_points(double* pts,int Npts,int size,
                                   kiva_gl::marker_e type=kiva_gl::marker_square);

        void draw_path_at_points(double* pts,int Npts,
                                  kiva_gl::compiled_path& marker,
                                  draw_mode_e mode);

        //---------------------------------------------------------------
        // Image rendering
        //---------------------------------------------------------------

        int draw_image(kiva_gl::graphics_context_base* img, double rect[4], bool force_copy=false);
        int draw_image(kiva_gl::graphics_context_base* img);

        //---------------------------------------------------------------
        // Text rendering (NOTE: disabled)
        //---------------------------------------------------------------

        %pythoncode
        %{
        def show_text(self, text, point = None):
            """Displays text at point, or at the current text pen position
               if point is None.  Returns true if text displayed properly,
               false if there was a font issue or a glyph could not be
               rendered.  Will handle multi-line text separated by backslash-ns
            """
            raise RuntimeError("Text is not supported by OpenGL.")

        def show_text_at_point(self, text, dx, dy):
            raise RuntimeError("Text is not supported by OpenGL.")

        def get_text_extent(self, text):
            raise RuntimeError("Text is not supported by OpenGL.")

        def get_full_text_extent(self, text):
            raise RuntimeError("Text is not supported by OpenGL.")

        def get_font(self):
            raise RuntimeError("Text is not supported by OpenGL.")

        def set_font(self, font):
            raise RuntimeError("Unable to load font.")

        def is_font_initialized(self):
            raise RuntimeError("Font not loaded/initialized.")

        def set_font_size(self, size):
            raise RuntimeError("Font not loaded/initialized.")

        def get_freetype_text_matrix(self, *args):
            raise RuntimeError("Text is not supported by OpenGL.")

        def get_text_matrix(self, matrix):
            raise RuntimeError("Text is not supported by OpenGL.")

        def set_text_matrix(self, matrix):
            raise RuntimeError("Text is not supported by OpenGL.")

        def set_text_position(self, tx, ty):
            raise RuntimeError("Text is not supported by OpenGL")

        def get_text_position(self):
            raise RuntimeError("Text is not supported by OpenGL")

        def set_character_spacing(self, value):
            raise RuntimeError("Text is not supported by OpenGL.")

        def get_character_spacing(self):
            raise RuntimeError("Text is not supported by OpenGL.")

        def set_text_drawing_mode(self, value):
            raise RuntimeError("Text is not supported by OpenGL.")

        %}

        //---------------------------------------------------------------------
        // Gradient support (raises NotImplementedError)
        //---------------------------------------------------------------------

        %pythoncode
        %{
        def linear_gradient(self, x1, y1, x2, y2, stops, spread_method, units="userSpaceOnUse"):
            raise NotImplementedError("Gradient fills are not supported by OpenGL")

        def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method, units="userSpaceOnUse"):
            raise NotImplementedError("Gradient fills are not supported by OpenGL")

        %}

        //---------------------------------------------------------------
        // Additional methods (added as pure python)
        //---------------------------------------------------------------

        %pythoncode
        %{
        def get_empty_path(self):
            return CompiledPath()

        def convert_pixel_format(self, pix_format, inplace=0):
            """ Convert gc from one pixel format to another.

            NOTE: This has never worked on OpenGL, because draw_image has never
            been implemented. It used to inherit the GraphicsContextArry
            implementation, which relied on the context having a working
            implementation of draw_image which would handle the pixel format
            conversion.
            """
            return self.__class__(self.width(), self.height(), pix_format=pix_format)

        def save(self, filename, file_format=None, pil_options=None):
            """ Save the GraphicsContext to a file.

            NOTE: This has never worked on OpenGL, because it draws this
            context into an image context by using the agg `rendering_buffer`
            as a source of pixel data. The buffer is never used with OpenGL,
            so it is always blank.
            """
            raise RuntimeError("Saving is not supported by OpenGL.")

        #----------------------------------------------------------------
        # context manager interface
        #----------------------------------------------------------------

        def __enter__(self):
            self.save_state()

        def __exit__(self, type, value, traceback):
            self.restore_state()

        %}

    };

}
