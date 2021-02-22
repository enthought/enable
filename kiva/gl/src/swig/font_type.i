// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
%{
    #include "kiva_gl_font_type.h"
%}

%include "agg_std_string.i"

namespace kiva_gl
{
    %rename(KivaGLFontType) font_type;
    class font_type
    {
        public:
            %mutable;
            int size;
            std::string name;
            int family;
            int style;
            int encoding;
            std::string filename;

            // constructor
            font_type(std::string _name="Arial",
                      int _size=12,
                      int _family=0,
                      int _style=0,
                      int _encoding=0,
                      bool validate=true);

            int change_filename(std::string _filename);

            bool is_loaded();
    };
}
%extend kiva_gl::font_type
{
    char *__repr__()
    {
        static char tmp[1024];
        // Write out elements of font_type in name, family, size, style, encoding order
        // !! We should work to make output formatting conform to
        // !! whatever it Numeric does (which needs to be cleaned up also).
        sprintf(tmp, "Font(%s, %d, %d, %d, %d)", self->name.c_str(), self->family,
                                                 self->size, self->style,
                                                 self->encoding);
        return tmp;
    }
    int __eq__(kiva_gl::font_type& other)
    {
        return (self->name == other.name &&
                self->family == other.family &&
                self->size == other.size &&
                self->style == other.style &&
                self->encoding == other.encoding);
    }
}

%pythoncode
%{
def unicode_safe_init(self, _name="Arial", _size=12, _family=0, _style=0,
                      _encoding=0, validate=True):
    ### HACK:  C++ stuff expects a string (not unicode) for the face_name, so fix
    ###        if needed.
    ### Only for python < 3
    if '' == b'':
        if isinstance(_name, unicode):
            _name = _name.encode("latin1")
    else:
        if isinstance(_name, bytes):
            _name = _name.decode()
    obj = _gl.new_KivaGLFontType(_name, _size, _family, _style,
                               _encoding, validate)
    _swig_setattr(self, KivaGLFontType, "this", obj)
    _swig_setattr(self, KivaGLFontType, "thisown", 1)

# This is a crappy way of overriding the constructor
KivaGLFontType.__init__ = unicode_safe_init
%}

