// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include <stdio.h>

#include "kiva_gl_font_type.h"

// In the python layer, the enthought.freetype library is used for font lookup.
// Since we can't use that, we emulate the functionality here.
#ifdef _WIN32
    const char* font_dirs[] = {
        "c:/windows/fonts/",
        "./",
        "c:/winnt/fonts/",
        "c:/windows/system32/fonts/",
    };
#elif defined SUNOS
    const char* font_dirs[] = {
        "/usr/openwin/lib/X11/fonts",
    };
#elif defined DARWIN
    const char* font_dirs[] = {
        "/Library/Fonts/",
    };
#else
    const char* font_dirs[] = {
        "./",
        "/usr/lib/X11/fonts/",
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/msttcorefonts/",
        "/var/lib/defoma/x-ttcidfont-conf.d/dirs/TrueType/",
        "/usr/share/fonts/truetype/msttcorefonts/", 
};
#endif

const char* freetype_suffixes[] = { ".ttf", ".pfa", ".pfb" };

// This really only for testing purposes.  Font searching is superceded by the code borrowed from
// matplotlib, however, since that is in python, we can't load a font from C++ for C++ tests.
// Therefore this simple function is left in.
kiva_gl::font_type::font_type(std::string _name, int _size, int _family,
                              int _style, int _encoding, bool validate)
: name(_name)
, size(_size)
, family(_family)
, style(_style)
, encoding(_encoding)
, _is_loaded(false)
{
    std::string full_file_name;
    if (validate)
    {
        if (this->name == "")
        {
            this->_is_loaded = false;
        }
        else
        {
            for (unsigned int d=0; d < sizeof(font_dirs) / sizeof(char*); d++)
            {
                for (unsigned int e=0; e < sizeof(freetype_suffixes) / sizeof(char*); e++)
                {
                    full_file_name = font_dirs[d];
                    full_file_name.append(this->name);
                    full_file_name.append(freetype_suffixes[e]);
                    FILE *f = fopen(full_file_name.c_str(), "rb");
                    if (f != NULL)
                    {
                        fclose(f);
                        this->filename = full_file_name;
                        this->_is_loaded = true;
                        break;
                    }
                }
            }
        }
        this->filename = "";
        this->name = "";
        this->_is_loaded = false;
    }
    else
    {
        this->filename = this->name;
        this->_is_loaded = true;
    }
}

kiva_gl::font_type::font_type(const kiva_gl::font_type &font)
: name(font.name)
, filename(font.filename)
, size(font.size)
, _is_loaded(font.is_loaded())
{
    this->family = font.family;
    this->style = font.style;
}

kiva_gl::font_type&
kiva_gl::font_type::operator=(const kiva_gl::font_type& font)
{
    this->size = font.size;
    this->family = font.family;
    this->style = font.style;
    this->encoding = font.encoding;
    this->name = font.name;
    this->filename = font.filename;
    this->_is_loaded = font.is_loaded();
    return *this;
}

int
kiva_gl::font_type::change_filename(std::string _filename)
{
    FILE *f = fopen(_filename.c_str(), "rb");
    if (f != NULL)
    {
        fclose(f);
        this->filename = _filename;
        this->_is_loaded = true;
        return 1;
    }

    return 0;
}
