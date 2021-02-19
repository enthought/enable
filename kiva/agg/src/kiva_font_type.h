// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef FONT_TYPE_H
#define FONT_TYPE_H

#include <string>

#ifdef _MSC_VER
// Turn off MSDEV warning about truncated long identifiers
#pragma warning(disable:4786)
#endif

namespace kiva
{
    class font_type
    {
        public:
            std::string name;
            std::string filename;
            int size;
            int family;
            int style;
            int encoding;
            int face_index;

            // Constructors
            
            // Creates a font object.  By default, searches the hardcoded
            // font paths for a file named like the face name; to override
            // this, set validate=false.
            font_type(std::string _name="Arial",
                      int _size=12,
                      int _family=0,
                      int _style=0,
                      int _encoding=0,
                      int _face_index=0,
                      bool validate=true);

            font_type(const font_type &font);
            font_type &operator=(const font_type& font);
        
            int change_filename(std::string _filename);
            
            // Is the font loaded properly?
            inline bool is_loaded() const { return _is_loaded; }
        
        private:
            bool _is_loaded;
    };

    inline bool operator==(font_type &a, font_type &b)
    {
        return (a.size == b.size) && (a.name == b.name) && 
               (a.style == b.style) && (a.encoding == b.encoding) &&
               (a.family == b.family) && (a.face_index == b.face_index);
    }
    
    
}
#endif
