%{
    #include "kiva_font_type.h"
%}

%include "agg_std_string.i"



namespace kiva
{
    %rename(AggFontType) font_type;
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
%extend kiva::font_type
{    
    char *__repr__()
    {
        static char tmp[1024];
        // Write out elements of trans_affine in a,b,c,d,tx,ty order
        // !! We should work to make output formatting conform to 
        // !! whatever it Numeric does (which needs to be cleaned up also).
        sprintf(tmp,"Font(%s,%d,%d,%d,%d)", self->name.c_str(), self->family,
                                         self->size, self->style, 
                                         self->encoding);
        return tmp;
    }
    int __eq__(kiva::font_type& other)
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
    if '' == b'':
        if isinstance(_name, unicode):
            _name = _name.encode("latin1")
    else:
        if isinstance(_name, bytes):
            _name = _name.decode()
    obj = _agg.new_AggFontType(_name, _size, _family, _style,
                               _encoding, validate)
    _swig_setattr(self, AggFontType, "this", obj)
    _swig_setattr(self, AggFontType, "thisown", 1)

# This is a crappy way of overriding the constructor
AggFontType.__init__ = unicode_safe_init
%}
    
