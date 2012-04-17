%{
    #include "kiva_compiled_path.h"
%}

// handle kiva::rect declarations

%include "rect.i"


%include "agg_typemaps.i"
%apply (double* point_array, int point_count) {(double* pts, int Npts)};
%apply (double* point_array, int point_count) {(double* start, int Nstart)};
%apply (double* point_array, int point_count) {(double* end, int Nend)};
%apply (double* rect_array, int rect_count) {(double* all_rects, 
                                              int Nrects)};
%apply (double *vertex_x, double* vertex_y) {(double* x, double *y)};



namespace kiva
{
    %rename(CompiledPath) compiled_path;
    
    class compiled_path 
    {
        public:
            compiled_path();
            void remove_all();
            void begin_path();
            void close_path();
            void move_to(double x, double y);
            void line_to(double x, double y);
            void quad_curve_to(double x_ctrl,  double y_ctrl, 
                               double x_to,    double y_to);            
            void curve_to(double x_ctrl1, double y_ctrl1, 
                          double x_ctrl2, double y_ctrl2, 
                          double x_to,    double y_to);
            void arc(double x, double y, double radius, double start_angle,

                     double end_angle, bool cw=false);

            void arc_to(double x1, double y1, double x2, double y2, double radius);


            void add_path(compiled_path& vs);
            void lines(double* pts, int Npts);
            void line_set(double* start, int Nstart, double* end, int Nend);
            void rect(kiva::rect_type &rect);
            void rect(double x, double y, double sx, double sy);
            void rects(double* all_rects, int Nrects);
            void translate_ctm(double x, double y);
            void rotate_ctm(double angle);
            void scale_ctm(double sx, double sy);            
            %rename(concat_ctm_agg) concat_ctm(agg24::trans_affine&);
            void concat_ctm(agg24::trans_affine& m);
            %rename(set_ctm_agg) set_ctm(agg24::trans_affine&);
            void set_ctm(agg24::trans_affine& m);
            %pythoncode
            %{
            def kivaaffine_to_aggaffine(self, ctm):
                return AffineMatrix(ctm[0,0], ctm[0,1], ctm[1,0], ctm[1,1],
                                    ctm[2,0], ctm[2,1])
            def concat_ctm(self, ctm):
                # This is really tortured and may cause performance problems.
                # Unfortunately I don't see a much better way right now.
                if '__class__' in dir(ctm) and ctm.__class__.__name__.count('AffineMatrix'):
                    self.concat_ctm_agg(ctm)
                else:
                    self.concat_ctm_agg(self.kivaaffine_to_aggaffine(ctm))
            def set_ctm(self, ctm):
                if '__class__' in dir(ctm) and ctm.__class__.__name__.count('AffineMatrix'):
                    self.set_ctm_agg(ctm)
                else:
                    self.set_ctm_agg(self.kivaaffine_to_aggaffine(ctm))
            %}
            agg24::trans_affine get_ctm();
            void save_ctm();           
            void restore_ctm();
            
            // methods from agg24::path_storage that are used in testing
            unsigned total_vertices() const;
            %rename(_rewind) rewind(unsigned);
            void rewind(unsigned start=0);
            
            %rename (_vertex) vertex(unsigned, double*, double*);
            unsigned vertex(unsigned idx, double* x, double* y) const;            
            
            %rename (_vertex) vertex(double*, double*);
            unsigned vertex(double* x, double* y);

    };    
}

%pythoncode {
from numpy import array, float64
def _vertices(self):
        """ This is only used for testing.  It allows us to retrieve
            all the vertices in the path at once.  The vertices are
            returned as an Nx4 array of the following format.

	        x0, y0, cmd0, flag0
                x1, y1, cmd0, flag1
                ...
        """
        vertices = []
        self._rewind()
        cmd_flag = 1
        while cmd_flag != 0:
            pt, cmd_flag = self._vertex()
            cmd,flag = _agg.path_cmd(cmd_flag),_agg.path_flags(cmd_flag)
            vertices.append((pt[0],pt[1], cmd, flag))        
        return array(vertices)

CompiledPath._vertices = _vertices    


def get_kiva_ctm(self):
        aff = self.get_ctm()
        return array([[aff[0], aff[1], 0],
                      [aff[2], aff[3], 0],
                      [aff[4], aff[5], 1]], float64)

CompiledPath.get_kiva_ctm = get_kiva_ctm
       
}

%clear (double *x, double *y);
