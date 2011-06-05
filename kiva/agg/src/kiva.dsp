# Microsoft Developer Studio Project File - Name="kiva" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=kiva - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "kiva.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "kiva.mak" CFG="kiva - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "kiva - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "kiva - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "kiva - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /W3 /GX /O2 /I "..\agg2\include" /I "..\agg2\font_freetype" /I "..\freetype2\include" /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 ..\freetype2\objs\ kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "kiva - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /G6 /MD /W3 /Gm /GX /ZI /Od /I "..\agg2\include" /I "..\agg2\font_win32_tt" /I "..\agg2\font_freetype" /I "..\..\..\freetype\freetype2\include" /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /GZ /Zm1000 /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ..\build\temp.win32-2.3\freetype2_src.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "kiva - Win32 Release"
# Name "kiva - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\kiva_affine_helpers.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_affine_helpers.h
# End Source File
# Begin Source File

SOURCE=.\kiva_affine_matrix.h
# End Source File
# Begin Source File

SOURCE=.\kiva_basics.h
# End Source File
# Begin Source File

SOURCE=.\kiva_compiled_path.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_compiled_path.h
# End Source File
# Begin Source File

SOURCE=.\kiva_constants.h
# End Source File
# Begin Source File

SOURCE=.\kiva_dash_type.h
# End Source File
# Begin Source File

SOURCE=.\kiva_font_type.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_font_type.h
# End Source File
# Begin Source File

SOURCE=.\kiva_graphics_context.h
# End Source File
# Begin Source File

SOURCE=.\kiva_graphics_context_base.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_graphics_context_base.h
# End Source File
# Begin Source File

SOURCE=.\kiva_graphics_state.h
# End Source File
# Begin Source File

SOURCE=.\kiva_hit_test.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_hit_test.h
# End Source File
# Begin Source File

SOURCE=.\kiva_image_filters.h
# End Source File
# Begin Source File

SOURCE=.\kiva_pix_format.h
# End Source File
# Begin Source File

SOURCE=.\kiva_rect.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_rect.h
# End Source File
# Begin Source File

SOURCE=.\kiva_span_conv_alpha.cpp
# End Source File
# Begin Source File

SOURCE=.\kiva_span_conv_alpha.h
# End Source File
# Begin Source File

SOURCE=.\kiva_text_image.h
# End Source File
# End Group
# Begin Group "SWIG wrappers"

# PROP Default_Filter ".i"
# Begin Source File

SOURCE=.\affine_matrix.i
# End Source File
# Begin Source File

SOURCE=.\agg_std_string.i
# End Source File
# Begin Source File

SOURCE=.\agg_typemaps.i
# End Source File
# Begin Source File

SOURCE=..\build\src\agg_wrap.cpp
# PROP Exclude_From_Build 1
# End Source File
# Begin Source File

SOURCE=.\compiled_path.i
# End Source File
# Begin Source File

SOURCE=.\constants.i
# End Source File
# Begin Source File

SOURCE=.\font_type.i
# End Source File
# Begin Source File

SOURCE=.\graphics_context.i
# End Source File
# Begin Source File

SOURCE=.\numeric.i
# End Source File
# Begin Source File

SOURCE=.\numeric_ext.i
# End Source File
# Begin Source File

SOURCE=.\rect.i
# End Source File
# Begin Source File

SOURCE=.\rgba.i
# End Source File
# Begin Source File

SOURCE=.\rgba_array.i
# End Source File
# Begin Source File

SOURCE=.\sequence_to_array.i
# End Source File
# End Group
# Begin Group "Python files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\chaco_ex.py
# End Source File
# Begin Source File

SOURCE=.\code_tst.py
# End Source File
# Begin Source File

SOURCE=.\examples.py
# End Source File
# Begin Source File

SOURCE=.\setup_swig_agg.py
# End Source File
# Begin Source File

SOURCE=.\tst_convert.py
# End Source File
# Begin Source File

SOURCE=.\win32_tst.py
# End Source File
# End Group
# Begin Group "Test files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\dummy.cpp
# End Source File
# End Group
# Begin Group "agg sources"

# PROP Default_Filter ""
# Begin Source File

SOURCE=..\agg2\src\agg_arc.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_arrowhead.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_bezier_arc.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_bspline.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_curves.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_embedded_raster_fonts.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_gsv_text.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_image_filters.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_line_aa_basics.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_line_profile_aa.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_path_storage.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_rasterizer_scanline_aa.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_rounded_rect.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_sqrt_tables.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_trans_affine.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_trans_double_path.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_trans_single_path.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_trans_warp_magnifier.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_bspline.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_contour.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_dash.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_markers_term.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_smooth_poly1.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vcgen_stroke.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vpgen_clip_polygon.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vpgen_clip_polyline.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\src\agg_vpgen_segmentator.cpp
# End Source File
# End Group
# Begin Group "agg freetype"

# PROP Default_Filter ""
# Begin Source File

SOURCE=..\agg2\font_freetype\agg_font_freetype.cpp
# End Source File
# Begin Source File

SOURCE=..\agg2\font_freetype\agg_font_freetype.h
# End Source File
# End Group
# Begin Group "agg_headers"

# PROP Default_Filter ""
# Begin Source File

SOURCE=..\agg2\include\agg_alpha_mask_u8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_arc.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_array.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_arrowhead.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_basics.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_bezier_arc.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_bitset_iterator.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_bounding_rect.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_bspline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_clip_liang_barsky.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_color_rgba.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_color_rgba8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_adaptor_vcgen.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_adaptor_vpgen.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_bspline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_clip_polygon.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_clip_polyline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_close_polygon.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_concat.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_contour.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_curve.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_dash.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_gpc.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_marker.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_marker_adaptor.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_segmentator.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_shorten_path.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_smooth_poly1.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_stroke.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_transform.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_conv_unclose_polygon.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_curves.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_dda_line.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_ellipse.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_ellipse_bresenham.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_embedded_raster_fonts.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_font_cache_manager.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_gamma_functions.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_gamma_lut.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_glyph_raster_bin.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_gray8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_gsv_text.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_image_filters.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_line_aa_basics.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_math.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_math_stroke.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_path_storage.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_path_storage_integer.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pattern_filters_rgba8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_amask_adaptor.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_crb_rgba32.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_crb_rgba32_pre.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_gray8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgb24.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgb24_gamma.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgb24_pre.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgb555.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgb565.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgba32.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgba32_plain.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_pixfmt_rgba32_pre.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rasterizer_outline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rasterizer_outline_aa.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rasterizer_scanline_aa.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_render_scanlines.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_base.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_markers.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_mclip.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_outline_aa.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_outline_image.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_primitives.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_raster_text.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_renderer_scanline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rendering_buffer.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rendering_buffer_dynarow.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_rounded_rect.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_bin.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_boolean_algebra.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_p.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_storage_aa.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_storage_bin.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_scanline_u.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_shorten_path.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_simul_eq.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_allocator.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_converter.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_generator.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gouraud.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gouraud_gray8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gouraud_rgba8.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gouraud_rgba8_gamma.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gradient.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_gradient_alpha.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_image_filter.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_image_filter_rgb24.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_image_filter_rgb24_gamma.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_image_filter_rgba32.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_interpolator_adaptor.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_interpolator_linear.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_interpolator_trans.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_pattern.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_pattern_filter_rgba32.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_pattern_rgb24.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_pattern_rgba32.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_span_solid.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_affine.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_bilinear.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_double_path.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_perspective.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_single_path.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_viewport.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_trans_warp_magnifier.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_bspline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_contour.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_dash.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_markers_term.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_smooth_poly1.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_stroke.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vcgen_vertex_sequence.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vertex_iterator.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vertex_sequence.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vpgen_clip_polygon.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vpgen_clip_polyline.h
# End Source File
# Begin Source File

SOURCE=..\agg2\include\agg_vpgen_segmentator.h
# End Source File
# End Group
# Begin Source File

SOURCE=.\readme.txt
# End Source File
# Begin Source File

SOURCE=.\swig_questions.txt
# End Source File
# Begin Source File

SOURCE=.\todo.txt
# End Source File
# End Target
# End Project
