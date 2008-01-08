# Microsoft Developer Studio Project File - Name="clip_polyline" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=clip_polyline - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "clip_polyline.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "clip_polyline.mak" CFG="clip_polyline - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "clip_polyline - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "clip_polyline - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "clip_polyline - Win32 Release"

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
# ADD CPP /nologo /G6 /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /profile /machine:I386

!ELSEIF  "$(CFG)" == "clip_polyline - Win32 Debug"

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
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR /YX /FD /GZ /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /profile /debug /machine:I386

!ENDIF 

# Begin Target

# Name "clip_polyline - Win32 Release"
# Name "clip_polyline - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_arc.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_arrowhead.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_bezier_arc.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_bspline.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_curves.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_embedded_raster_fonts.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_gsv_text.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_image_filters.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_line_aa_basics.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_line_profile_aa.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_path_storage.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_rasterizer_scanline_aa.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_rounded_rect.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_sqrt_tables.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_trans_affine.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_trans_double_path.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_trans_single_path.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_trans_warp_magnifier.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_bspline.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_contour.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_dash.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_markers_term.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_smooth_poly1.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vcgen_stroke.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vpgen_clip_polygon.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vpgen_clip_polyline.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\..\agg2\src\agg_vpgen_segmentator.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\clip_polyline.cpp
# End Source File
# Begin Source File

SOURCE=..\..\..\test_conv_curve.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
