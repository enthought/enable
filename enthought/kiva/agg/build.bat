rmdir /s /q build\temp.win32-2.4\Release
rmdir /s /q build\temp.win32-2.4\src
del build\temp.win32-2.4\*.a
del *.pyd
del agg.py
del agg_wrap.cpp
python setup.py build_src build_clib --compiler=mingw32 build_ext --inplace --compiler=mingw32
