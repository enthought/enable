#!/usr/bin/env python
import os, glob

for ext in ('bmp', 'png', 'jpg'):
    for pic in glob.glob('*.'+ext):
        if pic.startswith('doubleprom'):
            continue

        os.unlink(pic)
