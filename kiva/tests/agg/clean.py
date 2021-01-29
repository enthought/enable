#!/usr/bin/env python
import glob
import os

for ext in ("bmp", "png", "jpg"):
    for pic in glob.glob("*." + ext):
        if pic.startswith("doubleprom"):
            continue

        os.unlink(pic)
