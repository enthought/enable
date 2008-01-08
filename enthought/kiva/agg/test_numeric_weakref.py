import sys

import weave
from Numeric import array
import distutils.errors

try:
    a = array([0])
    weave.inline("((PyArrayObject*)py_a)->weakreflist;",['a'],compiler='gcc',force=1,verbose=0)
    sys.exit(0)
except distutils.errors.CompileError:
    # We don't return 1 b/c that is the same thing that os.system returns
    # if it couldn't locate/execute this file.
    sys.exit(10)