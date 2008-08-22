#for the benefit of people not using nose or trial 
#or any other sort of test discovery mechanism.
#there's probably a better way to do this


from svg.tests.test_pathdata import *
from svg.tests.test_document import *
from svg.tests.css import *
from svg.tests.test_attributes import *

if __name__ == '__main__':
    unittest.main(module="svg.tests")
