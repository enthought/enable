import os
import shutil
import tempfile

from PIL import Image as PILImage

from kiva.image import Image, GraphicsContext
from traits.testing.unittest_tools import unittest

class TestImage(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.filename = os.path.join(self.directory, 'temp.png')
        image = PILImage.new('RGB', (100, 120))
        image.save(self.filename)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_initialization(self):
        image = Image(self.filename)
        self.assertEqual(image.width(), 100)
        self.assertEqual(image.height(), 120)
        self.assertEqual(image.format(), 'rgb24')

if __name__ == "__main__":
    unittest.main()
