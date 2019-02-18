import unittest
try:
    from unittest import mock
except ImportError:
    import mock

from ..font_manager import createFontList


class TestCreateFontList(unittest.TestCase):

    @mock.patch("kiva.fonttools.font_manager.TTCollection")
    def test_fontlist_from_ttc(self, m_TTCollection):
        # Given
        mocked_collection = mock.MagicMock()
        m_TTCollection.return_value = mocked_collection
        mocked_collection.fonts = [mock.MagicMock(), mock.MagicMock()]

        # When
        with mock.patch("kiva.fonttools.font_manager.ttfFontProperty"):
            fontlist = createFontList(['/foo/bar/bazFont.ttc'])

        # Then
        self.assertEqual(len(fontlist), 2)
