from contextlib import contextmanager
import locale
import unittest

from kiva import agg
from kiva.fonttools import Font


@contextmanager
def locale_context(category, new=None):
    """ Temporarily set the locale.
    """
    old = locale.getlocale(category)
    try:
        locale.setlocale(category, new)
    except locale.Error as e:
        raise unittest.SkipTest(str(e))
    try:
        yield
    finally:
        locale.setlocale(category, old)


class TestText(unittest.TestCase):
    def test_locale_independence(self):
        # Ensure that >ASCII Unicode text is decoded correctly regardless of
        # the locale.
        text = u'\N{GREEK SMALL LETTER MU}'

        with locale_context(locale.LC_CTYPE, ('en', 'UTF-8')):
            gc = agg.GraphicsContextArray((200, 200))
            f = Font('modern')
            with gc:
                gc.set_font(f)
                gc.translate_ctm(50, 50)
                tx0, _, _, _ = gc.get_text_extent(text)
                gc.show_text(text)
                x0, _ = gc.get_text_position()

        with locale_context(locale.LC_CTYPE, ('en', 'ASCII')):
            gc = agg.GraphicsContextArray((200, 200))
            f = Font('modern')
            with gc:
                gc.set_font(f)
                gc.translate_ctm(50, 50)
                tx1, _, _, _ = gc.get_text_extent(text)
                gc.show_text(text)
                x1, _ = gc.get_text_position()

        self.assertEqual(tx1, tx0)
        self.assertEqual(x1, x0)
