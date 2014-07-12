
from unittest import TestCase, main

from numpy import pi
from kiva.fonttools import Font
from kiva.constants import MODERN
from kiva.image import GraphicsContext

class DrawingTest(TestCase):
    
    def setUp(self):
        self.gc = GraphicsContext((300, 300))
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))
        
    def test_line(self):
        self.gc.begin_path
        self.gc.move_to(107, 204)
        self.gc.line_to(107, 104)
        self.gc.stroke_path()
        self.gc.save("line.bmp")
    
    def test_rectangle(self):
        self.gc.begin_path()
        self.gc.move_to(107, 104)
        self.gc.line_to(107, 184)
        self.gc.line_to(187, 184)
        self.gc.line_to(187, 104)
        self.gc.line_to(107, 104)
        self.gc.stroke_path()
        self.gc.save("rectangle.bmp")
    
    def test_rect(self):
        self.gc.begin_path()
        self.gc.rect(0,0,200,200)
        self.gc.stroke_path()
        self.gc.save("rect.bmp")
    
    def test_circle(self):
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, 2 * pi)
        self.gc.stroke_path()
        self.gc.save("circle.bmp")
    
    def test_quarter_circle(self):
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, pi / 2)
        self.gc.stroke_path()
        self.gc.save("quarter_circle.bmp")
        
    def test_text(self):
        font = Font(family=MODERN)
        font.size = 24
        self.gc.set_font(font)
        self.gc.set_text_position(23, 67)
        self.gc.show_text("hello kiva")
        self.gc.save("text.bmp")
        
    def test_circle_fill(self):
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, 2 * pi)
        self.gc.fill_path()
        self.gc.save("circle_fill.bmp")
    
    def test_star_fill(self):
        self.gc.begin_path()
        self.gc.move_to(100, 100)
        self.gc.line_to(150, 200)
        self.gc.line_to(200, 100)
        self.gc.line_to(100, 150)
        self.gc.line_to(200, 150)
        self.gc.line_to(100, 100)
        self.gc.fill_path()
        self.gc.save("star.bmp")
    
    def test_star_eof_fill(self):
        self.gc.begin_path()
        self.gc.move_to(100, 100)
        self.gc.line_to(150, 200)
        self.gc.line_to(200, 100)
        self.gc.line_to(100, 150)
        self.gc.line_to(200, 150)
        self.gc.line_to(100, 100)
        self.gc.eof_fill_path()
        self.gc.save("star_eof.bmp")

    def test_circle_clip(self):
        self.gc.clip_to_rect(150, 150, 100, 100)
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, 2 * pi)
        self.gc.fill_path()
        self.gc.save("circle_clip.bmp")
        
    def test_text_clip(self):
        self.gc.clip_to_rect(23, 77, 100, 23)
        font = Font(family=MODERN)
        font.size = 24
        self.gc.set_font(font)
        self.gc.set_text_position(23, 67)
        self.gc.show_text("hello kiva")
        self.gc.save("text_clip.bmp")
       
    def test_star_clip(self):
        self.gc.begin_path()
        self.gc.move_to(100, 100)
        self.gc.line_to(150, 200)
        self.gc.line_to(200, 100)
        self.gc.line_to(100, 150)
        self.gc.line_to(200, 150)
        self.gc.line_to(100, 100)
        self.gc.close_path()
        self.gc.clip()
        
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, 2 * pi)
        self.gc.fill_path()
        self.gc.save("star_clip.bmp")
     
    
        
if __name__ == "__main__":
    main()

