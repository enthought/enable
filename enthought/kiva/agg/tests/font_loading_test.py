from string import lowercase, uppercase
import os
import time

from enthought.kiva.image import font_metrics_provider as FMP
from enthought.kiva.fonttools import Font

counts = (500,)
strings = ("hello", ) # lowercase + uppercase)
fonts = ( ("arial", 12), ) # ("times", 16), ("courier", 10) )

def test():

    allmetrics = []
    for count in counts:
        start = time.time()
        for i in range(count):
            metrics = FMP()
            for face, size in fonts:
                metrics.set_font(Font(face, size))
                for s in strings:
                    dims = metrics.get_text_extent(s)
            allmetrics.append(metrics)
        end = time.time()
        print "finished count=%d" % count
        print "   total time:", end - start
        print "   time/set_font:", (end-start) / float(count * len(fonts))

if __name__ == "__main__":
    test()
