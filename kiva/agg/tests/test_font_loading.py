# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import time

from kiva.image import font_metrics_provider as FMP
from kiva.api import Font

counts = (500,)
strings = ("hello",)  # ascii_lowercase + ascii_uppercase)
fonts = (("arial", 12),)  # ("times", 16), ("courier", 10) )


def test():

    allmetrics = []
    for count in counts:
        start = time.time()
        for i in range(count):
            metrics = FMP()
            for face, size in fonts:
                metrics.set_font(Font(face, size))
                for s in strings:
                    metrics.get_text_extent(s)
            allmetrics.append(metrics)
        end = time.time()
        print("finished count=%d" % count)
        print("   total time:", end - start)
        print("   time/set_font:", (end - start) / float(count * len(fonts)))


if __name__ == "__main__":
    test()
