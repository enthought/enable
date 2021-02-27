# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import argparse
import os
import sys

from pyface.qt import QtGui

from enable.gcbench.bench import benchmark
from enable.gcbench.publish import publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    # Create a QApplication instance so that the QPainter backend can be tested
    app = QtGui.QApplication(sys.argv)  # noqa: F841

    outdir = args.output
    if outdir is not None and not os.path.isdir(outdir):
        os.mkdir(outdir)

    results = benchmark(outdir=outdir)
    if outdir is not None:
        publish(results, outdir)


if __name__ == '__main__':
    main()
