# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os

from traits.api import (
    Enum, File, Float, HasStrictTraits, Instance, Int, Property, Str, Tuple
)


class BenchResult(HasStrictTraits):
    """ The result of a benchmark run on a single backend
    """
    #: Short status field for checking the outcome of a benchmark
    # Default to "fail"!
    summary = Enum("fail", "skip", "success")

    #: A path to an output file with a format and size
    output = File()
    output_format = Property(Str(), observe="output")
    output_size = Tuple(Int(), Int())

    #: Timing results
    timing = Instance("BenchTiming")

    def _get_output_format(self):
        if self.output:
            return os.path.splitext(self.output)[-1]
        return ""

    def compare_to(self, other):
        return BenchComparison.from_pair(self, baseline=other)


class BenchComparison(HasStrictTraits):
    """ A comparison table entry.
    """
    #: CSS class to use for `td`
    css_class = Enum("valid", "invalid", "skipped")

    #: The content for the `td`
    value = Str()

    @classmethod
    def from_pair(cls, result, baseline=None):
        """ Create an instance from two BenchResult instances.
        """
        if result.summary == "fail":
            return cls(value="\N{HEAVY BALLOT X}", css_class="invalid")

        elif result.summary == "skip":
            return cls(value="\N{HEAVY MINUS SIGN}", css_class="skipped")

        elif result.summary == "success":
            if result.timing is not None:
                # Compare timing to the baseline result
                relvalue = baseline.timing.mean / result.timing.mean
                return cls(value=f"{relvalue:0.2f}", css_class="valid")
            else:
                # No timing, but the result was successful
                return cls(value="\N{HEAVY CHECK MARK}", css_class="valid")

        else:
            raise RuntimeError("Unhandled result `summary`")

        return None


class BenchTiming(HasStrictTraits):
    """ The timing results of a single benchmark.
    """
    #: How many times the benchmark ran
    count = Int(0)

    #: avg/min/max/std
    mean = Float(0.0)
    minimum = Float(0.0)
    maximum = Float(0.0)
    stddev = Float(0.0)

    def to_html(self):
        """ Format this instance as an HTML <table>
        """
        names = ("mean", "minimum", "maximum", "stddev", "count")
        rows = [
            (f"<tr><td>{name.capitalize()}</td>"
             f"<td>{getattr(self, name):0.4f}</td></tr>")
            for name in names
        ]

        rows = "\n".join(rows)
        return f'<table>{rows}</table>'
