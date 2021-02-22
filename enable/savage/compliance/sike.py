#!/usr/bin/env python
# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from collections import defaultdict
import os
import pstats

from traits.api import (
    Any, Bool, Constant, Dict, Event, Float, HasTraits, Instance, Int, List,
    Property, Str, on_trait_change
)
from traitsui.api import (
    CodeEditor, Group, HGroup, Item, Label, TabularAdapter, TabularEditor,
    UItem, VGroup, View,
)


class SuperTuple(tuple):
    """ Generic super-tuple using pre-defined attribute names.
    """

    __names__ = []

    def __new__(cls, *args, **kwds):
        self = tuple.__new__(cls, *args, **kwds)
        for i, attr in enumerate(cls.__names__):
            setattr(self, attr, self[i])
        return self


class Subrecord(SuperTuple):
    """ The records referring to the calls a function makes.
    """

    __names__ = [
        "file_line_name",
        "ncalls",
        "nonrec_calls",
        "inline_time",
        "cum_time",
    ]

    @property
    def file(self):
        return self[0][0]

    @property
    def line(self):
        return self[0][1]

    @property
    def func_name(self):
        return self[0][2]


class Record(Subrecord):
    """ The top-level profiling record of a function.
    """

    __names__ = [
        "file_line_name",
        "ncalls",
        "nonrec_calls",
        "inline_time",
        "cum_time",
        "callers",
    ]


profile_columns = [
    ("# Calls", "ncalls"),
    ("# Nonrec", "nonrec_calls"),
    ("Self Time", "inline_time"),
    ("Cum. Time", "cum_time"),
    ("Name", "func_name"),
    ("Line", "line"),
    ("File", "file"),
]


class ProfileAdapter(TabularAdapter):
    """ Display profiling records in a TabularEditor.
    """

    columns = profile_columns

    # Whether filenames should only be displayed as basenames or not.
    basenames = Bool(True, update=True)

    # Whether times should be shown as percentages or not.
    percentages = Bool(True, update=True)

    # The total time to use for calculating percentages.
    total_time = Float(1.0, update=True)

    ncalls_width = Constant(55)
    nonrec_calls_width = Constant(55)
    inline_time_width = Constant(75)
    cum_time_width = Constant(75)
    func_name_width = Constant(200)
    line_width = Constant(50)

    ncalls_alignment = Constant("right")
    nonrec_calls_alignment = Constant("right")
    inline_time_alignment = Constant("right")
    cum_time_alignment = Constant("right")
    line_alignment = Constant("right")

    file_text = Property(Str)
    inline_time_text = Property(Str)
    cum_time_text = Property(Str)

    def _get_file_text(self):
        fn = self.item.file_line_name[0]
        if self.basenames and fn != "~":
            fn = os.path.basename(fn)
        return fn

    def _get_inline_time_text(self):
        if self.percentages:
            return "%2.3f" % (self.item.inline_time * 100.0 / self.total_time)
        else:
            return str(self.item.inline_time)

    def _get_cum_time_text(self):
        if self.percentages:
            return "%2.3f" % (self.item.cum_time * 100.0 / self.total_time)
        else:
            return str(self.item.cum_time)


def get_profile_editor(adapter):
    return TabularEditor(
        adapter=adapter,
        editable=False,
        operations=[],
        selected="selected_record",
        column_clicked="column_clicked",
        dclicked="dclicked",
    )


class ProfileResults(HasTraits):
    """ Display profiling results.
    """

    # The sorted list of Records that mirrors this dictionary.
    records = List()
    selected_record = Any()
    dclicked = Event()
    column_clicked = Event()

    # The total time in seconds for the set of records.
    total_time = Float(1.0)

    # The column name to sort on.
    sort_key = Str("inline_time")
    sort_ascending = Bool(False)

    adapter = Instance(ProfileAdapter)
    basenames = Bool(True)
    percentages = Bool(True)

    def trait_view(self, name=None, view_element=None):
        if name or view_element is not None:
            return super(ProfileResults, self).trait_view(
                name=name, view_element=view_element
            )

        view = View(
            Group(Item("total_time", style="readonly")),
            Item(
                "records",
                editor=get_profile_editor(self.adapter),
                show_label=False,
            ),
            width=1024,
            height=768,
            resizable=True,
        )
        return view

    def sorter(self, record):
        """ Return the appropriate sort key for sorting the records.
        """
        return getattr(record, self.sort_key)

    def sort_records(self, records):
        """ Resort the records according to the current settings.
        """
        records = sorted(records, key=self.sorter)
        if not self.sort_ascending:
            records = records[::-1]
        return records

    def _adapter_default(self):
        return ProfileAdapter(
            basenames=self.basenames,
            percentages=self.percentages,
            total_time=self.total_time,
        )

    @on_trait_change("total_time,percentages,basenames")
    def _adapter_traits_changed(self, object, name, old, new):
        setattr(self.adapter, name, new)

    @on_trait_change("sort_key,sort_ascending")
    def _resort(self):
        self.records = self.sort_records(self.records)

    def _column_clicked_changed(self, new):
        if new is None:
            return
        if isinstance(new.column, int):
            key = profile_columns[new.column][1]
        else:
            key = new.column
        if key == self.sort_key:
            # Just flip the order.
            self.sort_ascending = not self.sort_ascending
        else:
            self.trait_set(sort_ascending=False, sort_key=key)


class SillyStatsWrapper(object):
    """ Wrap any object with a .stats attribute or a .stats dictionary such
    that it can be passed to a Stats() constructor.
    """

    def __init__(self, obj=None):
        if obj is None:
            self.stats = {}
        elif isinstance(obj, dict):
            self.stats = obj
        elif isinstance(obj, str):
            # Load from a file.
            self.stats = pstats.Stats(obj)
        elif hasattr(obj, "stats"):
            self.stats = obj.stats
        elif hasattr(obj, "create_stats"):
            obj.create_stats()
            self.stats = obj.stats
        else:
            raise TypeError("don't know how to fake a Stats with %r" % (obj,))

    def create_stats(self):
        pass

    @classmethod
    def getstats(cls, obj=None):
        self = cls(obj)
        return pstats.Stats(self)


class Sike(HasTraits):
    """ Tie several profile-related widgets together.

    Sike is like Gotcha, only less mature.
    """

    # The main pstats.Stats() object providing the data.
    stats = Any()

    # The main results and the subcalls.
    main_results = Instance(ProfileResults, args=())
    caller_results = Instance(ProfileResults, args=())
    callee_results = Instance(ProfileResults, args=())

    # The records have list of callers. Invert this to give a map from function
    # to callee.
    callee_map = Dict()

    # Map from the (file, lineno, name) tuple to the record.
    record_map = Dict()

    # GUI traits ############################################################

    basenames = Bool(True)
    percentages = Bool(True)
    filename = Str()
    line = Int(1)
    code = Str()

    traits_view = View(
        VGroup(
            HGroup(Item("basenames"), Item("percentages")),
            HGroup(
                UItem("main_results"),
                VGroup(
                    Label("Callees"),
                    UItem("callee_results"),
                    Label("Callers"),
                    UItem("caller_results"),
                    UItem("filename", style="readonly"),
                    UItem("code", editor=CodeEditor(line="line")),
                ),
                style="custom",
            ),
        ),
        width=1024,
        height=768,
        resizable=True,
        title="Profiling results",
    )

    @classmethod
    def fromstats(cls, stats, **traits):
        """ Instantiate an Sike from a Stats object, Stats.stats dictionary, or
        Profile object, or a filename of the saved Stats data.
        """
        stats = SillyStatsWrapper.getstats(stats)

        self = cls(stats=stats, **traits)
        self._refresh_stats()
        return self

    def add_stats(self, stats):
        """ Add new statistics.
        """
        stats = SillyStatsWrapper.getstats(stats)
        self.stats.add(stats)
        self._refresh_stats()

    def records_from_stats(self, stats):
        """ Create a list of records from a stats dictionary.
        """
        records = []
        for (
            file_line_name,
            (ncalls, nonrec_calls, inline_time, cum_time, calls),
        ) in stats.items():
            newcalls = []
            for sub_file_line_name, sub_call in calls.items():
                newcalls.append(Subrecord((sub_file_line_name,) + sub_call))
            records.append(
                Record(
                    (
                        file_line_name,
                        ncalls,
                        nonrec_calls,
                        inline_time,
                        cum_time,
                        newcalls,
                    )
                )
            )
        return records

    def get_callee_map(self, records):
        """ Create a callee map.
        """
        callees = defaultdict(list)
        for record in records:
            for caller in record.callers:
                callees[caller.file_line_name].append(
                    Subrecord((record.file_line_name,) + caller[1:])
                )
        return callees

    @on_trait_change("percentages,basenames")
    def _adapter_traits_changed(self, object, name, old, new):
        for obj in [
            self.main_results,
            self.callee_results,
            self.caller_results,
        ]:
            setattr(obj, name, new)

    @on_trait_change("main_results:selected_record")
    def update_sub_results(self, new):
        if new is None:
            return
        self.caller_results.total_time = new.cum_time
        self.caller_results.records = new.callers
        self.callee_results._resort()
        self.caller_results.selected_record = (
            self.caller_results.activated_record
        ) = None

        self.callee_results.total_time = new.cum_time
        self.callee_results.records = self.callee_map.get(
            new.file_line_name, []
        )
        self.callee_results._resort()
        self.callee_results.selected_record = (
            self.callee_results.activated_record
        ) = None

        filename, line, name = new.file_line_name
        if os.path.exists(filename):
            with open(filename, "ru") as f:
                code = f.read()
            self.code = code
            self.filename = filename
            self.line = line
        else:
            self.trait_set(code="", filename="", line=1)

    @on_trait_change("caller_results:dclicked," "callee_results:dclicked")
    def goto_record(self, new):
        if new is None:
            return
        if new.item.file_line_name in self.record_map:
            record = self.record_map[new.item.file_line_name]
            self.main_results.selected_record = record

    @on_trait_change("stats")
    def _refresh_stats(self):
        """ Refresh the records from the stored Stats object.
        """
        self.main_results.records = self.main_results.sort_records(
            self.records_from_stats(self.stats.stats)
        )
        self.callee_map = self.get_callee_map(self.main_results.records)
        self.record_map = {}
        total_time = 0.0
        for record in self.main_results.records:
            self.record_map[record.file_line_name] = record
            total_time += record.inline_time
        self.main_results.total_time = total_time


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file")

    args = parser.parse_args()
    stats = pstats.Stats(args.file)
    app = Sike.fromstats(stats)

    app.configure_traits()


if __name__ == "__main__":
    main()
