# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from cProfile import Profile

from traits.api import Any, Dict, Event, HasTraits, Str


class ProfileThis(HasTraits):
    """ Control profiling of different parts of the code.
    """

    # Mapping of str name to Profile instance.
    profilers = Dict()

    # The active Profile instance.
    active_profile = Any()
    active_profile_name = Str()

    # An event with the profiler just ending.
    profile_ended = Event()

    def start(self, name):
        """ Start a particular profile.
        """
        if self.active_profile is None:
            if name not in self.profilers:
                self.profilers[name] = Profile()
            self.active_profile = self.profilers[name]
            self.active_profile_name = name
            self.active_profile.clear()
            self.active_profile.enable()

    def stop(self):
        """ Stop the running profile.
        """
        if self.active_profile is not None:
            p = self.active_profile
            name = self.active_profile_name
            p.disable()
            self.active_profile = None
            self.active_profile_name = ""
            self.profile_ended = (name, p)
