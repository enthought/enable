====================
Contributors's Guide
====================


Expectations
------------

* Code must have associated tests and good documentation.
* GUI code is hard to test. Have a look at ``enable.testing`` and existing
  tests for hints about what is possible in the tests.


Overall Style Guide
-------------------

Enable itself is largely a Python project, and new code is generally expected
to follow PEP8. For kiva, things are a bit more complicated. Cython code can
also follow PEP8 (for the most part). C++/C/ObjC code should aim to follow the
style of the surrounding code.


Python Style Guide
-------------------

Enable uses PEP8 for all Python code. If you need guidance, we recommend using
the flake8 tool, which in addition to PEP8 compliance will also find common
programming errors.

**NOTE**: Try to avoid making formatting changes to parts of the code where you
are _not_ currently making changes. This adds noise to the diff and can
distract the developer who reviews the code.


Branches and Releases
---------------------

Mainline Enable development occurs on the master branch. Features and bug fixes
should be developed in their own topical branches. Once a branch is ready to be
merged into the master branch, a pull request should be created on GitHub so
that the code can be reviewed. Once another developer has signed off on the
changes, they should be **squash merged** into the master branch. This allows
features and bug fixes to be easily cherry-picked into later patch release
branches if needed.

Given that disruptive changes should only be happening in topical branches, the
master branch should always be in a buildable/runnable condition. Continuous
Integration is your friend here, not your nemesis.

Releases will always be tagged. If a patch release is later needed, and major
changes have occured on the master branch (not just bug fixes), then a branch
should be created from the release tag and fix commits cherry-picked from master
until the appropriately patched result is arrived at. That result should then be
tagged as a normal release and the branch deleted.

**NOTE**: The release tag is generally added to a commit _within_ a release
topical branch. For this reason, release branches should be merged normally.
