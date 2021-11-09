# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Tasks for Test Runs
===================

This file is intended to be used with a python environment with the
click library to automate the process of setting up test environments
and running the test within them.  This improves repeatability and
reliability of tests be removing many of the variables around the
developer's particular Python environment.  Test environment setup and
package management is performed using `EDM
http://docs.enthought.com/edm/`_

To use this to run you tests, you will need to install EDM and click
into your working environment.  You will also need to have git
installed to access required source code from github repositories.
You can then do::

    python edmtool.py install --runtime=... --toolkit=...

to create a test environment from the current codebase and::

    python edmtool.py test --runtime=... --toolkit=...

to run tests in that environment.  You can remove the environment with::

    python edmtool.py cleanup --runtime=... --toolkit=...

If you make changes you will either need to remove and re-install the
environment or manually update the environment using ``edm``, as
the install performs a ``python setup.py install`` rather than a ``develop``,
so changes in your code will not be automatically mirrored in the test
environment.  You can update with a command like::

    edm run --environment ... -- python setup.py install

You can run all three tasks at once with::

    python edmtool.py test_clean --runtime=... --toolkit=...

which will create, install, run tests, and then clean-up the environment.  And
you can run tests in all supported runtimes and toolkits (with cleanup)
using::

    python edmtool.py test_all

Currently supported runtime value is `3.6``, and currently
supported toolkits are ``null``, ``pyqt``, and ``pyqt5``.  Not all
combinations of toolkits and runtimes will work, but the tasks will fail with
a clear error if that is the case.

Tests can still be run via the usual means in other environments if that suits
a developer's purpose.

Changing This File
------------------

To change the packages installed during a test run, change the dependencies
variable below.  To install a package from github, or one which is not yet
available via EDM, add it to the `ci/requirements.txt` file (these will be
installed by `pip`).

Other changes to commands should be a straightforward change to the listed
commands for each task. See the EDM documentation for more information about
how to run commands within an EDM enviornment.

"""

import glob
import os
import subprocess
import sys
from shutil import rmtree, copy as copyfile
from tempfile import mkdtemp
from contextlib import contextmanager

import click

supported_combinations = {
    '3.6': {'pyside2', 'pyqt', 'pyqt5', 'wx', 'null'},
}

dependencies = {
    "apptools",
    "celiagg",
    "coverage",
    "Cython",
    "fonttools",
    "hypothesis",
    "kiwisolver",
    "numpy",
    "pillow",
    "pyface",
    "pygments",
    "pyparsing",
    "pypdf2",
    "reportlab",
    "swig",
    "traits",
    "traitsui",
}

# Dependencies we install from source for cron tests
# Order from packages with the most dependencies to one with the least
# dependencies. Packages are forced re-installed in this order.
source_dependencies = [
    "apptools",
    "traitsui",
    "pyface",
    "traits",
]

extra_dependencies = {
    'pyqt': {'pyqt'},
    'pyqt5': {'pyqt5'},
    'pyside2': {'pyside2'},
    # XXX once wxPython 4 is available in EDM, we will want it here
    "wx": set(),
    'null': set()
}

environment_vars = {
    'pyside2': {'ETS_TOOLKIT': 'qt4', 'QT_API': 'pyside2'},
    'pyqt': {'ETS_TOOLKIT': 'qt4', 'QT_API': 'pyqt'},
    'pyqt5': {'ETS_TOOLKIT': 'qt4', 'QT_API': 'pyqt5'},
    'wx': {'ETS_TOOLKIT': 'wx'},
    'null': {'ETS_TOOLKIT': 'null.image'},
}

#: Path to the top-level source directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


github_url_fmt = "git+http://github.com/enthought/{0}.git#egg={0}"


# Ensure that "-h" is supported for getting help.
CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}

# Config file for EDM
EDM_CONFIG = os.path.join(ROOT, "ci", ".edm.yaml")


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
@click.option(
    "--source/--no-source",
    default=False,
    help="Install ETS packages from source",
)
def install(runtime, toolkit, environment, source):
    """ Install project and dependencies into a clean EDM environment.

    """
    parameters = get_parameters(runtime, toolkit, environment)
    parameters['packages'] = ' '.join(
        dependencies | extra_dependencies.get(toolkit, set()))
    # edm commands to setup the development environment
    commands = [
        ("edm --config {edm_config} environments create {environment} "
         "--force --version={runtime}"),
        ("edm --config {edm_config} install -y -e {environment} {packages} "
         "--add-repository enthought/lgpl"),
        ("edm run -e {environment} -- pip install -r ci/requirements.txt"
         " --no-dependencies"),
    ]

    if toolkit == "wx":
        if sys.platform == "darwin":
            commands.append(
                "edm run -e {environment} -- python -m pip install wxPython<4.1"  # noqa: E501
            )
        elif sys.platform == "linux":
            # XXX this is mainly for TravisCI workers; need a generic solution
            commands.append(
                "edm run -e {environment} -- pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04/ wxPython<4.1"  # noqa: E501
            )
        else:
            commands.append(
                "edm run -e {environment} -- python -m pip install wxPython"
            )

    click.echo("Creating environment '{environment}'".format(**parameters))
    execute(commands, parameters)

    if source:
        # Remove EDM ETS packages and install them from source
        cmd_fmt = (
            "edm plumbing remove-package "
            "--environment {environment} --force "
        )
        commands = [cmd_fmt + source_pkg for source_pkg in source_dependencies]
        execute(commands, parameters)
        source_pkgs = [
            github_url_fmt.format(pkg) for pkg in source_dependencies
        ]
        # Without the --no-dependencies flag such that new dependencies on
        # master are brought in.
        commands = [
            "python -m pip install --force-reinstall {pkg}".format(pkg=pkg)
            for pkg in source_pkgs
        ]
        commands = [
            "edm run -e {environment} -- " + command for command in commands
        ]
        execute(commands, parameters)

    # No matter what happens before, always install local source again with no
    # dependencies or we risk testing against an released enable.
    install_local = (
        "edm run -e {environment} -- "
        "pip install --force-reinstall --no-dependencies " + ROOT
    )
    execute([install_local], parameters)

    click.echo('Done install')


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
def docs(runtime, toolkit, environment):
    """ Build documentation. """
    parameters = get_parameters(runtime, toolkit, environment)
    packages = " ".join([
        "sphinx",
        "enthought_sphinx_theme",
    ])
    enable_ignore = " ".join([
        "enable/gcbench",
        "enable/null",
        "enable/qt4",
        "enable/savage/trait_defs/ui/qt4",
        "enable/savage/trait_defs/ui/wx",
        "enable/trait_defs/ui/qt4",
        "enable/trait_defs/ui/wx",
        "enable/wx",
        "*/tests",
    ])
    kiva_ignore = " ".join([
        "kiva/fonttools/font_manager.py",
        "kiva/basecore2d.py",
        "kiva/cairo.py",
        "kiva/celiagg.py",
        "kiva/oldagg.py",
        "kiva/pdf.py",
        "kiva/pdfmetrics.py",
        "kiva/ps.py",
        "kiva/qpainter.py",
        "kiva/svg.py",
        "kiva/agg",
        "kiva/gl",
        "kiva/quartz",
        "*/tests",
    ])
    commands = [
        "edm install -y -e {environment} " + packages,
    ]
    click.echo(
        "Installing documentation tools in  '{environment}'".format(
            **parameters
        )
    )
    execute(commands, parameters)
    click.echo("Done installing documentation tools")

    click.echo(
        "Regenerating API docs in  '{environment}'".format(**parameters)
    )
    api_path = os.path.join("docs", "source", "api")
    templates_path = os.path.join(api_path, "templates")
    commands = [
        "edm run -e {environment} -- sphinx-apidoc --separate --no-toc -o "
        + api_path
        + " -t " + templates_path
        + " enable "
        + enable_ignore
    ]
    execute(commands, parameters)
    click.echo("Done regenerating enable API docs")

    commands = [
        "edm run -e {environment} -- sphinx-apidoc --separate --no-toc -o "
        + api_path
        + " -t " + templates_path
        + " kiva "
        + kiva_ignore
    ]
    execute(commands, parameters)
    click.echo("Done regenerating kiva API docs")

    try:
        os.chdir("docs")
        command = (
            "edm run -e {environment} -- sphinx-build -b html "
            "-d build/doctrees "
            "source "
            "build/html"
        )
        click.echo(
            "Building documentation in  '{environment}'".format(**parameters)
        )
        execute([command], parameters)
    finally:
        os.chdir("..")
    click.echo("Done building documentation")


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
def shell(runtime, toolkit, environment):
    """ Create a shell into the EDM development environment
    (aka 'activate' it).

    """
    parameters = get_parameters(runtime, toolkit, environment)
    commands = [
        "edm shell -e {environment}",
    ]
    execute(commands, parameters)


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
def test(runtime, toolkit, environment):
    """ Run the test suite in a given environment with the specified toolkit.

    """
    parameters = get_parameters(runtime, toolkit, environment)
    environ = environment_vars.get(toolkit, {}).copy()
    environ["PYTHONUNBUFFERED"] = "1"
    environ["PYTHONFAULTHANDLER"] = "1"
    commands = [
        ("edm run -e {environment} -- python -W default -m"
         "coverage run -m unittest discover enable -v"),
        ("edm run -e {environment} -- python -W default -m"
         "coverage run -a -m unittest discover kiva -v"),
    ]

    # We run in a tempdir to avoid accidentally picking up wrong traitsui
    # code from a local dir.  We need to ensure a good .coveragerc is in
    # that directory, plus coverage has a bug that means a non-local coverage
    # file doesn't get populated correctly.
    click.echo("Running tests in '{environment}'".format(**parameters))
    with do_in_tempdir(files=['.coveragerc'], capture_files=['./.coverage*']):
        os.environ.update(environ)
        execute(commands, parameters)
    click.echo('Done test')


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
def cleanup(runtime, toolkit, environment):
    """ Remove a development environment.

    """
    parameters = get_parameters(runtime, toolkit, environment)
    commands = [
        "edm run -e {environment} -- python setup.py clean",
        "edm environments remove {environment} --purge -y",
    ]
    click.echo("Cleaning up environment '{environment}'".format(**parameters))
    execute(commands, parameters)
    click.echo('Done cleanup')


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
def test_clean(runtime, toolkit):
    """ Run tests in a clean environment, cleaning up afterwards

    """
    args = [
        '--toolkit={}'.format(toolkit),
        '--runtime={}'.format(runtime),
    ]
    try:
        install(args=args, standalone_mode=False)
        test(args=args, standalone_mode=False)
    finally:
        cleanup(args=args, standalone_mode=False)


@cli.command()
@click.option('--runtime', default='3.6')
@click.option('--toolkit', default='null')
@click.option('--environment', default=None)
def update(runtime, toolkit, environment):
    """ Update/Reinstall package into environment.

    """
    parameters = get_parameters(runtime, toolkit, environment)
    commands = [
        "edm run -e {environment} -- python setup.py install"]
    click.echo("Re-installing in  '{environment}'".format(**parameters))
    execute(commands, parameters)
    click.echo('Done update')


@cli.command()
def test_all():
    """ Run test_clean across all supported environment combinations.

    """
    failed_command = False
    for runtime, toolkits in supported_combinations.items():
        for toolkit in toolkits:
            args = ['--toolkit={}'.format(toolkit),
                    '--runtime={}'.format(runtime)]
            try:
                test_clean(args, standalone_mode=True)
            except SystemExit:
                failed_command = True
    if failed_command:
        sys.exit(1)


# ----------------------------------------------------------------------------
# Utility routines
# ----------------------------------------------------------------------------

def get_parameters(runtime, toolkit, environment):
    """Set up parameters dictionary for format() substitution
    """
    parameters = {'runtime': runtime, 'toolkit': toolkit,
                  'environment': environment}
    if toolkit not in supported_combinations[runtime]:
        msg = ("Python {runtime} and toolkit {toolkit} "
               "not supported by test environments")
        raise RuntimeError(msg.format(**parameters))
    if environment is None:
        tmpl = 'enable-test-{runtime}-{toolkit}'
        environment = tmpl.format(**parameters)
        parameters['environment'] = environment

    parameters["edm_config"] = EDM_CONFIG
    return parameters


@contextmanager
def do_in_tempdir(files=(), capture_files=()):
    """ Create a temporary directory, cleaning up after done.

    Creates the temporary directory, and changes into it.  On exit returns to
    original directory and removes temporary dir.

    Parameters
    ----------
    files : sequence of filenames
        Files to be copied across to temporary directory.
    capture_files : sequence of filenames
        Files to be copied back from temporary directory.
    """
    path = mkdtemp()
    old_path = os.getcwd()

    # send across any files we need
    for filepath in files:
        click.echo('copying file to tempdir: {}'.format(filepath))
        copyfile(filepath, path)

    os.chdir(path)
    try:
        yield path
        # retrieve any result files we want
        for pattern in capture_files:
            for filepath in glob.iglob(pattern):
                click.echo('copying file back: {}'.format(filepath))
                copyfile(filepath, old_path)
    finally:
        os.chdir(old_path)
        rmtree(path)


def execute(commands, parameters):
    for command in commands:
        print("[EXECUTING]", command.format(**parameters))
        try:
            subprocess.check_call(command.format(**parameters).split())
        except subprocess.CalledProcessError:
            sys.exit(1)


if __name__ == '__main__':
    cli()
