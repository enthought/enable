HELP = """
Build script for Kiva

This script performs some common build tasks related to Kiva.  It does
not replace setup.py, but augments it for the casual user/developer.

Usage:
    buildkiva.py [command] <args>

Commands:
    clean   - removes all build products, including in-place extensions

    inplace - builds kiva in-place, without installing into site-packages
              or modifying easy-install.pth.  In order to import it, the
              current directory (or the parent of enthought/kiva/) must be
              in the PYTHONPATH

    develop - builds kiva in-place but modifies easy-install.pth; equivalent
              to building inplace and then running "setup.py develop"

Optional Arguments:
    For the inplace and develop commands, please specify "-c mingw32"
    if you want to use the MingW compiler instead of the MSVC compiler.
    You can also pass other arguments, like "-d /install/path/" to the
    develop command.

"""

import os, shutil, sys

# The full list of files that are potentially produced by an in-place build.
# This is relative to the enthought/kiva/ directory.
INPLACE_FILES = (
    # Mac
    "mac/ABCGI.so",
    "mac/macport.so",
    "mac/ABCGI.c",
    "mac/ATSFont.so",
    "mac/ATSFont.c",
    
    # Common AGG
    "agg/agg.py",
    "agg/plat_support.py",
    "agg/agg_wrap.cpp",
    
    # Win32 Agg
    "agg/_agg.pyd",
    "agg/_plat_support.pyd",
    "agg/src/win32/plat_support_wrap.cpp"
    
    # *nix Agg
    "agg/_agg.so",
    "agg/_plat_support.so",
    "agg/src/x11/plat_support_wrap.cpp",
    
    # Misc
    "agg/src/gl/plat_support_wrap.cpp",
    "agg/src/gl/plat_support.py",
    )

def main():
    args = sys.argv[1:]
    if (len(args) == 0) or (args[0] in ("-h", "--help", "help", "-H")):
        print HELP
        sys.exit(0)
    
    mingw = False
    develop_args = []
    if len(args) > 1:
        from getopt import getopt
        opts, pargs = getopt(args[1:], "c:d:")
        for opt, val in opts:
            if opt == "-c":
                if val == "mingw" or val == "mingw32":
                    mingw = True
            elif opt == "-d":
                develop_args.extend([opt, val])

    cmd = args[0]
    if cmd == "clean":
        do_clean()
    elif cmd == "inplace":
        do_inplace(mingw)
    elif cmd == "develop":
        do_develop(mingw, develop_args)
    else:
        print '\nUnknown command "%s"' % cmd
        print HELP

def do_clean():
    if os.path.isdir("build"):
        shutil.rmtree("build", ignore_errors=True)
    if os.path.isdir("dist"):
        shutil.rmtree("dist", ignore_errors=True)
    
    for f in INPLACE_FILES:
        f = os.path.join("enthought", "kiva", f)
        if os.path.isfile(f):
            os.remove(f)

def do_inplace(mingw, additional_cmds=None):
    cmd = ["build_src", "--inplace", "build_clib"]
    if mingw:
        cmd.append("-c mingw32")

    cmd += ["build_ext", "--inplace"]
    if mingw:
        cmd.append("-c mingw32")
    
    if additional_cmds:
        cmd.extend(additional_cmds)
    
    cmd = " ".join(cmd)
    os.system("python setup.py " + cmd)

def do_develop(mingw, devel_args):
    do_inplace(mingw, additional_cmds = ["develop"] + devel_args)


if __name__ == "__main__":
    main()
