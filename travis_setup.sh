#! /bin/sh
set -e

sudo apt-get update

# Install swig for agg
sudo apt-get install swig

# Simlinks for PIL compilation
sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/
sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/
sudo ln -s /usr/lib/`uname -i`-linux-gnu/libpng.so /usr/lib/
sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/

# setup gui
if [ "$ETS_TOOLKIT" = "wx" ]; then
    sudo apt-get install python-wxtools python-wxgtk2.8-dbg
elif [ "$ETS_TOOLKIT" = "qt4" ]; then
    sudo apt-get install python-qt4 python-qt4-dev python-sip python-qt4-gl libqt4-scripttools
fi

# compile cairo
wget -nv http://cairographics.org/releases/py2cairo-1.10.0.tar.bz2
tar -xf py2cairo-1.10.0.tar.bz2
cd py2cairo-1.10.0
./waf configure
./waf build
sudo ./waf install
cd ..
