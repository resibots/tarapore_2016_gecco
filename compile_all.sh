#!/bin/bash

set -x
INSTALL="$(realpath install)"
echo "Install directory: ${INSTALL}"
NO_OSG=yes
OSG=no

echo "INSTALL: $INSTALL"
echo "NO_OSG: $NO_OSG"
echo "OSG: $OSG"

mkdir install

#### robdyn
cd robdyn
./waf configure  --disable_osg=$NO_OSG --prefix=$INSTALL
./waf install
cd ..

#### sferes
pwd
cd sferes2/exp
ln -s ../../exp/* .
cd ../modules
ln -s ../../modules/* .
cd ..
echo "robdyn" > modules.conf

CXXFLAGS='-fpermissive' ./waf configure --cpp11=yes --boost-libs=/usr/lib/x86_64-linux-gnu/ --robdyn=$INSTALL --eigen=/usr/include/eigen3

./waf


