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
ln -s ../../modules/nn2
ln -s ../../robdyn/sferes/robdyn
cd ..
echo "robdyn" > modules.conf
# yes, we use eigen2...
./waf configure --robdyn=$INSTALL --robdyn-osg=$OSG --eigen=/usr/include/eigen2

./waf


