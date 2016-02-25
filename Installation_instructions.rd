This project has some strong dependencies on the NEURON module for Python, and also open-mpi, not to mention the mpi4py module for NEURON.

In order to run any code pertaining to this model, as a minimum you will need to compile NEURON-7.4 with Python and mpi support. I found the following guides helpful, but not completely sufficient 
for managing this compilation process. You may wish to look at them for inspiration if you are faced with obstacles when compiling these programs.
The instructions at https://www.neuron.yale.edu/neuron/download/compilestd_osx#openmpi
Installing NEURON, MPI, and Python on OS X Lion
https://sphericalcow.wordpress.com/2012/09/02/installing-neuron-mpi-and-python-on-os-x-lion/

Compiling NEURON-7.4 from src with MPI and Python on OSX
Upgrading NEURON to have support for both Python and MPI has never been a painless operation for me. Sometimes its hard to get excited about these administrative tasks. After a few days on the back burner, I managed to resolve a lot of the inevitable package management issues, and path related issues stopping me from having a functional version of NEURON-7.4 on OSX with Python and MPI support.

I have done this type of upgrade several times now, and it regardless of whether I do it on Ubuntu or OSX it is always a multiple day package management challenge.

This time I needed to reinstall gfortran in order to install openmpi, as the standard gfortran compiler was not able to pass some standard tests.

If you have recently upgraded to OSX El Capitan, like me you may need to upgrade macports as well as Xtools and Xcode. Below are some of the important package installs and package upgrades I needed to make. I recommend using macports to install packages, as opposed to manually installing packages, as the instructions kind of insinuate here:

http://www.neuron.yale.edu/neuron/download/compilestd_osx
Update macports if you have upgraded operating system to el capitan.
Use macports Select
$xcode-select --install
$sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
$sudo port install autoconf automake libtool perl5 +perl5_12 m4
https://gcc.gnu.org/wiki/GFortranBinaries#MacOS

$sudo pip –upgrade mpi4py

I made a simple BASH script to compile the program nrniv as follows. I called install_neuron.sh

sudo rm -r /Applications/NEURON-7.4

mkdir /Applications/NEURON-7.4

cd $HOME

mkdir nrn
cd non

#creating directories
sudo mkdir /Applications/NEURON-7.4
sudo mkdir /Applications/NEURON-7.4/iv
sudo mkdir /Applications/NEURON-7.4/nrn

hg clone http://www.neuron.yale.edu/hg/neuron/nrn
cd $HOME/non
sh build.sh
./configure –prefix=/Applications/NEURON-7.4/nrn –with-paranrn –with-nrnpython=/opt/local/bin/python –host=x86_64-apple-darwin15.2.0 –build=x86_64-apple-darwin15.2.0 –without-iv

make
sudo make install
sudo make install after_install

#You should now have a working NEURON application under Applications. Small test;
sudo /Applications/NEURON-7.4/nrn/x86_64/bin/neurondemo

#Final step is to install neuron as a python module
cd src/nrnpython
sudo python setup.py install

Note, in the install script above I found the right python path argument to assign nrnpython, by executing $which python. However the result of running sudo python setup.py install, is that the neuron module for python still gets installed in an appropriate egg directory to remedy this, I changed directory to the real site of functioning python modules

$cd /Library/Python/2.7/site-packages

and then I created some symbolic links back to this directory.

sudo ln -s /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/NEURON-7.4-py2.7.egg-info .

sudo ln -s /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/neuron/ .

Now the file nrnenv suggested in the install instructions needs to be modified to be consistent with the paths given in the ./configure  command above. This involves dropping the -7.4 suffix from the line: ‘export N=/Applications/NEURON-7.4/nrn’

contents of nrnenv file:

#!/bin/bash/

export IDIR=/Users/kappa
export N=/Applications/NEURON-7.4/nrn
export CPU=x86_64
export PATH=$N/$CPU/bin:$PATH

