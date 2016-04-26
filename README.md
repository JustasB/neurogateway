##3Drodent stripped of dependencies such that it can run the [Neuroscience Gateway] (https://www.facebook.com/photo.php?fbid=226969161006304&set=a.107659599603928.1073741828.100010796645745&type=3&theater.)

##3Drodent:

A parallel algorithm for distance dependent wiring of arbitrarily detailed neuron morphologies. Outputs from the algorithm facilitate the study of network activity in 3D networks, making it possible to observe contributions from neurite atrophy and growth in 3D neural networks. This model has been implemented as an extension of the Allen Brain SDK Utils class.

The advantages to this wiring algorithm is that all the distance dependant calculations occur inside the Python-HOC frame work. No additional programs are run, and only one configuration file needs to be set. The network size (the number of neurons) and the number of CPUs used to execute code is flexible.

##[Results:](http://russelljjarvis.github.io/3Drodent/web/index.html)
View results after they have been commited to the gh-pages branch or in a similar file on your forked repository.


##This program has two use cases:
In the first case it can take advantage of digital representations of 3D neurons grown using artificial growth algorithms such as NeuroMAC and [L-NEURON](http://krasnow1.gmu.edu/cn3/L-Neuron/HTM/paper.htm) 
In the second case advances in light imaging technology are making neural network reconstruction more likely. The algorithm may take advantage of whole network reconstructions which that may be obtained in the future using light-sheet microscopy.

##Requirements:
This model needs an installation of NEURON-7.4/NEURON-7.3 configured to work with MPI and Python. Python must be able to use the NEURON module, and the NEURON module must be able to call Python.

Additionally the model depends on having a subdirectory 3Drodent/main populated by neuron morphology files in the format *.swc. These files are read in to the program systematically onto each host using a serialised python list in the pickle format. These files were downloaded on mass from www.neuromorpho.org. With some thought a similar list of of swc files could be created and turned into a pickled object. Note that all files with the *.p extension are pickled files.

Note that many of the neurons and ion channels represented in this model are merely place holders. Neuron morphology files, ion channel types, and synapse receptor types will be subsituted with more refined and accurate third party components as such model components become available via the open source brain research community. At the moment current model is not a valid or repeatable neuroscience model of any particular brain region, its more of a proof of concept of to facilitate that end.

##License 
Much of the code in this repository except for utils.py, init.py, and morph.hoc, and rigp.py is based on open source third party code. This project appeals to the license of the third party code which this project depends on. I will make this more explicit in the future.

The python methods that are used to achieve MPI aware distance dependent wiring are located in the Utils.py file. The relevant function definitions are:
```sh
def pre_synapse(self):
def post_synapse(self):
def wirecells(self):
```
##Installation Instructions: 

Instructions for Ubuntu-Linux are coming, this file should be updated in May 2016 a month after the release of Ubuntu Ubuntu 16.04 LTS Xenial Xerus April 2016. Note the Ubuntu instructions should be very similar to the OSX instructions, except that apt-get is used in place of macports, and the X-org and X-server is native to most Linux OS's

##Instructions for OSX:

This project has some strong dependencies on the NEURON module for Python, and also open-mpi, not to mention the mpi4py module for NEURON.

In order to run any code pertaining to this model, as a minimum you will need to compile NEURON-7.4 with Python and mpi support. I found the following guides helpful, but not completely sufficient 
for managing this compilation process. You may wish to look at them for inspiration if you are faced with obstacles when compiling these programs.
[The instructions at](https://www.neuron.yale.edu/neuron/download/compilestd_osx#openmpi)
[Installing NEURON, MPI, and Python on OS X Lion](https://sphericalcow.wordpress.com/2012/09/02/installing-neuron-mpi-and-python-on-os-x-lion/
)

It is important to note that the version of Python that is built into OSX is not very suitable for scientific programming, and that it is appropriate to install an additional Python binary using macports.

##Compiling NEURON-7.4 from src with MPI and Python on OSX 

This is not a comprehensive guide, but its an informed example based on my own experience. If you have recently upgraded to OSX El Capitan, like me you may need to upgrade macports as well as Xtools and Xcode. Below are some of the important package installs and package upgrades I needed to make. I recommend using macports to install packages, as opposed to manually installing packages, [http://www.neuron.yale.edu/neuron/download/compilestd_osx
](as these instructions insinuate).

Update macports if you have upgraded operating system to el capitan.
Use macports Select
```sh
$xcode-select --install
$sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
$sudo port install autoconf automake libtool perl5 +perl5_12 m4
```

This time I needed to reinstall gfortran in order to install openmpi, as the standard gfortran compiler was not able to pass some standard tests. I had to reinstall [https://gcc.gnu.org/wiki/GFortranBinaries#MacOS](gfortran) as the log generated by ./configure told my that my current gfortran program/configuration was not passing basic tests.
```sh
$sudo pip –upgrade mpi4py
```
I made a simple BASH script to compile the program nrniv as follows. The script was made by lumping togethor commands in the compilation instructions at [http://www.neuron.yale.edu/neuron/download/compilestd_osx
](as these instructions insinuate). I called install_neuron.sh
```sh
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
```
Note, in the install script above you can find the right python path argument to assign nrnpython by executing $which python. However the result of running sudo python setup.py install, is that the neuron module for python still gets installed in an appropriate egg directory to remedy this, I changed directory to the real site of functioning python modules

$cd /Library/Python/2.7/site-packages

and then I created some symbolic links back to this directory.
```sh
sudo ln -s /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/NEURON-7.4-py2.7.egg-info .

sudo ln -s /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/neuron/ .
```
Now the file nrnenv suggested in the install instructions needs to be modified to be consistent with the paths given in the ./configure  command above. This involves dropping the -7.4 suffix from the line: ‘export N=/Applications/NEURON-7.4/nrn’

contents of nrnenv file:

```sh
#!/bin/bash/
export IDIR=/Users/kappa
export N=/Applications/NEURON-7.4/nrn
export CPU=x86_64
export PATH=$N/$CPU/bin:$PATH
```

##After NEURON+Python+MPI is built:

##More Python packages required that are required beyond the operation of NEURON+Python+MPI 
Install Quartz X11 for OSX
Use macports/apt-get to install python27 in addition to the version of Python that ships with OSX.

```sh
sudo pip install allensdk glob2 unittest numpy scipy
```
Note: The models dependency on the Allen Brain SDK, numba, and neuro_electro is very weak and could easily be factored out.

```sh
$git clone https://github.com/numba/numba.git
$cd numba
sudo python setup.py install

```


Once NEURON+Python+MPI has been successfuly built, the next step is to build the NMODL code. The NMODL code is comprised by files with the extension .mod.

Currently the main directory contains the *.mod files (TODO move these to a seperate directory). Navigating to the root 3Drodent dir and executing 
$nrnivmodl 
is sufficient to build the nmodl code before running the model.

##Running the model
To facilitate easy running of the model  (in both OSX and Ubuntu) I have created the BASH alias:
```sh
alias rspp='cd /git/3Drodentm; mpiexec -np 4 xterm -e "ipython -i use_local_files.py"'

```
The argument -np 4 specifies the number of processes to be used, you may add more or less as required.

Adding the alias to my ~/.bashrc file means that I can launch the model simply by executing $rspp at the command line.
However note that this depends on the program xterm being available. In Ubuntu xterm can readily by installed using apt-get however in OSX xterm depends on XQuartz the X11 server which will need to be installed. [Instructions for installing XQuartz](https://www.neuron.yale.edu/neuron/download/compilestd_osx)

An alias can also be used to kill all invocations of xterm.
```sh
alias kx='pgrep xterm | xargs kill'
```
Other command line shortcuts that are sometimes useful for viewing results include:
```sh
on Ubuntu:
$eog  membrane_traces*.png

on OSX:
$open preview  membrane_traces*.png

```
