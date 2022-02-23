# e080 Laue
Scripts and instructions for processing the e080 dataset using diffBragg

# Instructions for setting up diffBragg on the [harvard cluster](https://www.rc.fas.harvard.edu/).

Environment:

```bash
salloc --x11 -p seas_gpu_requeue --gres=gpu  -t 240 -n 1 --cpus-per-task 32 --mem 32G --constraint=v100

module load cuda/11.1.0-fasrc01 gcc/8.3.0-fasrc01 openmpi
export SETUPTOOLS_USE_DISTUTILS=1  # IMPORTANT FOR UNKNOWN REASONS!
export CCTBXLAND=$HOME/cctbx_land
```

Build instructions:

```bash
mkdir $CCTBXLAND
cd $CCTBXLAND
# get cctbx bootstrap https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py
python bootstrap.py  hot update --builder dials --python=38 --use-conda

python bootstrap.py  base --builder dials --python=38 --use-conda=./conda_base 

# the above two commands take FOREVER on the cluster, if you time out, log back in and reset the modules! Theres something weird with the network requests maybe... 

# we will skip iota module build, as the SETUPTOOLS_USE_DISTUTILS=1 flag causes it to fail
cd $CCTBXLAND/modules
mv iota iotaBAK
cd $CCTBXLAND

# build
python bootstrap.py  build --builder dials --python=38 --use-conda=./conda_base --nproc=40 --config-flags="--enable_cuda" --config-flags="--enable_openmp_if_possible=True" --config-flags="--enable_cxx11"

# set your build environment
source $CCTBXLAND/build/setpaths.sh

# for ipython (in case it doesnt install automatically)
libtbx.python -m pip install jupyter
libtbx.refresh
# then one can do the following to launcher an ipython shell :
libtbx.ipython

# install mpi4py (not sure if the CC def is necessary)
CC=/n/helmod/apps/centos7/Core/gcc/8.3.0-fasrc01/bin/gcc MPICC=mpicc libtbx.python -m pip install -v --no-binary mpi4py mpi4py

```

To use:

```bash
salloc --x11 -p seas_gpu_requeue --gres=gpu  -t 240 -n 1 --cpus-per-task 32 --mem 32G --constraint=v100

module load cuda/11.1.0-fasrc01 gcc/8.3.0-fasrc01 openmpi
export SETUPTOOLS_USE_DISTUTILS=1 # not sure if needed after building
source $CCTBXLAND/build/setpaths.sh
```

To test

```bash
# go to a clean folder and run
libtbx.run_tests_parallel nproc=Auto module=simtbx
```

To test mpi (note the `--mpi=mpi2` flag is important lest you receive messages of doom)

```bash
srun -n16 -c2  --mpi=pmi2 libtbx.python -c "from mpi4py import MPI;C=MPI.COMM_WORLD;print(C.rank,C.size)"
```

Potential issues

* Eigen version: if the build step fails with error messages indicating eigen as the culprit, ensure your build is using the stable eigen 3.4 release:

```bash
cd $CCTBXLAND/modules
rm -rf eigen
wget  https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
ln -s $PWD/eigen-3.4.0 eigen
```

