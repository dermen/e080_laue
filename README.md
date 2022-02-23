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

## Pre-Processing

Get the tarball from Rick

Replace the paths in the expts which point to Ricks private folder:

```bash
libtbx.python replace_path_expts.py  optimized.expt  optimized_newpath.expt
```

Unpack the mask (NOTE: this mask will be used throughout analysis!)

```
cd /wherevers/e080_laue
libtbx.python unpack_mask.py
```

Use dials to get the strong reflections. Use the unpacked mask.

```bash
dials.find_spots optimized_newpath.expt  sigma_strong=1 sigma_background=1 mask=~/e080_laue/newbad.pkl  gain=0.4 kernel_size=[2,2] output.reflections=strongs.refl min_spot_size=3 filter.d_min=1.493 max_spot_size=90
```

Split expts and refls into single files:

```bash
mkdir split ; cd split
dials.split_experiments ../optimized_newpath.expt ../strongs.refls
```

Get an initial spectrum suitable for diffBragg (the following script writes 4 spectra files at 1,2,3 and 4 eV resolutions)

```bash
libtbx.python fit_spec.py
```

There is a convenience script to check the eV per pixel at various resolutions to gauge the necessary energy resolution for the diffBragg input spectrum

```
libtbx.python estimate_ev_per_pix.py 
	eV spread per pixel is 18.7475 eV at 4.0 Angstrom resolution

	eV spread per pixel is 13.2167 eV at 3.0 Angstrom resolution

	eV spread per pixel is 10.3319 eV at 2.5 Angstrom resolution

	eV spread per pixel is 7.3135 eV at 2.0 Angstrom resolution

	eV spread per pixel is 4.1113 eV at 1.5 Angstrom resolution

	eV spread per pixel is 2.4612 eV at 1.2 Angstrom resolution

```

Based on these numbers, it seems 3eV is suitable (1.2 Angstrom is the corner res)

Make the input file for diffBragg

```bash
libtbx.python make_input_f.py  optimized_newpath.expt  strongs.refl  spec_3eV.lam 
```

Download the PDB file needed to structure factor generation

```
iotbx.fetch_pdb 7lvc
```


