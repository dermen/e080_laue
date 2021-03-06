# e080 Laue
Scripts and instructions for processing the e080 dataset using diffBragg. If a the location of a filename in the instructions is not specified, assume it exists in this repository! Also, the input files required for this processing (optimized.expt and optimized.refl) were provided by Rick. They are on FAS cluster:

```
-rw-r--r-- 1 dermen hekstra_lab   181965 Feb 11 12:22 /n/holyscratch01/hekstra_lab/dermen/from_rick/optimized.expt
-rw-r--r-- 1 dermen hekstra_lab 28666953 Feb 11 12:22 /n/holyscratch01/hekstra_lab/dermen/from_rick/optimized.refl
```


## Instructions for setting up diffBragg on the [harvard cluster](https://www.rc.fas.harvard.edu/).

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
libtbx.python replace_path_expts.py  optimized.expt  optimized_newpath.expt /n/hekstra_lab/data/201903_APS_BioCARS/e080/
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

Get an initial spectrum suitable for diffBragg (the following script writes 4 spectra files at 1,2,3 and 4 eV resolutions)

```bash
libtbx.python fit_spec.py optimized.refl
```


<p align="center">
<img src=https://user-images.githubusercontent.com/2335439/155448196-86426657-b2a4-48bc-b37f-60c2f83212ba.png />
</p>

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

Note, the above command creates a folder `split` in the folder where the command was run from, and that should not be moved!

Download the PDB file needed to structure factor generation

```
iotbx.fetch_pdb 7lvc
```

# Processing

Grab a node for production work, we will allocate a session where every 2 ranks share 1 GPU

```bash
salloc --x11 -p seas_gpu_requeue -t 240 -n8 --mem 32G --constraint=v100 --gres=gpu:4

source ~/setup_db.sh
```

where `set_db.sh` contains the environment settings (note the path to `setpaths.sh` will depend on how you defined `CCTBXLAND` above, here `CCTBXLAND` is `~/xtal` ) , e.g.

```bash
# contents of setup_db.sh
module load cuda/11.1.0-fasrc01 gcc/8.3.0-fasrc01 openmpi
export SETUPTOOLS_USE_DISTUTILS=1
source ~/xtal/build/setpaths.sh 
```

Edit the script `hopper.phil` to contain the proper paths to any files and execute the command. Before doing multi process run, try a single image

```bash
export INPUT_F=/path/to/exp_ref_spec.txt
export OUTPUT_D=/somewheres/to/dump/outputs
DIFFBRAGG_USE_CUDA=1 srun --mpi=pmi2 simtbx.diffBragg.hopper hopper.phil  exp_ref_spec_file=$INPUT_F outdir=$OUTPUT_D num_devices=4 first_n=8

# examine the results
libtbx.python pred_offsets.py "$OUTPUT_D/refls/rank*/*.refl"
#median over 16 shots=0.569268 pixels (8624 refls)
#Min refls per shot=344, max refls per shot = 691, ave refls per shot=539.0

# produces the plot below
```
A median prediction offset of 0.57 pixels is a good start!

<p align="center">
<img src=https://user-images.githubusercontent.com/2335439/155447902-d87a02f3-b471-4e06-92ea-11aff82672f8.png />
</p>



