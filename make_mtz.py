
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("glob", type=str, help="input glob of int refls")
parser.add_argument("mtz", type=str, help="output mtz name")
parser.add_argument("--max", default=2500, type=int, help="max number of refls in a shot (default 2500)")
parser.add_argument("--ucell", default=None, nargs=6, type=float, help="unit cell params (default will be average experiment crystal)")
args = parser.parse_args()

import glob
import os
import numpy as np
from dials.array_family import flex
from cctbx import miller, crystal
from iotbx import mtz
from copy import deepcopy
from dxtbx.model import ExperimentList
import sys

fnames = glob.glob(args.glob)

vals = []
batch_num_from_f = {}
for f in fnames:
    f2 = f.replace(".refl", ".expt")
    path = ExperimentList.from_file(f2, False)[0].imageset.get_path(0)
    mccd_num = path.split("_")[-1].split(".mccd")[0]
    batch_num = int(mccd_num)-1
    vals.append([int(mccd_num)-1,f])
    batch_num_from_f[f] = batch_num

_,fnames = map(list, zip(* sorted(vals)))
all_ucell_p = []

for f in fnames:
    f2 = f.replace(".refl", ".expt")
    exp = ExperimentList.from_file(f2, False)[0]
    ucell_p = exp.crystal.get_unit_cell().parameters()
    all_ucell_p.append( ucell_p)
    path = exp.imageset.get_path(0)
    print(path)

ucell = args.ucell
if args.ucell is None:
    ucell = list(np.median(all_ucell_p, axis=0))

R = None
for i_shot,f in enumerate(fnames):
    refls = flex.reflection_table.from_file(f)
    batch_num = batch_num_from_f[f]
    refls['id'] = flex.int(len(refls), batch_num)
    if len(refls) > args.max:
        print("Not processing shot %d because too many (%d) refls!!!" % (i_shot, len(refls)))
        continue
    if R is None:
        R = refls
    else:
        R.extend(refls)
    print(batch_num, len(refls))

sy = crystal.symmetry(ucell, "P212121")

wave = R['ave_wavelen']
x,y,_ = R['xyzobs.px.value'].parts()
hkl = R['miller_index']
hkl_orig = deepcopy(hkl)
miller.map_to_asu(sy.space_group().type(), False, hkl)
h,k,l = zip(*list(hkl))
I_vals = R['intensity.sum.value']
sigI_vals = flex.sqrt(R['intensity.sum.variance'])
batch = R['id']


MTZ = mtz.object()
MTZ.set_title("shouldnt_suck")
MTZ.set_space_group_info(sy.space_group_info())
MTZ.adjust_column_array_sizes(len(R))
MTZ.set_n_reflections(len(R))
C = MTZ.add_crystal("Crys", "CrysProj", ucell)
dset = C.add_dataset("obs", 0)

I = dset.add_column("I","J")
I.set_values(flex.float(list(I_vals)))

WAVE = dset.add_column('Wavelength', 'R')
WAVE.set_values(flex.float(list(wave)))

Y = dset.add_column('Y', 'R')
Y.set_values(flex.float(list(y)))

X = dset.add_column('X', 'R')
X.set_values(flex.float(list(x)))

H = dset.add_column('H','H')
H.set_values(flex.float(list(h)))

K = dset.add_column('K','H')
K.set_values(flex.float(list(k)))

L = dset.add_column('L','H')
L.set_values(flex.float(list(l)))

SigI = dset.add_column("SigI", "Q")
SigI.set_values(flex.float(list(sigI_vals)))

BATCH = dset.add_column("BATCH", "B")
BATCH.set_values(flex.float(list(batch)))

misym = dset.add_column("M_ISYM", "Y")
misym.set_values(flex.float(len(R)))
MTZ.replace_original_index_miller_indices(hkl_orig)

MTZ.write(args.mtz)
os.system("iotbx.mtz.dump %s" % args.mtz)

