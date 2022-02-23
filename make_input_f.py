from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("expt", help="path to combined expt file", type=str)
parser.add_argument("refl", help="path to combined refl file", type=str)
parser.add_argument("spec", help="path to .lam file", type=str)
parser.add_argument("--ers", default="exp_ref_spec.txt",  help="writes a file with this name, a suitable input for simtbx.diffBragg.hopper", type=str)
parser.add_argument("--split", help="name of empty folder where split expts/refls will be stored (default=./split) (will be created if non existent)", type=str, default="split")

args = parser.parse_args()

import os
import sys
import glob
A = os.path.abspath

expts = os.path.abspath(args.expt)
refls = os.path.abspath(args.refl)


if not os.path.exists(args.split):
    os.makedirs(args.split)

if any(os.scandir(args.split)):
    raise OSError("The split folder (see --split argument) should be empty!" )


CWD = os.path.abspath(os.getcwd())
os.chdir(args.split)
os.system("dials.split_experiments %s %s" % (expts, refls))

fnames = [A(f) for f in glob.glob("./split*.expt")]
refls_fnames = [f.replace(".expt", ".refl") for f in fnames]

assert all(os.path.exists(f) for f in refls_fnames)

os.chdir(CWD)

ers = A(args.ers)
o = open(ers, "w")
for e,r in zip(fnames, refls_fnames):
    o.write("%s %s %s\n" % (A(e),A(r),A(args.spec)))
o.close()
print("Wrote %d expts to %s ,  the input `exp_ref_spec_file` for simtbx.diffBragg.hopper" % (len(fnames), ers))
