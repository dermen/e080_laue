

import argparse as ap

parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
parser.add_argument("glob", type=str, help="glob for pandas pickles output by simtbx.diffBragg.hopper or diffBragg.hopper_process")
parser.add_argument("out", type=str, help="output pickle filename, which will then be a suitable input for diffBragg.geometry_refiner")
parser.add_argument("--thresh", type=float, default=10, help="MAD score threshold for outliers. Lower to remove more outliers")
parser.add_argument("--tag", type=str, default="filt", help="New refl tables will be written with this tag added to end of the filename")
parser.add_argument("--ers", type=str, default=None, help="if provided, write an exp-ref-spec file suitable input for simtbx.diffBragg.hopper")
parser.add_argument("--mind", type=float, default=None, help="if provided, filter shots whose median prediction offset is above this number (units=pixels)")

args = parser.parse_args()

import pandas
import glob
import h5py
from dials.array_family import flex
import numpy as np
from simtbx.diffBragg import utils

fnames = glob.glob(args.glob)
df = pandas.concat( [pandas.read_pickle(f) for f in fnames])
df.reset_index(inplace=True, drop=True)
R2names = []

def get_dist_from_R(R):
    """ returns prediction offset, R is reflection table"""
    x,y,_ = R['xyzobs.px.value'].parts()
    x2,y2,_ = R['xyzcal.px'].parts()
    dist = np.sqrt((x-x2)**2 + (y-y2)**2)
    return dist

keep = []
all_d = []
all_d2 = []
n = 0
n2 = 0
for i in range(len(df)):
    row = df.iloc[i]
    h = h5py.File(row.stage1_output_img, 'r')
    Rname = row.opt_exp_name.replace("/expers/", "/refls/").replace(".expt", ".refl")
    R = flex.reflection_table.from_file(Rname)
    d = np.median(get_dist_from_R(R))
    n += len(R)
    all_d.append(d)
    if args.mind is not None:
        keep.append( d < args.mind)
    else:
        keep.append(True)

    vals = h['sigmaZ_vals'][()]
    vals[np.isnan(vals)] = np.inf

    bad = list(np.where(utils.is_outlier(vals, args.thresh))[0])
    sel = [R[i_r]['h5_roi_idx'] not in bad for i_r in range(len(R))]
    R2 = R.select(flex.bool(sel))
    d2 = np.median(get_dist_from_R(R2))
    all_d2.append(d2)
    if args.tag is not None:
        R2name = Rname.replace(".refl", "_%s.refl" % args.tag)
        R2.as_file(R2name)
        print(i, len(df), "New refl table written=%s" % R2name)
    R2names.append(R2name)
    n2 += len(R2)
    
    
df['filtered_refls'] = R2names
df = df.loc[keep]
df.reset_index(inplace=True, drop=True)
df.to_pickle(args.out)
print("\nSummary\n<><><><><>")

print("Wrote %s which can be passed into diffBragg.geometry_refiner  input_pickle=%s" % (args.out, args.out))
print("Kept %d / %d refls. Removed %.2f %% " 
    % (n2, n, (n-n2)/float(n)*100. ))
if args.ers is not None:
    with open(args.ers, "w") as ersFile:
        for e, r, s in df[["exp_name", "filtered_refls", "spectrum_filename"]].values:
            ersFile.write("%s %s %s\n" % (e,r,s))
    print("Wrote %s which can be passed into simtbx.diffBragg.hopper exp_ref_spec_file=%s" % (args.ers, args.ers))
