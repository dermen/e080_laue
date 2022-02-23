import argparse as ap 
parser = ap.ArgumentParser() 
parser.add_argument("philFile", type=str, help="path to a phil config file")
parser.add_argument("inputGlob", type=str, help="glob of input pandas tables (those that are output by simtbx.diffBragg.hopper or diffBragg.hopper_process")
parser.add_argument("outdir", type=str, help="path to output refls")
parser.add_argument("--cmdlinePhil", nargs="+", default=None, type=str, help="command line phil params")
parser.add_argument("--numdev", type=int, default=1, help="number of GPUs (default=1)")
parser.add_argument("--weakFrac", type=float, default=0, help="Fraction of weak reflections to keep (default=0; allowed range: 0-1 )")
args = parser.parse_args()

from mpi4py import MPI
import numpy as np
from dials.array_family import flex
from simtbx.diffBragg import utils
COMM = MPI.COMM_WORLD
from simtbx.diffBragg import utils
from simtbx.modeling import predictions
import glob
import pandas
import os
import shutil
from scipy.interpolate import interp1d

if COMM.rank==0:
    if not os.path.exists( args.outdir):
        os.makedirs(args.outdir)
COMM.barrier()

params = utils.get_extracted_params_from_phil_sources(args.philFile, args.cmdlinePhil)
fnames = glob.glob(args.inputGlob)
if params.predictions.verbose:
    params.predictions.verbose = COMM.rank==0

dev = COMM.rank % args.numdev

def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)

print0("Found %d input files" % len(fnames))

all_wave = []
npred_per_shot = []
for i_f, f in enumerate(fnames):
    if i_f % COMM.size != COMM.rank:
        continue
    print0("Shot %d / %d" % (i_f, len(fnames)))

    df = pandas.read_pickle(f)

    expt_name = df.opt_exp_name.values[0]
    new_expt_name = "%s/pred%d.expt" % (args.outdir, i_f)
    shutil.copyfile(expt_name,  new_expt_name)

    pred = predictions.get_predicted_from_pandas(
        df, params, strong=None, device_Id=dev, spectrum_override=None)

    Rstrong = flex.reflection_table.from_file(df.strong_refl.values[0])
    predictions.label_weak_predictions(pred, Rstrong, q_cutoff=8, col="xyzobs.px.value" )
    strong_sel = np.logical_not(pred['is_weak'])

    pred["refl_idx"] = flex.int(np.arange(len(pred))) 
    weaks = pred.select(pred['is_weak'])
    weaks_sorted = np.argsort(weaks["scatter"])[::-1]
    nweak = len(weaks)
    num_keep = int(nweak*args.weakFrac)
    weak_refl_inds_keep = set(np.array(weaks["refl_idx"])[weaks_sorted[:num_keep]])

    weak_sel = flex.bool([i in weak_refl_inds_keep for i in pred['refl_idx']])
    keeps = np.logical_or( strong_sel, weak_sel)
    print("Sum keeps=%d; num_strong=%d, num_kept_weak=%d" % (sum(keeps), sum(strong_sel), sum(weak_sel)))
    pred = pred.select(flex.bool(keeps))
    nstrong = np.sum(strong_sel)
    all_wave  += list(pred["ave_wavelen"])
    print("Will save %d refls" % len(pred))
    npred_per_shot.append(len(pred))
    pred.as_file("%s/pred%d.refl" % ( args.outdir, i_f))

npred_per_shot = COMM.reduce(npred_per_shot)
all_wave = COMM.reduce(all_wave)

if COMM.rank==0:
    from pylab import *
    style.use("ggplot")
    #a2,b2 = utils.load_spectra_file("spectra/model_0_4eV.lam")
    a2,b2 = utils.load_spectra_file("spectra/model_0_4eV_shiftBack.lam")

    out = hist( utils.ENERGY_CONV/array(all_wave), bins=100)
    plot( b2, a2*out[0].max(), label="reference spectrum")
    xlabel("energy (eV)")
    ylabel("# of spots")
    legend()
    s="Min pred=%d; Max pred = %d; Ave pred=%d" % (min(npred_per_shot), max(npred_per_shot), mean(npred_per_shot))
    title("Total number of predictions=%d; threshold=%1.1e\n%s" % (len(all_wave),  params.predictions.threshold, s))
    savefig("preds_%1.1e.png" % params.predictions.threshold)

print("Reflections written to folder %s" % args.outdir)
