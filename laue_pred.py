
import argparse as ap 
parser = ap.ArgumentParser() 
parser.add_argument("predPhil", type=str, help="path to a phil config file for diffbragg prediction")
parser.add_argument("procPhil", type=str, help="path to a phil config file for stills process (used for spot finding and integration)")
parser.add_argument("inputGlob", type=str, help="glob of input pandas tables (those that are output by simtbx.diffBragg.hopper or diffBragg.hopper_process")
parser.add_argument("outdir", type=str, help="path to output refls")

parser.add_argument("--cmdlinePhil", nargs="+", default=None, type=str, help="command line phil params")
parser.add_argument("--numdev", type=int, default=1, help="number of GPUs (default=1)")
parser.add_argument("--weakFrac", type=float, default=0, help="Fraction of weak reflections to keep (default=0; allowed range: 0-1 )")
parser.add_argument("--pklTag", type=str, help="optional suffix for globbing for pandas pickles (default .pkl)", default=".pkl")
parser.add_argument("--loud", action="store_true", help="show lots of screen output")
parser.add_argument("--hopInputName", default="preds_for_hopper", type=str, help="write exp_ref_spec file and best_pickle pointing to the preditction models, such that one can run predicted rois through simtbx.diffBragg.hopper (e.g. to fit per-roi scale factors)")

args = parser.parse_args()

from mpi4py import MPI
COMM = MPI.COMM_WORLD

def printR(*args, **kwargs):
    print("RANK %d" % COMM.rank, *args, **kwargs)
def print0(*args, **kwargs):
    if COMM.rank==0:
        print(*args, **kwargs)

import numpy as np
from dials.array_family import flex
from simtbx.diffBragg import utils
from simtbx.modeling import predictions
import glob
import pandas
import os
import shutil
from scipy.interpolate import interp1d
from dxtbx.model import ExperimentList
from dials.algorithms.integration.stills_significance_filter import SignificanceFilter
from dials.algorithms.indexing.stills_indexer import calc_2D_rmsd_and_displacements
import logging
import sys

if not args.loud:
    logging.disable(logging.CRITICAL)
else:
    logging.basicConfig(level=logging.DEBUG)


# Note: these imports and following 3 methods will eventually be in CCTBX/simtbx/diffBragg/utils
from dials.algorithms.spot_finding.factory import SpotFinderFactory
from dials.algorithms.spot_finding.factory import FilterRunner
from dials.model.data import PixelListLabeller, PixelList
from dials.algorithms.spot_finding.finder import pixel_list_to_reflection_table
from libtbx.phil import parse
from dials.command_line.stills_process import phil_scope





from dials.algorithms.integration.integrator import create_integrator
from dials.algorithms.profile_model.factory import ProfileModelFactory
from dxtbx.model import ExperimentList
from dials.array_family import flex

for i,arg in enumerate(sys.argv):
    if os.path.isfile(arg) or os.path.isdir(arg):
        sys.argv[i] = os.path.abspath(arg)
print0("COMMANDLINE: libtbx.python %s" % " ".join(sys.argv))

def stills_process_params_from_file(phil_file):
    """
    :param phil_file: path to phil file for stills_process
    :return: phil params object
    """
    phil_file = open(phil_file, "r").read()
    user_phil = parse(phil_file)
    phil_sources = [user_phil]
    working_phil, unused = phil_scope.fetch(
        sources=phil_sources, track_unused_definitions=True)
    params = working_phil.extract()
    return params



def process_reference(reference):
    """Load the reference spots."""
    assert "miller_index" in reference
    assert "id" in reference
    mask = reference.get_flags(reference.flags.indexed)
    rubbish = reference.select(~mask)
    if mask.count(False) > 0:
        reference.del_selected(~mask)
    if len(reference) == 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    Expected > %d indexed spots, got %d
  """
            % (0, len(reference))
        )
    mask = reference["miller_index"] == (0, 0, 0)
    if mask.count(True) > 0:
        rubbish.extend(reference.select(mask))
        reference.del_selected(mask)
    mask = reference["id"] < 0
    if mask.count(True) > 0:
        raise RuntimeError(
            """
    Invalid input for reference reflections.
    %d reference spots have an invalid experiment id
  """
            % mask.count(True)
        )
    return reference, rubbish



def integrate(phil_file, experiments, indexed, predicted):
    """
    integrate a single experiment at the locations specified by the predicted table
    The predicted table should have a column specifying strong reflections
    """
    assert len(experiments)==1

    for refls in [predicted, indexed]:
        refls['id'] = flex.int(len(refls), 0)
        refls['entering'] = flex.bool(len(refls), False)
        eid = refls.experiment_identifiers()
        for k in eid.keys():
            del eid[k]
        eid[0] = '0'
    experiments[0].identifier = '0'

    params = stills_process_params_from_file(phil_file) 
    indexed,_ = process_reference(indexed)
    experiments = ProfileModelFactory.create(params, experiments, indexed)

    new_experiments = ExperimentList()
    new_reflections = flex.reflection_table()
    for expt_id, expt in enumerate(experiments):
        if (
                params.profile.gaussian_rs.parameters.sigma_b_cutoff is None
                or expt.profile.sigma_b()
                < params.profile.gaussian_rs.parameters.sigma_b_cutoff
        ):
            refls = indexed.select(indexed["id"] == expt_id)
            refls["id"] = flex.int(len(refls), len(new_experiments))
            del refls.experiment_identifiers()[expt_id]
            refls.experiment_identifiers()[len(new_experiments)] = expt.identifier
            new_reflections.extend(refls)
            new_experiments.append(expt)

    experiments = new_experiments
    indexed = new_reflections
    if len(experiments) == 0:
        raise RuntimeError("No experiments after filtering by sigma_b")

    predicted.match_with_reference(indexed)
    integrator = create_integrator(params, experiments, predicted)
    integrated = integrator.integrate()

    if params.significance_filter.enable:

        sig_filter = SignificanceFilter(params)
        filtered_refls = sig_filter(experiments, integrated)
        accepted_expts = ExperimentList()
        accepted_refls = flex.reflection_table()
        for expt_id, expt in enumerate(experiments):
            refls = filtered_refls.select(filtered_refls["id"] == expt_id)
            if len(refls) > 0:
                accepted_expts.append(expt)
                refls["id"] = flex.int(len(refls), len(accepted_expts) - 1)
                accepted_refls.extend(refls)

        if len(accepted_refls) == 0:
            raise RuntimeError("No reflections left after applying significance filter")
        experiments = accepted_expts
        integrated = accepted_refls

    # Delete the shoeboxes used for intermediate calculations, if requested
    if params.integration.debug.delete_shoeboxes and "shoebox" in integrated:
        del integrated["shoebox"]


    rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
    log_str = "RMSD indexed (px): %f\n" % rmsd_indexed
    for i in range(6):
        bright_integrated = integrated.select(
            (
                    integrated["intensity.sum.value"]
                    / flex.sqrt(integrated["intensity.sum.variance"])
            )
            >= i
        )
        if len(bright_integrated) > 0:
            rmsd_integrated, _ = calc_2D_rmsd_and_displacements(bright_integrated)
        else:
            rmsd_integrated = 0
        log_str += (
                "N reflections integrated at I/sigI >= %d: % 4d, RMSD (px): %f\n"
                % (i, len(bright_integrated), rmsd_integrated)
        )

    for crystal_model in experiments.crystals():
        if hasattr(crystal_model, "get_domain_size_ang"):
            log_str += ". Final ML model: domain size angstroms: {:f}, half mosaicity degrees: {:f}".format(
                crystal_model.get_domain_size_ang(),
                crystal_model.get_half_mosaicity_deg(),
            )

    #print0(log_str)
    return experiments, integrated




def dials_find_spots(data_img, params, trusted_flags=None):
    """
    :param data_img: numpy array image
    :param params: instance of stills_process params.spotfinder
    :param trusted_flags:
    :return:
    """
    if trusted_flags is None:
        trusted_flags = np.ones(data_img.shape, bool)
    thresh = SpotFinderFactory.configure_threshold(params)
    flex_data = flex.double(np.ascontiguousarray(data_img))
    flex_trusted_flags = flex.bool(np.ascontiguousarray(trusted_flags))
    spotmask = thresh.compute_threshold(flex_data, flex_trusted_flags)
    return spotmask.as_numpy_array()


def refls_from_sims(panel_imgs, detector, beam, thresh=0, filter=None, panel_ids=None,
                    max_spot_size=1000, phil_file=None, **kwargs):
    """
    This is for converting the centroids in the noiseless simtbx images
    to a multi panel reflection table
    :param panel_imgs: list or 3D array of detector panel simulations
    :param detector: dxtbx  detector model of a caspad
    :param beam:  dxtxb beam model
    :param thresh: threshol intensity for labeling centroids
    :param filter: optional filter to apply to images before
        labeling threshold, typically one of scipy.ndimage's filters
    :param pids: panel IDS , else assumes panel_imgs is same length as detector
    :param kwargs: kwargs to pass along to the optional filter
    :return: a reflection table of spot centroids
    """
    if panel_ids is None:
        panel_ids = np.arange(len(detector))
    pxlst_labs = []
    badpix_all =None
    for i, pid in enumerate(panel_ids):
        plab = PixelListLabeller()
        img = panel_imgs[i]
        if phil_file is not None:
            params = stills_process_params_from_file(phil_file)
            badpix = None
            if params.spotfinder.lookup.mask is not None:
                if badpix_all is None:
                    badpix_all = utils.load_mask(params.spotfinder.lookup.mask)
                badpix = badpix_all[pid]
            mask = dials_find_spots(img, params, badpix)
        elif filter is not None:
            mask = filter(img, **kwargs) > thresh
        else:
            mask = img > thresh
        img_sz = detector[int(pid)].get_image_size()  # for some reason the int cast is necessary in Py3
        flex_img = flex.double(img)
        flex_img.reshape(flex.grid(img_sz))

        flex_mask = flex.bool(mask)
        flex_mask.resize(flex.grid(img_sz))
        pl = PixelList(0, flex.double(img), flex.bool(mask))
        plab.add(pl)

        pxlst_labs.append(plab)

    El = utils.explist_from_numpyarrays(panel_imgs, detector, beam)
    iset = El.imagesets()[0]
    refls = pixel_list_to_reflection_table(
        iset, pxlst_labs,
        min_spot_size=1,
        max_spot_size=max_spot_size,  # TODO: change this ?
        filter_spots=FilterRunner(),  # must use a dummie filter runner!
        write_hot_pixel_mask=False)[0]
    if phil_file is not None:
        x,y,z = refls['xyzobs.px.value'].parts()
        x -=0.5
        y -=0.5
        refls['xyzobs.px.value'] = flex.vec3_double(x,y,z)

    return refls




if __name__=="__main__":

    if COMM.rank==0:
        if not os.path.exists( args.outdir):
            os.makedirs(args.outdir)
    COMM.barrier()

    params = utils.get_extracted_params_from_phil_sources(args.predPhil, args.cmdlinePhil)
    #fnames = glob.glob(args.inputGlob + "/*%s" % args.pklTag)
    fnames = glob.glob(args.inputGlob) # + "/*%s" % args.pklTag)
    #if not fnames:
    #    fnames = glob.glob(args.inputGlob + "/rank*/*%s" % args.pklTag)
    if params.predictions.verbose:
        params.predictions.verbose = COMM.rank==0

    dev = COMM.rank % args.numdev


    print0("Found %d input files" % len(fnames))

    all_wave = []
    npred_per_shot = []
    all_dfs = []
    exp_ref_spec_lines = []
    for i_f, f in enumerate(fnames):
        if i_f % COMM.size != COMM.rank:
            continue
        printR("Shot %d / %d" % (i_f+1, len(fnames)), flush=True)

        df = pandas.read_pickle(f)

        expt_name = df.opt_exp_name.values[0]
        tag = os.path.splitext(os.path.basename(expt_name))[0]
        new_expt_name = "%s/%s_pred%d.expt" % (args.outdir,tag,  i_f)
        shutil.copyfile(expt_name,  new_expt_name)
        new_expt_name = os.path.abspath(new_expt_name)
        df["opt_exp_name"] = new_expt_name
        

        pred = predictions.get_predicted_from_pandas(
            df, params, strong=None, device_Id=dev, spectrum_override=None)

        data_exptList = ExperimentList.from_file(expt_name)
        data_expt = data_exptList[0]
        data = utils.image_data_from_expt(data_expt) 
        Rstrong = refls_from_sims(data, data_expt.detector, data_expt.beam, phil_file=args.procPhil )
        predictions.label_weak_predictions(pred, Rstrong, q_cutoff=8, col="xyzobs.px.value" )
        
        pred['is_strong'] = flex.bool(np.logical_not(pred['is_weak']))
        strong_sel = np.logical_not(pred['is_weak'])

        pred["refl_idx"] = flex.int(np.arange(len(pred))) 
        weaks = pred.select(pred['is_weak'])
        weaks_sorted = np.argsort(weaks["scatter"])[::-1]
        nweak = len(weaks)
        num_keep = int(nweak*args.weakFrac)
        weak_refl_inds_keep = set(np.array(weaks["refl_idx"])[weaks_sorted[:num_keep]])

        weak_sel = flex.bool([i in weak_refl_inds_keep for i in pred['refl_idx']])
        keeps = np.logical_or( pred['is_strong'], weak_sel)
        printR("Sum keeps=%d; num_strong=%d, num_kept_weak=%d" % (sum(keeps), sum(strong_sel), sum(weak_sel)))
        pred = pred.select(flex.bool(keeps))
        nstrong = np.sum(strong_sel)
        all_wave  += list(pred["ave_wavelen"])
        printR("Will save %d refls" % len(pred))
        npred_per_shot.append(len(pred))
        pred_file = os.path.abspath("%s/%s_pred%d.refl" % ( args.outdir, tag, i_f))
        pred.as_file(pred_file)

        Rindexed = Rstrong.select(Rstrong['indexed'])
        utils.refls_to_hkl(Rindexed, data_expt.detector, data_expt.beam, data_expt.crystal, update_table=True)
        int_expt, int_refl = integrate(args.procPhil, data_exptList, Rindexed, pred)
        int_expt.as_file("%s/%s_integ%d.expt" % ( args.outdir,tag, i_f))
        int_refl.as_file("%s/%s_integ%d.refl" % ( args.outdir,tag, i_f))
        all_dfs.append(df)
        exp_ref_spec_lines.append("%s %s %s\n" % (new_expt_name, pred_file, df.spectrum_filename.values[0]))

    all_dfs = COMM.reduce(all_dfs)
    exp_ref_spec_lines = COMM.reduce(exp_ref_spec_lines)
    print0("\nReflections written to folder %s.\n" % args.outdir)
    if COMM.rank==0:
        hopper_input_name = os.path.abspath(os.path.join(args.outdir , "%s.txt" % args.hopInputName))
        o = open(hopper_input_name, "w")
        for l in exp_ref_spec_lines:
            o.write(l)
        o.close()
        all_dfs = pandas.concat(all_dfs)
        all_dfs.reset_index(inplace=True, drop=True)
        best_pkl_name = os.path.abspath(os.path.join(args.outdir , "%s.pkl" % args.hopInputName))
        all_dfs.to_pickle(best_pkl_name)
        print("Wrote %s (best_pickle option for simtbx.diffBragg.hopper) and %s (exp_ref_spec option for simtbx.diffBragg.hopper). Use them to run the predictions through hopper. Use the centroid=cal option to specify the predictions" % (best_pkl_name, hopper_input_name))
    npred_per_shot = COMM.reduce(npred_per_shot)
    all_wave = COMM.reduce(all_wave)

