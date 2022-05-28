# coding: utf-8
from dials.array_family import flex
import h5py
import numpy as np
from dxtbx.model import ExperimentList
from simtbx.nanoBragg import utils
from mpi4py import MPI
COMM = MPI.COMM_WORLD
from simtbx.diffBragg import utils as db_utils
from iotbx.reflection_file_reader import any_reflection_file
from simtbx.nanoBragg import nanoBragg
import socket
from simtbx.nanoBragg.sim_data import SimData
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
import os
import time


def diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                      oversample=0, Ncells_abc=(50, 50, 50),
                      mos_dom=1, mos_spread=0, beamsize_mm=0.001, device_Id=0,
                      show_params=True, crystal_size_mm=0.01, printout_pix=None,
                      verbose=0, default_F=0, interpolate=0, profile="gauss",
                      spot_scale_override=None,
                      mosaicity_random_seeds=None,
                      nopolar=False, diffuse_params=None, cuda=False):

    if cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"
    CRYSTAL, Famp = utils.ensure_p1(CRYSTAL, Famp)

    nbBeam = NBbeam()
    nbBeam.size_mm = beamsize_mm
    nbBeam.unit_s0 = BEAM.get_unit_s0()
    wavelengths = utils.ENERGY_CONV / np.array(energies)
    nbBeam.spectrum = list(zip(wavelengths, fluxes))

    nbCrystal = NBcrystal(init_defaults=False)
    nbCrystal.isotropic_ncells = False
    nbCrystal.dxtbx_crystal = CRYSTAL
    nbCrystal.miller_array = Famp
    nbCrystal.Ncells_abc = Ncells_abc
import sys

OUTDIR = sys.argv[1]
if COMM.rank==0:
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
COMM.barrier()


def diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                      oversample=0, Ncells_abc=(50, 50, 50),
                      mos_dom=1, mos_spread=0, beamsize_mm=0.001, device_Id=0,
                      show_params=True, crystal_size_mm=0.01, printout_pix=None,
                      verbose=0, default_F=0, interpolate=0, profile="gauss",
                      spot_scale_override=None,
                      mosaicity_random_seeds=None,
                      nopolar=False, diffuse_params=None, cuda=False):

    if cuda:
        os.environ["DIFFBRAGG_USE_CUDA"] = "1"
    CRYSTAL, Famp = utils.ensure_p1(CRYSTAL, Famp)

    nbBeam = NBbeam()
    nbBeam.size_mm = beamsize_mm
    nbBeam.unit_s0 = BEAM.get_unit_s0()
    wavelengths = utils.ENERGY_CONV / np.array(energies)
    nbBeam.spectrum = list(zip(wavelengths, fluxes))

    nbCrystal = NBcrystal(init_defaults=False)
    nbCrystal.isotropic_ncells = False
    nbCrystal.dxtbx_crystal = CRYSTAL
    nbCrystal.miller_array = Famp
    nbCrystal.Ncells_abc = Ncells_abc
    nbCrystal.symbol = CRYSTAL.get_space_group().info().type().lookup_symbol()
    nbCrystal.thick_mm = crystal_size_mm
    nbCrystal.xtal_shape = profile
    nbCrystal.n_mos_domains = mos_dom
    nbCrystal.mos_spread_deg = mos_spread

    S = SimData()
    S.detector = DETECTOR
    npan = len(DETECTOR)
    nfast, nslow = DETECTOR[0].get_image_size()
    img_shape = npan, nslow, nfast
    S.beam = nbBeam
    S.crystal = nbCrystal
    if mosaicity_random_seeds is not None:
        S.mosaic_seeds = mosaicity_random_seeds

    S.instantiate_diffBragg(verbose=verbose, oversample=oversample, interpolate=interpolate, device_Id=device_Id,
                            default_F=default_F)

    if spot_scale_override is not None:
        S.update_nanoBragg_instance("spot_scale", spot_scale_override)
    S.update_nanoBragg_instance("nopolar", nopolar)

    if show_params:
        S.D.show_params()

    S.D.verbose = 2
    S.D.store_ave_wavelength_image = True
    S.D.record_time = True
    if diffuse_params is not None:
        S.D.use_diffuse = True
        S.D.gamma_miller_units = diffuse_params["gamma_miller_units"]
        S.D.diffuse_gamma = diffuse_params["gamma"]
        S.D.diffuse_sigma = diffuse_params["sigma"]
    S.D.add_diffBragg_spots_full()
    S.D.show_timings()
    t = time.time()
    data = S.D.raw_pixels_roi.as_numpy_array().reshape(img_shape)
    wavelen_data = S.D.ave_wavelength_image().as_numpy_array().reshape(img_shape)
    t = time.time() - t
    print("Took %f sec to recast and reshape" % t)
    if printout_pix is not None:
        S.D.raw_pixels_roi*=0
        p,f,s = printout_pix
        S.D.printout_pixel_fastslow = f,s
        S.D.show_params()
        S.D.add_diffBragg_spots(printout_pix)

    # free up memory
    S.D.free_all()
    S.D.free_Fhkl2()
    if S.D.gpu_free is not None:
        S.D.gpu_free()
    return data, wavelen_data


num_shots = 1000000

# gather all hostnames and create sub-communicators for all processes on a given host
HOST = socket.gethostname()
unique_hosts = COMM.gather(HOST)
HOST_MAP = None
if COMM.rank == 0:
    HOST_MAP = {HOST: i for i, HOST in enumerate(set(unique_hosts))}
HOST_MAP = COMM.bcast(HOST_MAP)
HOST_COMM = COMM.Split(color=HOST_MAP[HOST])
NUM_HOSTS = len(HOST_MAP)

El = ExperimentList.from_file("/global/cfs/cdirs/m3992/dermen/ultra_refined000.expt", False)
R = flex.reflection_table.from_file("/global/cfs/cdirs/m3992/dermen/ultra_refined000.refl")


DETECTOR = El.detectors()[0]
DETECTOR = db_utils.strip_thickness_from_detector(DETECTOR)

weights, energies = db_utils.load_spectra_file('/global/cfs/cdirs/m3992/dermen/e080_2.lam')
stride = 1
weights = weights[::stride]
energies = energies[::stride]
ave_en = np.mean(energies)
ave_wave = utils.ENERGY_CONV / ave_en
BEAM = El.beams()[0]
BEAM.set_wavelength(ave_wave)


Famp = any_reflection_file("/global/cfs/cdirs/m3992/dermen/7lvc.pdb.mtz").as_miller_arrays()[0].as_amplitude_array()
total_flux=1e12
#water_bkgrnd = utils.sim_background(
#    DETECTOR, BEAM, [waves[0]], [1], total_flux, pidx=0, beam_size_mm=0.01,
#    Fbg_vs_stol=None, sample_thick_mm=50, density_gcm3=1, molecular_weight=18)
#np.save("water", water_bkgrnd.as_numpy_array())
img_sh = 3840, 3840

num_en = len(energies)
fluxes = weights / weights.sum() * total_flux

rank_inds = np.array_split(np.arange(num_en), HOST_COMM.size)[HOST_COMM.rank]
energies_rank = energies[rank_inds]
fluxes_rank = fluxes[rank_inds]

CRYSTAL = El.crystals()[0]

water_bkgrnd = None
if HOST_COMM.rank==0:
    #water_bkgrnd = flex.double(np.load("water.npy").reshape(img_sh))
    water_bkgrnd = np.load("/global/cfs/cdirs/m3992/dermen/water.npy").reshape(img_sh)
    #bg = [water_bkgrnd]

for i_shot in range(num_shots):
    if i_shot % NUM_HOSTS != HOST_MAP[HOST]:
        continue
    if HOST_COMM.rank==0:
        from scipy.spatial.transform import Rotation
        randU = Rotation.random()
        randU = randU.as_matrix()
        CRYSTAL.set_U(randU.ravel())
    CRYSTAL = HOST_COMM.bcast(CRYSTAL)
    #print("rank%d will simulate %d energies" %(COMM.rank, len(energies_rank)))

    img, wave_img = diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies_rank, fluxes_rank,
                      oversample=1, Ncells_abc=(120,120,30),
                      mos_dom=1, mos_spread=0, beamsize_mm=0.01, device_Id=0,
                      show_params=False, crystal_size_mm=0.01, printout_pix=None,
                      verbose=0, default_F=0, interpolate=0, profile="gauss",
                      spot_scale_override=6e9,
                      mosaicity_random_seeds=None,
                      nopolar=False, diffuse_params=None, cuda=False)
    if len(img.shape)==3:
        img = img[0]
        wave_img = wave_img[0]

    #model = utils.flexBeam_sim_colors(
    #    CRYSTAL, DETECTOR, BEAM, Famp, energies_rank, fluxes_rank,
    #    pids=None, cuda=False, oversample=2, Ncells_abc=(120, 120, 30),
    #    mos_dom=1, mos_spread=0, beamsize_mm=0.01, device_Id=0, omp=False,
    #    show_params=False, crystal_size_mm=0.1, printout_pix=None, time_panels=False,
    #    verbose=0, default_F=0, interpolate=0, recenter=True, profile="gauss",
    #    spot_scale_override=5e7, background_raw_pixels=bg, include_noise=False,
    #    add_water=False, add_air=False, water_path_mm=0.1, air_path_mm=50, rois_perpanel=None,
    #    adc_offset=0, readout_noise=3, psf_fwhm=20, gain=1, mosaicity_random_seeds=None, nopolar=False)
    #img = model[0][1]
    img = HOST_COMM.reduce(img)

    if HOST_COMM.rank==0:
        img_with_bg = img +water_bkgrnd
        SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
        SIM.beamsize_mm = 0.01
        SIM.exposure_s = 1
        SIM.flux = total_flux
        SIM.adc_offset_adu =0
        SIM.detector_psf_kernel_radius_pixels = 5
        SIM.detector_psf_fwhm_mm = .1
        SIM.quantum_gain = 0.7
        SIM.readout_noise_adu = 3
        SIM.raw_pixels = flex.double((img_with_bg).ravel())
        SIM.add_noise()
        img = SIM.raw_pixels.as_numpy_array().reshape(img_sh)
        SIM.free_all()
        del SIM
        houtfile = os.path.join(OUTDIR, "shot%d.h5" % i_shot)
        h = h5py.File(houtfile, "w")
        h.create_dataset("data", data=img, dtype=np.float32, compression="lzf")
        h.create_dataset("wave_data", data=wave_img, dtype=np.float32, compression="lzf")
        h.close()
