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
from simtbx.modeling.forward_models import diffBragg_forward
from simtbx.nanoBragg.sim_data import SimData
from simtbx.nanoBragg.nanoBragg_crystal import NBcrystal
from simtbx.nanoBragg.nanoBragg_beam import NBbeam
from scitbx.matrix import col, sqr
import os
import time
import sys

OUTDIR = sys.argv[1]
if COMM.rank==0:
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
COMM.barrier()

num_shots = 180

El = ExperimentList.from_file("/global/cfs/cdirs/m3992/dermen/ultra_refined000.expt", False)
#R = flex.reflection_table.from_file("/global/cfs/cdirs/m3992/dermen/ultra_refined000.refl")

DETECTOR = El.detectors()[0]
DETECTOR = db_utils.strip_thickness_from_detector(DETECTOR)

weights, energies = db_utils.load_spectra_file('/global/cfs/cdirs/m3992/dermen/e080_2.lam')
stride = 1
weights = weights[::stride]
energies = energies[::stride]
ave_en = np.mean(energies)
shift = 11808 - ave_en
energies += shift

ave_en = np.mean(energies)
ave_wave = utils.ENERGY_CONV / ave_en
ave_en = np.mean(energies)
ave_wave = utils.ENERGY_CONV / ave_en

BEAM = El.beams()[0]
BEAM.set_wavelength(ave_wave)

#Famp = any_reflection_file("/global/cfs/cdirs/m3992/dermen/7lvc.pdb.mtz").as_miller_arrays()[0].as_amplitude_array()

pdb_file = "/global/cfs/cdirs/m3992/dermen/e080_laue/7lvc.pdb"
Fcalc = db_utils.get_complex_fcalc_from_pdb(pdb_file, wavelength=ave_wave) #, k_sol=-0.8, b_sol=120) #, k_sol=0.8, b_sol=100)
Famp = Fcalc.as_amplitude_array()

total_flux=5e9
beam_size_mm=0.01
water_bkgrnd = utils.sim_background(
    DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
    Fbg_vs_stol=None, sample_thick_mm=2.5, density_gcm3=1, molecular_weight=18)

air_name = '/pscratch/sd/d/dermen/diffraction_ai_sims_data/air.stol'
air_Fbg, air_stol = np.loadtxt(air_name).T
air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
air = utils.sim_background(DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
                        molecular_weight=14,
                           sample_thick_mm=5,
                           Fbg_vs_stol=air_stol, density_gcm3=1.2e-3) 

#np.save("water", water_bkgrnd.as_numpy_array())
fdim, sdim = DETECTOR[0].get_image_size()
img_sh = sdim, fdim
water_bkgrnd = water_bkgrnd.as_numpy_array().reshape(img_sh)
air = air.as_numpy_array().reshape(img_sh)

num_en = len(energies)
fluxes = weights / weights.sum() * total_flux
print("Simulating with %d energies" % num_en)
print("Mean energy:", ave_wave)

CRYSTAL = El.crystals()[0]

#    #water_bkgrnd = flex.double(np.load("water.npy").reshape(img_sh))
#    water_bkgrnd = np.load("/global/cfs/cdirs/m3992/dermen/water.npy").reshape(img_sh)
#    #bg = [water_bkgrnd]

from scipy.spatial.transform import Rotation

randU = None
if COMM.rank==0:
    randU = Rotation.random(random_state=0)
    randU = randU.as_matrix()
randU = COMM.bcast(randU)
CRYSTAL.set_U(randU.ravel())

delta_phi =  np.pi / 180 # 1 degree

gonio_axis = col((1,0,0))
U0 = sqr(CRYSTAL.get_U())  # starting Umat

mos_spread = float(sys.argv[2])
num_mos = 150
device_Id = COMM.rank % 4

for i_shot in range(num_shots):
    if i_shot % COMM.size != COMM.rank:
        continue

    
    print("Doing shot %d" % i_shot)
    Rphi = gonio_axis.axis_and_angle_as_r3_rotation_matrix(delta_phi*i_shot, deg=False)
    Uphi = Rphi * U0
    CRYSTAL.set_U(Uphi)

    t = time.time()
    img, wave_img = diffBragg_forward(CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                      oversample=1, Ncells_abc=(100,100,100),
                      mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm, 
                      device_Id=device_Id,
                      show_params=COMM.rank==0, crystal_size_mm=10, printout_pix=None,
                      verbose=0, default_F=0, interpolate=0, profile="square",
                      mosaicity_random_seeds=None, div_mrad=0.02,
                      nopolar=False, diffuse_params=None, cuda=True, perpixel_wavelen=True)
    t = time.time()-t
    print("Took %.4f sec to sim" % t)
    if len(img.shape)==3:
        img = img[0]
        wave_img = wave_img[0]

    img_with_bg = img +water_bkgrnd + air
        
    SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
    SIM.beamsize_mm = beam_size_mm 
    SIM.exposure_s = 1
    SIM.flux = total_flux
    SIM.adc_offset_adu = 10
    SIM.detector_psf_kernel_radius_pixels = 5
    SIM.detector_calibration_noice_pct = 3
    SIM.detector_psf_fwhm_mm = .1
    SIM.quantum_gain = 0.7
    SIM.readout_noise_adu = 3
    SIM.raw_pixels += flex.double((img_with_bg).ravel())
    SIM.add_noise()
    cbf_name = os.path.join(OUTDIR, "shot_1_%05d.cbf" % (i_shot+1))
    SIM.to_cbf(cbf_name, cbf_int=True)
    img = SIM.raw_pixels.as_numpy_array().reshape(img_sh)
    SIM.free_all()
    del SIM
    h5_name = cbf_name.replace(".cbf", ".h5")
    h = h5py.File(h5_name, "w")
    h.create_dataset("wave_data", data=wave_img, dtype=np.float32, compression="lzf")
    h.create_dataset("delta_phi", data=delta_phi)
    h.create_dataset("Umat", data=CRYSTAL.get_U())
    h.create_dataset("Bmat", data=CRYSTAL.get_B())
    h.create_dataset("mos_spread", data=mos_spread)
    h.close()
