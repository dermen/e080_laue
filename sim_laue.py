from argparse import ArgumentParser

#@profile
def main():

    parser = ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("--specFile", type=str, default=None)
    parser.add_argument("--mosSpread", type=float, default=0.025)
    parser.add_argument("--mosDoms", type=int, default=150)
    parser.add_argument("--div", type=float, default=0, help="divergence in mrad")
    parser.add_argument("--divSteps", type=int, default=0)
    parser.add_argument("--enSteps", type=int, default=None)
    parser.add_argument("--testShot", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--mono", action='store_true')
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--numimg", type=int, default=180)
    parser.add_argument("--noWaveImg", action="store_true")
    parser.add_argument("--xtalShape", type=str, default="gauss")
    parser.add_argument("--sphereMos", action="store_true")
    parser.add_argument("--xtalSize", type=float, default=0.5, help="xtal size in mm")
    parser.add_argument("--flatFamps", action="store_true")
    parser.add_argument("--pyNoise", action="store_true")
    parser.add_argument("--gain", default=0.7, type=float)
    parser.add_argument("--deltaPhi", default=None, type=float)
    parser.add_argument("--phiSteps", default=1, type=int)
    args = parser.parse_args()

    from dials.array_family import flex
    import h5py
    import numpy as np
    from dxtbx.model import ExperimentList
    from simtbx.nanoBragg import utils
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    from simtbx.diffBragg import utils as db_utils
    from simtbx.nanoBragg import nanoBragg
    from simtbx.modeling.forward_models import diffBragg_forward
    from scitbx.matrix import col, sqr
    from cctbx import miller
    import os
    import time
    import sys

    if COMM.rank==0:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        cmdfile = args.outdir+"/commandline_run%d.txt" % args.run
        with open(cmdfile, "w") as o:
            cmd = " ".join(sys.argv)
            o.write(cmd+"\n")
    COMM.barrier()


    El = ExperimentList.from_file("ultra_refined000.expt", False)

    DETECTOR = El.detectors()[0]
    DETECTOR = db_utils.strip_thickness_from_detector(DETECTOR)

    spec_file = args.specFile
    if spec_file is None:
        spec_file = 'from_vukica.lam'
        spec_file = 'e080_2.lam'

    try:
        weights, energies = db_utils.load_spectra_file(spec_file)
    except:
        weights, energies = db_utils.load_spectra_file(spec_file, delim=" ")

    if args.enSteps is not None:
        from scipy.interpolate import interp1d
        wts_I = interp1d(energies, weights)# bounds_error=False, fill_value=0)
        energies = np.linspace(energies.min()+1e-6, energies.max()-1e-6, args.enSteps)
        weights = wts_I(energies)

    weights = weights[::args.stride]
    energies = energies[::args.stride]
    ave_en = np.mean(energies)
    #shift = 11808 - ave_en
    #energies += shift

    ave_en = np.mean(energies)
    ave_wave = utils.ENERGY_CONV / ave_en
    ave_en = np.mean(energies)
    ave_wave = utils.ENERGY_CONV / ave_en

    BEAM = El.beams()[0]
    BEAM.set_wavelength(ave_wave)

    pdb_file = "7lvc.pdb"
    Fcalc = db_utils.get_complex_fcalc_from_pdb(pdb_file, wavelength=ave_wave) #, k_sol=-0.8, b_sol=120) #, k_sol=0.8, b_sol=100)
    Famp = Fcalc.as_amplitude_array()
    if args.flatFamps:
        ave_F = np.mean(Famp.data())
        flat_data = flex.double(len(Famp.data()), ave_F)
        Famp = miller.array(Famp.set(), flat_data)

    total_flux=5e9
    beam_size_mm=0.01
    water_bkgrnd = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        Fbg_vs_stol=None, sample_thick_mm=2.5, density_gcm3=1, molecular_weight=18)

    air_name = 'air.stol'
    air_Fbg, air_stol = np.loadtxt(air_name).T
    air_stol = flex.vec2_double(list(zip(air_Fbg, air_stol)))
    air = utils.sim_background(DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
                            molecular_weight=14,
                            sample_thick_mm=5,
                            Fbg_vs_stol=air_stol, density_gcm3=1.2e-3)

    fdim, sdim = DETECTOR[0].get_image_size()
    img_sh = sdim, fdim
    water_bkgrnd = water_bkgrnd.as_numpy_array().reshape(img_sh)
    air = air.as_numpy_array().reshape(img_sh)

    if args.mono:
        energies = np.array([ave_en])
        weights = np.array([1])

    num_en = len(energies)
    fluxes = weights / weights.sum() * total_flux
    print("Simulating with %d energies" % num_en)
    print("Mean energy:", ave_wave)

    CRYSTAL = El.crystals()[0]

    from scipy.spatial.transform import Rotation

    randU = None
    if COMM.rank==0:
        randU = Rotation.random(random_state=0)
        randU = randU.as_matrix()
    randU = COMM.bcast(randU)
    CRYSTAL.set_U(randU.ravel())

    delta_phi =  np.pi / args.numimg

    gonio_axis = col((1,0,0))
    U0 = sqr(CRYSTAL.get_U())  # starting Umat

    Nabc = 100,100,100
    if args.sphereMos:
        a,b,c,_,_,_ = CRYSTAL.get_unit_cell().parameters()
        Na = 30
        Nb = 30*a/b
        Nc = 30*a/c
        Nabc = Na, Nb, Nc

    mos_spread = args.mosSpread
    num_mos = args.mosDoms
    device_Id = COMM.rank % args.ndev
    from resonet.sims import make_sims
    STOL = make_sims.get_theta_map(DETECTOR, BEAM)
    reso, Bfac_img = make_sims.get_Bfac_img(STOL)
    tsims = []

    for i_shot in range(args.numimg):

        if i_shot % COMM.size != COMM.rank:
            continue
        tsim = time.time()
        print("Doing shot %d/%d" % (i_shot+1, args.numimg))
        Rphi = gonio_axis.axis_and_angle_as_r3_rotation_matrix(delta_phi*i_shot, deg=False)
        Uphi = Rphi * U0
        CRYSTAL.set_U(Uphi)

        t = time.time()

        #xtal_size_mm=10 # this was used for laueSims to account for small flux?

        printout_pix=0,2557,2573
        printout_pix=None
        out = diffBragg_forward(
            CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
            oversample=args.oversample, Ncells_abc=Nabc,
            mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm,
            device_Id=device_Id,
            show_params=False, crystal_size_mm=args.xtalSize, printout_pix=printout_pix,
            verbose=0, default_F=0, interpolate=0, profile=args.xtalShape,
            mosaicity_random_seeds=None, div_mrad=args.div,
            divsteps=args.divSteps,
            nopolar=False, diffuse_params=None,
            cuda=True, perpixel_wavelen=not args.noWaveImg,
            delta_phi=args.deltaPhi, num_phi_steps=args.phiSteps, spindle_axis=gonio_axis)

        if args.noWaveImg:
            img = out
            wave_img = h_img = k_img = l_img = None
        else:
            img, wave_img, h_img, k_img, l_img = out

        t = time.time()-t
        print("Took %.4f sec to sim" % t)
        if len(img.shape)==3:
            img = img[0]
            if wave_img is not None:
                wave_img = wave_img[0]
                h_img = h_img[0]
                k_img = k_img[0]
                l_img = l_img[0]

        img_with_bg = img +water_bkgrnd + air
        if args.pyNoise:
            img_with_bg = np.random.poisson(img_with_bg)

        SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
        SIM.beamsize_mm = beam_size_mm
        SIM.exposure_s = 1
        SIM.flux = total_flux
        SIM.adc_offset_adu = 10
        SIM.detector_psf_kernel_radius_pixels = 5
        SIM.detector_calibration_noice_pct = 3
        SIM.detector_psf_fwhm_mm = .1
        SIM.quantum_gain = args.gain
        SIM.readout_noise_adu = 3
        SIM.raw_pixels += flex.double((img_with_bg).ravel())
        if not args.pyNoise:
            SIM.add_noise()
        cbf_name = os.path.join(args.outdir, "shot_%d_%05d.cbf" % (args.run, i_shot+1))
        SIM.to_cbf(cbf_name, cbf_int=True)
        img = SIM.raw_pixels.as_numpy_array().reshape(img_sh)
        SIM.free_all()
        del SIM
        h5_name = cbf_name.replace(".cbf", ".h5")
        h = h5py.File(h5_name, "w")
        if wave_img is not None:
            h.create_dataset("wave_data", data=wave_img, dtype=np.float32, compression="lzf")
            h.create_dataset("h_data", data=h_img, dtype=np.float32, compression="lzf")
            h.create_dataset("k_data", data=k_img, dtype=np.float32, compression="lzf")
            h.create_dataset("l_data", data=l_img, dtype=np.float32, compression="lzf")
        h.create_dataset("delta_phi", data=delta_phi)
        h.create_dataset("Umat", data=CRYSTAL.get_U())
        h.create_dataset("Bmat", data=CRYSTAL.get_B())
        h.create_dataset("mos_spread", data=mos_spread)
        #h.create_dataset("img_with_bg", data=img_with_bg)
        h.close()
        tsim = time.time()-tsim
        if COMM.rank==0:
            print("TSIM=%f" % tsim)
        tsims.append(tsim)
        if args.testShot:
            break

    tsims = COMM.reduce(tsims)
    if COMM.rank==0:
        ave_tsim = np.median(tsims)
        print("Done", flush=True)
        print("Ave time to sim a shot=%f sec" % ave_tsim)

if __name__=="__main__":
    main()