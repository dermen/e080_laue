
#@profile
def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--mosSpread", type=float, default=0.025)
    parser.add_argument("--mosDoms", type=int, default=150)
    parser.add_argument("--div", type=float, default=0, help="divergence in mrad")
    parser.add_argument("--divSteps", type=int, default=0)
    parser.add_argument("--enSteps", type=int, default=322, help="Number of spectrum samples, use to upsample or downsample the specFile")
    parser.add_argument("--testShot", action="store_true", help="only simulate a single shots")
    parser.add_argument("--ndev", type=int, default=1, help="number of GPU devices per compute node")
    parser.add_argument("--run", type=int, default=1, help="run number in filename shot_R_#####.cbf")
    parser.add_argument("--mono", action='store_true', help="use the average wavelength to do mono simulation")
    parser.add_argument("--oversample", type=int, default=1, help="pixel oversample factor (increase if spots are sharp)")
    parser.add_argument("--numimg", type=int, default=180, help="number of images in 180 deg rotation")
    parser.add_argument("--noWaveImg", action="store_true", help="Dont write the wavelength-per-pixel image")
    parser.add_argument("--xtalSize", type=float, default=0.5, help="xtal size in mm")
    parser.add_argument("--gain", default=1, type=float, help="ADU per photon")
    args = parser.parse_args()

    from dials.array_family import flex
    import h5py
    import numpy as np
    from simtbx.nanoBragg import utils
    from libtbx.mpi4py import MPI
    from scipy.spatial.transform import Rotation
    COMM = MPI.COMM_WORLD
    from simtbx.diffBragg import utils as db_utils
    from simtbx.nanoBragg import nanoBragg
    from simtbx.modeling.forward_models import diffBragg_forward
    from dxtbx.model import DetectorFactory, BeamFactory, Crystal
    from scitbx.matrix import col, sqr
    import os
    import time
    import sys

    # convenience files from this repository
    spec_file = 'from_vukica.lam'  # intensity vs wavelength
    pdb_file = "7lvc.pdb" # structure
    air_name = 'air.stol'  # air sin theta over lambda
    total_flux=5e9
    beam_size_mm=0.01

    if COMM.rank==0:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        cmdfile = args.outdir+"/commandline_run%d.txt" % args.run
        with open(cmdfile, "w") as o:
            cmd = " ".join(sys.argv)
            o.write(cmd+"\n")
    COMM.barrier()

    # Rayonix model
    DETECTOR = DetectorFactory.simple(
        sensor='PAD',
        distance=200,  # mm
        beam_centre=(170, 170),  # mm
        fast_direction='+x',
        slow_direction='-y',
        pixel_size=(.08854, .08854),  # mm
        image_size=(3840, 3840))

    try:
        weights, energies = db_utils.load_spectra_file(spec_file)
    except:
        weights, energies = db_utils.load_spectra_file(spec_file, delim=" ")

    if args.enSteps is not None:
        from scipy.interpolate import interp1d
        wts_I = interp1d(energies, weights)# bounds_error=False, fill_value=0)
        energies = np.linspace(energies.min()+1e-6, energies.max()-1e-6, args.enSteps)
        weights = wts_I(energies)

    ave_en = np.mean(energies)
    ave_wave = utils.ENERGY_CONV / ave_en

    BEAM = BeamFactory.simple(ave_wave)

    Fcalc = db_utils.get_complex_fcalc_from_pdb(pdb_file, wavelength=ave_wave) #, k_sol=-0.8, b_sol=120) #, k_sol=0.8, b_sol=100)
    Famp = Fcalc.as_amplitude_array()

    water_bkgrnd = utils.sim_background(
        DETECTOR, BEAM, [ave_wave], [1], total_flux, pidx=0, beam_size_mm=beam_size_mm,
        Fbg_vs_stol=None, sample_thick_mm=2.5, density_gcm3=1, molecular_weight=18)

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
    fluxes = weights / weights.sum() * total_flux * len(weights)
    print("Simulating with %d energies" % num_en)
    print("Mean energy:", ave_wave)
    sg = Famp.space_group()
    print("unit cell, space group:\n", Famp, "\n")

    ucell = Famp.unit_cell()
    Breal = ucell.orthogonalization_matrix()
    # real space vectors
    a = Breal[0], Breal[3], Breal[6]
    b = Breal[1], Breal[4], Breal[7]
    c = Breal[2], Breal[5], Breal[8]
    CRYSTAL = Crystal(a, b, c, sg)

    randU = None
    if COMM.rank==0:
        randU = Rotation.random(random_state=0)
        randU = randU.as_matrix()
    randU = COMM.bcast(randU)
    CRYSTAL.set_U(randU.ravel())

    delta_phi = np.pi/ args.numimg

    gonio_axis = col((1,0,0))
    U0 = sqr(CRYSTAL.get_U())  # starting Umat

    Nabc = 100,100,100

    mos_spread = args.mosSpread
    num_mos = args.mosDoms
    device_Id = COMM.rank % args.ndev
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

        printout_pix=None
        from simtbx.diffBragg.device import DeviceWrapper
        with DeviceWrapper(device_Id) as _:

            out = diffBragg_forward(
                CRYSTAL, DETECTOR, BEAM, Famp, energies, fluxes,
                oversample=args.oversample, Ncells_abc=Nabc,
                mos_dom=num_mos, mos_spread=mos_spread, beamsize_mm=beam_size_mm,
                device_Id=device_Id,
                show_params=False, crystal_size_mm=args.xtalSize, printout_pix=printout_pix,
                verbose=COMM.rank==0, default_F=0, interpolate=0,
                mosaicity_random_seeds=None, div_mrad=args.div,
                divsteps=args.divSteps,
                show_timings=COMM.rank==0,
                nopolar=False, diffuse_params=None,
                perpixel_wavelen=not args.noWaveImg)

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

        SIM = nanoBragg(detector=DETECTOR, beam=BEAM)
        SIM.beamsize_mm = beam_size_mm
        SIM.exposure_s = 1
        SIM.flux = total_flux
        SIM.adc_offset_adu = 10
        SIM.detector_psf_kernel_radius_pixels = 5
        SIM.detector_calibration_noice_pct = 3
        SIM.detector_psf_fwhm_mm = 0.1
        SIM.quantum_gain = args.gain
        SIM.readout_noise_adu = 3
        SIM.raw_pixels += flex.double((img_with_bg).ravel())
        SIM.add_noise()
        cbf_name = os.path.join(args.outdir, "shot_%d_%05d.cbf" % (args.run, i_shot+1))
        SIM.to_cbf(cbf_name, cbf_int=True)
        #img = SIM.raw_pixels.as_numpy_array().reshape(img_sh)
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