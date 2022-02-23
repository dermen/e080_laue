import numpy as np
from scipy import constants
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt

def ev_per_pix(energy, d_res=2, pixsize=0.08854, detdist=200.46):
#   energy shift is 1eV
    wave1 = ENERGY_CONV/energy
    wave2 = ENERGY_CONV/(energy+1)

    theta1 = np.arcsin(wave1/2/d_res) 
    theta2 = np.arcsin(wave2/2/d_res) 

    R1 = detdist/pixsize * np.tan(2*theta1)
    R2 = detdist/pixsize * np.tan(2*theta2)

#   shift in 1eV corresponds to this many pixels:
    pix_per_eV= abs(R1-R2)

#   1 pixel is then worth 
    eV_per_pix = 1./ pix_per_eV 

    print("\teV spread per pixel is %.4f eV at %.1f Angstrom resolution\n" %  (eV_per_pix, d_res) )

nominal_wave = 1.04
nominal_en = ENERGY_CONV / nominal_wave
for d in [4,3,2.5,2,1.5,1.25]:
    ev_per_pix(nominal_en, d)

