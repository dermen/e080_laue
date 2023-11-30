# Imports
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Central peak wavelength for Gaussian spectrum
central_lam = 1.050 # Angstroms

# BW -> std.dev. conversion
bandwidths = np.asarray([0.005, 0.01, 0.022, 0.05]) # percentage values as decimals
fwhms = bandwidths*central_lam
sigmas = fwhms / 2.355

# Wavelengths to get intensities for
wavs = np.linspace(0.9, 1.2, 201) # 0.8-1.4 Angstroms

# List to store intensity arrays
intens = []

# Simulate and plot intensities to check
for i in range(len(sigmas)):
    spectrum = scipy.stats.norm.pdf(wavs, loc=central_lam, scale=sigmas[i])
    norm_spectrum = spectrum / sum(spectrum)
    intens.append(norm_spectrum)
    plt.plot(wavs, intens[i])
    plt.show()

# Write data to files
filenames = ["half_percent.txt","one_percent.txt","two_dot_two_percent.txt","five_percent.txt"]

for i in range(len(filenames)):
    table = np.vstack((wavs, intens[i])).T
    np.savetxt(filenames[i], table, fmt='%8.6f %8.6f')
