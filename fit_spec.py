import numpy as np
from simtbx.diffBragg import utils
from dials.array_family import flex
from scipy.interpolate import interp1d

def smooth(x, beta=10.0, window_size=11):
    """
    https://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html
    """
    if window_size % 2 == 0:
        window_size += 1
    s = np.r_[x[window_size - 1:0:-1], x, x[-1:-window_size:-1]]
    w = np.kaiser(window_size, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')
    b = int((window_size - 1) / 2)
    smoothed = y[b:len(y) - b]
    return smoothed



res = 1/utils.ENERGY_CONV   # 1 eV 


###
R = flex.reflection_table.from_file("optimized.refl")
wave = R["Wavelength"].as_numpy_array()
wave_range = wave.max()-wave.min()

nbin = int(wave_range / res)
print("Using %d bins for 1eV res" % nbin)
bins = np.linspace(wave.min() - 1e-6, wave.max() + 1e-6, nbin+1)
wt = np.histogram(wave, bins)[0]
wavelen = (bins[:-1] + bins[1:]) * 0.5
wt = smooth(wt, window_size=201)

for stride in [1,2,3,4]:
    name ="spec_%deV.lam" % stride
    x = utils.ENERGY_CONV/wavelen[::stride]
    y = wt[::stride]
    utils.save_spectra_file(name, x,y)
    print("Wrote %s (%d channels)" % (name, len(x)))
from pylab import *
figure()
plot( x,y)
legend()
xlabel("eV")
ylabel("intensity (a.u.)")
show()
###
# below is a script to fit sum of gaussian to spectra
#
#
#import lmfit
#params = lmfit.Parameters()
#
#xvals = utils.ENERGY_CONV / wavelen
#
#yvals_orig = wt / wt.max()
#yvals = smooth(wt, window_size=201)
#yvals /= yvals.max()
#
#peak = xvals[np.argmax(yvals)]
#
#
#num_gauss = 15
#
#mu_bins = np.linspace(xvals.min()+1e-6, xvals.max()-1e-6, num_gauss+1)
#mus = (mu_bins[:-1] + mu_bins[1:])*.5
#
#all_params = []
#for i in range(num_gauss):
#    mu = mus[i]
#    sigma_param = lmfit.Parameter(name="sig%d" % i, value=30, min=0)
#    amp_init = interp1d(xvals, yvals, bounds_error=False, fill_value=0)(mu)
#    amp_param = lmfit.Parameter(name="amp%d" % i, value=amp_init,min=0)
#    mu_param = lmfit.Parameter(name="mu%d" % i, value=mu, vary=False)
#    all_params += [sigma_param, amp_param, mu_param]
#
#P = lmfit.Parameters()
#P.add_many(*all_params)
#
#
#def multiG(params, xvals, yvals, return_model=False, falloff=0):
#
#
#    num_gauss = int(len(params)/ 3)
#
#    model = np.zeros_like(yvals)
#    for i in range(num_gauss):
#        mu = params["mu%d" % i].value
#        distance_to_peak = abs(peak - mu)
#        frac = 1 / (1 + falloff*distance_to_peak)
#        amp = frac*params["amp%d" %i].value
#        sig = params["sig%d" % i].value
#
#        xdiff = xvals-mu
#        exp_arg = xdiff**2/ sig**2
#        G = amp*np.exp(-exp_arg)
#        model += G
#
#    if return_model:
#        return model
#    else:
#        yresid = yvals - model
#        return np.sum(yresid**2)
#
#
#out = lmfit.minimize(multiG, P, args=(xvals, yvals, False), method="Nelder-Mead")
#
#scales = 0,.001,.01,.1
#models = [multiG(out.params, xvals, yvals, True, falloff=s) for s in scales]
#from pylab import *
#plot( xvals, yvals_orig, 'x', label="wave")
#plot( xvals, yvals, '--', label="smooth wave")
#for i,m in enumerate(models):
#    m /= m.max()
#    spec_name = "model_%d.lam" % i
#    utils.save_spectra_file(spec_name, wavelen, m)
#    plot(xvals, m, label=spec_name)
#xlabel("eV")
#ylabel("intensity (a.u.)")
#legend()
#show()
