from pylab import *
from dials.array_family import flex
import sys
import glob
import numpy as np
style.use("ggplot")
for glob_s in sys.argv[1:]:
    fnames = glob.glob(glob_s)
    all_d = []
    all_shotd = []
    nref_per_shot = []
    for f in fnames:
        R = flex.reflection_table.from_file(f)
        x,y,_=R['xyzobs.px.value'].parts()
        x2,y2,_=R['xyzcal.px'].parts()
        import numpy as np
        d = np.sqrt( (x-x2)**2 + (y-y2)**2)
        print('diffBragg: %.3f (%s)' % (np.median(d), f ) )
        #try:
        #    x3,y3,_=R['dials.xyzcal.px'].parts()
        #    d2 = np.sqrt( (x-x3)**2 + (y-y3)**2)
        #    print('dials: %.3f' %np.median(d2))
        #except:pass
        all_d.append(d)
        all_shotd.append( np.median(d))
        nref_per_shot .append( len(d))

    all_d = hstack(all_d)
    print("median over %d shots=%f pixels (%d refls)" % (len(fnames), median(all_d), len(all_d)))
    print("Min refls per shot=%d, max refls per shot = %d, ave refls per shot=%.1f" % (min(nref_per_shot), max(nref_per_shot), mean(nref_per_shot)))
    subplot(121)
    hist( all_d, bins=200, histtype='step', lw=2, label=glob_s)
    xlabel("|calc-obs| (pixels)")
    ylabel("num refls")
    legend(prop={'size':7})
    subplot(122)
    hist( all_shotd, bins=40, histtype='step', lw=2, label=glob_s)
    gca().set_yscale("log")
    xlabel("median |calc-obs| (pixels)")
    ylabel("num shots", labelpad=0)
    legend(prop={'size':7})
    gca().tick_params(direction='in', which='both')

gcf().set_size_inches((7,4))
suptitle(" %d shots ; %d refls" % (len(fnames), len(all_d)))
subplots_adjust(bottom=0.2, right=0.97, top=0.92, left=0.1)

show()

