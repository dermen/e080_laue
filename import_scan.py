from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--dirname", type=str, default='.')
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--totalDeg", type=float, default=180)
parser.add_argument("--numimg", type=int, default=180) 
parser.add_argument("--outfile", type=str, default="imported_scan.expt")
args = parser.parse_args()

from dxtbx.imageset import ImageSetFactory
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.model import Goniometer, ScanFactory

delta_phi = args.totalDeg / args.numimg
print("Delta phi= %f deg." % delta_phi)
scan = ScanFactory.make_scan([1,args.numimg], 1, [0,delta_phi], epochs=list(range(args.numimg)))
G = Goniometer()
iseq = ImageSetFactory.make_sequence(
    "%s/shot_%d_00###.cbf" % (args.dirname, args.run), indices=list(range(1,args.numimg+1)), 
    goniometer=G, scan=scan)
El = ExperimentListFactory.from_sequence_and_crystal(iseq, None)
El.as_file(args.outfile)
print("Done.")
