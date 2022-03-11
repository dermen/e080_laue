import sys
import os
import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("orig", type=str, help="expt file from ricks tarball")
parser.add_argument("new", type=str, help="new expt file to write with updated paths")
parser.add_argument("newdir", type=str, help="root folder for the e080 mccds on current system")
args = parser.parse_args()

o = open(args.orig, 'r').readlines()

new_l = []
for l in o:
    if "mccd" in l:
        s = re.search('".*mccd"', l)
        path = l[s.start() + 1 : s.end() - 1]
        new_path = os.path.join(args.newdir, os.path.basename(path))
        assert os.path.exists(new_path)
        l = l[: s.start() + 1] + new_path + '"\n'
    new_l.append(l)

o = open(args.new, "w")
o.writelines(new_l)
o.close()
print("Wrote %s" % sys.argv[2])
