import sys
import os
import re

o = open(sys.argv[1], 'r').readlines()
newdir = "/n/hekstra_lab/data/201903_APS_BioCARS/e080/"

new_l = []
for l in o:
    if "mccd" in l:
        s = re.search('".*mccd"', l)
        path = l[s.start() + 1 : s.end() - 1]
        new_path = os.path.join(newdir, os.path.basename(path))
        assert os.path.exists(new_path)
        l = l[: s.start() + 1] + new_path + '"\n'
    new_l.append(l)

o = open(sys.argv[2], "w")
o.writelines(new_l)
o.close()
print("Wrote %s" % sys.argv[2])
