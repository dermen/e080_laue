import h5py
from simtbx.diffBragg import utils
print("loading mask")
mask = h5py.File("mask.h5", "r")["mask"][()]
print("saving as newbad.pkl")
utils.save_numpy_mask_as_flex(mask, "newbad.pkl")

