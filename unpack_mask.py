import h5py
from simtbx.diffBragg import utils
mask = h5py.File("mask.h5", "r")["mask"][()]
utils.save_numpy_mask_as_flex(mask, "newbad.pkl")

