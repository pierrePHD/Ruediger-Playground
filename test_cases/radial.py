import os
import re

from createICs import create_particles_healpix
from ic import create_wd
import numpy as np
import pandas as pd
from pyhelm_eos import loadhelm_eos

from const import msol
import decays
from gadget import gadget_readsnap
import get_path_hack
import loaders
from loaders import load_species
from main_utils import SuppressStdout, wdCOgetRhoCFromMassExact, wdCOgetMassFromRhoCExact

# from pylab import *
ppath = os.path.dirname(get_path_hack.__file__)

def test_radial():
    import calcGrid

    np.random.seed(0)
    coords = np.random.random((100, 3))
    quant = np.random.random(100)

    ret_arr = calcGrid.calcRadialProfile(coords.astype('float64'), quant.astype('float64'), 0, 200, 0, 0.5, 0.5, 0.5)
    print(ret_arr)
    np.savetxt(f"./radial.txt", ret_arr, delimiter=",", newline=",")


if __name__ == "__main__":

    test_radial()
