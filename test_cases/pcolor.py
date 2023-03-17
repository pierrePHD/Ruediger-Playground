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


def test_pcolor():
    import calcGrid

    np.random.seed(0)
    coords = np.random.random((100, 3))
    quant = np.random.random(100)
    box = [0.5, 0.5, 0.5]
    centerx = 0.5
    centery = 0.5
    centerz = 0.5

    ret_dict = calcGrid.calcASlice(coords, quant, nx=500, ny=500, nz=0, boxx=box[0], boxy=box[1], boxz=box[2],
                                   centerx=centerx, centery=centery, centerz=centerz, grid3D=False)

    for key in ret_dict:
        np.savetxt(f"./pcolor_{key}.txt", ret_dict[key], delimiter=",", newline=",")


if __name__ == "__main__":

    test_pcolor()

