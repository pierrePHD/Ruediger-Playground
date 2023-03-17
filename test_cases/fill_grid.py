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


def test_fill_grid():
    import createICs

    np.random.seed(0)
    coords = np.random.random((100, 3))

    boxsize = 1
    res = 32
    p = createICs.create_particles_fill_grid(coords, boxsize, res)
    np.savetxt(f"./fill_grid.txt", p, delimiter=",", newline=",")

if __name__ == "__main__":

    test_fill_grid()

