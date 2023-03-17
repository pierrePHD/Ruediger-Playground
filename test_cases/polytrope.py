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


def test_polytrope():
    import ic

    species_file = ppath + "/eostable/species05.txt"
    eos = loadhelm_eos(ppath + "/eostable/helm_table.dat", species_file, True)
    xnuc = np.array([0.0, 0.3, 0.7, 0.0, 0.0])
    ret_dict = ic.create_polytrope(eos, 3.0, 5e6, xnuc, pres0=2.97e23, temp0=0.0, dr=1e5)
    print(ret_dict["rho"].shape)
    np.savetxt(f"./polytrope_density.txt", ret_dict["rho"], delimiter=",", newline=",")


if __name__ == "__main__":

    test_polytrope()

