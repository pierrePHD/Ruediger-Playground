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


def test_central_densities():
    species_file = ppath + "/eostable/species05.txt"
    eos = loadhelm_eos(ppath + "/eostable/helm_table.dat", species_file, True)

    # Implicit xnuc of 0.5 C, 0.5 O
    temp_c = 5e5

    for mass in [0.35, 0.65, 1.10]:
        wd_mass = mass * msol
        with SuppressStdout():
            rhoc = wdCOgetRhoCFromMassExact(wd_mass, eos, temp=temp_c)
            
        print(f"{mass=} {rhoc=:e}")
        
        
def test_total_mass():
    species_file = ppath + "/eostable/species05.txt"
    eos = loadhelm_eos(ppath + "/eostable/helm_table.dat", species_file, True)

    # Implicit xnuc of 0.5 C, 0.5 O
    temp_c = 5e5

    for rho_c in [8e5, 5e6, 7e7]:
        with SuppressStdout():
            mass = wdCOgetMassFromRhoCExact(rho_c, eos, temp=temp_c)
        print(f"{rho_c=:e} {mass=}")


if __name__ == "__main__":

    test_central_densities()
    test_total_mass()

