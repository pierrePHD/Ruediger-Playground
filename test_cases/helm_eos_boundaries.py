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


def test_helm_eos_boundaries():
    species_file = ppath + "/eostable/species05.txt"
    helm_file = ppath + "/eostable/helm_table.dat"
    eos = loadhelm_eos(helm_file, species_file, True)
    xnuc = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    print(f"{'-' * 20}")

    for rho, e in zip([1e-11, 2e14], [1e11, 9.3e20]):
        temp_calculated, p_calculated = eos.egiven(rho, xnuc, e)
        print(f"{rho=:e} {e=:e} ")
        print(f"{temp_calculated=:e} {p_calculated=:e}")

    print(f"{'-' * 20}")

    for rho, p in zip([1e-11, 2e14], [1.0, 6.55e34]):
        temp_calculated, e_calculated = eos.pgiven(rho, xnuc, p)
        print(f"{rho=:e} {p=:e} ")
        print(f"{temp_calculated=:e} {e_calculated=:e}")

    print(f"{'-' * 20}")

    for rho, t in zip([1e-11, 2e14], [1.5e3, 1.65e12]):
        e_calculated, _, p_calculated, _ = eos.tgiven(rho, xnuc, t)
        print(f"{rho=:e} {t=:e} ")
        print(f"{p_calculated=:e} {e_calculated=:e}")

    print(f"{'-' * 20}")

    for rho, e in zip([1e-7], [1e13]):
        temp_calculated, p_calculated = eos.egiven(rho, xnuc, e)
        print(f"{rho=:e} {e=:e} ")
        print(f"{temp_calculated=:e} {p_calculated=:e}")

    print(f"{'-' * 20}")


if __name__ == "__main__":

    test_helm_eos_boundaries()

