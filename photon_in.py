#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:11:45 2023

@author: mamara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:02:12 2023


@author: mamara
"""


import numpy as np
import scipy.interpolate, scipy.integrate
from scipy import constants as sc


# Planck law
def Particle_flux(E1, E2, T, V):
    fac = sc.e * (2 * np.pi/(sc.c ** 2 * sc.h ** 3))
    func = lambda x: sc.e ** 2 * x ** 2 / (np.exp((sc.e * x - V) /(sc.k * T)) - 1.)
    return (fac * scipy.integrate.quad(func, E1, E2)[0])

def Energy_flux(E1, E2, T, V):
    fac = sc.e * (2 * np.pi/(sc.c ** 2 * sc.h ** 3))
    func = lambda x: sc.e * x * sc.e ** 2 * x ** 2 / (np.exp((sc.e * x - V) /(sc.k * T)) - 1.)
    return (fac * scipy.integrate.quad(func, E1, E2)[0])
