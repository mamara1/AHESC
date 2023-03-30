#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:05:00 2023

@author: mamara
"""

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import scipy.interpolate, scipy.integrate, pandas, sys
import scipy.constants as constants
from scipy.optimize import fmin
from scipy import constants as sc
import photon_in
import SC


#HCSC = np.array(pd.read_table("data_mathematica__Suchet/HCSC.csv", sep=',', header=0))
#plt.plot(HCSC[:,0], HCSC[:,1])

Th = 3000.
alpha_sol =  32 / 60 * np.pi / 180

eps_in = np.pi * (np.sin(alpha_sol / 2)) ** 2

eps_em = np.pi

listEg1 = np.arange(0.5, 2.51, 0.05)

Tamb = 300
Tsun = 5800

#PCE_lost = SC.Shockley_Quesseir(listEg1, 300, 5800, eps_sol, eps_emit)
#PCE_lost = SC.HCSS(listEg1, Tamb, Tsun, Th, eps_in, eps_em)
#PCE_lost =  SC.Shockley_Quesseir(listEg1, 300., eps_sol, eps_emit)  #OK

PCE_lost =  SC.IBSC(listEg1, Tamb, Tsun, eps_in, eps_em)

fig, ax = plt.subplots()
ax.fill_between(listEg1, PCE_lost[0], 0, color='blue', alpha=0.350, label ='Convertie')
# ax.fill_between(listEg1,  PCE_lost[1], PCE_lost[0], color='red', alpha=0.25, label ='Perte de collecte')
# ax.fill_between(listEg1,  PCE_lost[2], PCE_lost[1], color='purple', alpha=0.55, label ='Thermalisation')
# ax.fill_between(listEg1,  PCE_lost[3], PCE_lost[2], color='orange', alpha=0.25, label ='Pertes radiatives')
# ax.fill_between(listEg1,  PCE_lost[4], PCE_lost[3], color='green', alpha=0.25, label ='Photons non absorbés')
# ax.set_ylim(0, 1.)
# ax.set_xlim(0.5, 2.5)
# ax.set_xlabel('E$_g$ [eV]')
# ax.set_ylabel('P/P$_{in}$')
# # ax.text(1.12,0.15,"Convertie", color = 'blue',alpha=0.75, Fontsize =9.5)
# # ax.text(0.7,0.3,"Collectée", color = 'red',alpha=0.5, Fontsize =8.5)
# # ax.text(0.55, 0.45, "Recombinée", color = 'purple',alpha=0.75, Fontsize = 9.5)
# # ax.text(0.75, 0.7, "Thermalisée", color = 'orange',alpha=1., Fontsize =9.5)
# # ax.text(2, 0.8, "Non absorbée", color = 'green',alpha=1., Fontsize =9.5)
# ax.legend()
# ##plt.savefig('SQbis.pdf')
# ##plt.savefig('SQbis.png')

plt.show()

