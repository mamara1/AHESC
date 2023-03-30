#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:59:05 2023

@author: mamara
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.interpolate, scipy.integrate, pandas, sys
import scipy.constants as constants
from scipy.optimize import fmin
from scipy import constants as sc

    # Planck law
def Photon_flux(E1, E2, T, V):
    fac = sc.e * (2 * np.pi/(sc.c ** 2 * sc.h ** 3))
    func = lambda x: sc.e ** 2 * x ** 2 / (np.exp((sc.e * x - V) /(sc.k * T)) - 1.)
    return (fac * scipy.integrate.quad(func, E1, E2)[0])

def Energy_flux(E1, E2, T, V):
    fac = sc.e * (2 * np.pi/(sc.c ** 2 * sc.h ** 3))
    func = lambda x: sc.e * x * sc.e ** 2 * x ** 2 / (np.exp((sc.e * x - V) /(sc.k * T)) - 1.)
    return (fac * scipy.integrate.quad(func, E1, E2)[0])

def HCSS(listEg1, Tc, Tsol, Th, eps_sol, eps_emit):
    Tref = Tc / Th
    Carnot = 1 - Tc / Th
    P_el = np.zeros(len(listEg1))
    P_rad_f = np.zeros(len(listEg1))
    N_sun = np.zeros(len(listEg1))
    P_sun = np.zeros(len(listEg1))
    current_density = np.zeros(len(listEg1))
    Vmax = np.zeros(len(listEg1))
    Emax = np.zeros(len(listEg1))
    Piso = np.zeros(len(listEg1))
    Pabs = np.zeros(len(listEg1))
    Pth = np.zeros(len(listEg1))
    Pcol = np.zeros(len(listEg1))

    Pin = eps_sol * Energy_flux(1.E-9, 10., Tsol, 0.) / np.pi


    for j in range(len(listEg1)):
        N_sun[j]= eps_sol * Photon_flux(listEg1[j],10., 5800., 0.) / np.pi
        P_sun[j] = eps_sol * Energy_flux(listEg1[j], 10., Tsol, 0) / np.pi

        listV = np.arange(-6, 0.95 * listEg1[j], 0.05)
        P1 = np.zeros(len(listV))
        P2 = np.zeros(len(listV))
        P3 = np.zeros(len(listV))
        P4 = np.zeros(len(listV))
        N_rad = np.zeros(len(listV))
        P_rad = np.zeros(len(listV))

        for i in range(len(listV)):
            N_rad[i] = eps_emit * Photon_flux(listEg1[j], 10., Th, sc.e * listV[i]) / np.pi

            P_rad[i] = eps_emit * Energy_flux(listEg1[j], 10., Th, sc.e * listV[i]) / np.pi

            P1[i] = sc.e * listV[i] * (N_sun[j] - N_rad[i]) * Tref

            P2[i] = (P_sun[j] - P_rad[i]) * Carnot

            P4[i] =  (P1[i] + P2[i])

        max_index = np.argmax(P4)
        delta_mu = listV[max_index]
        P_el[j] =  P4[max_index]
        P_rad_f[j] = eps_emit * Energy_flux(listEg1[j], 10., Th,  sc.e * delta_mu) / np.pi
        current_density[j] = sc.e * (N_sun[j] - eps_emit * Photon_flux(listEg1[j], 10., Th, sc.e * listV[max_index]) / np.pi)
        Vmax[j] = P_el[j] / current_density[j]
        Emax[j] = ( Vmax[j]  - delta_mu  * Tref ) / Carnot
        Piso[j] = (N_sun[j] - eps_emit * Photon_flux(listEg1[j], 10., Th, sc.e * listV[max_index])) * Tref * sc.e * (Emax [j] - listV[max_index])

        Pcol[j] = current_density[j] * Tref *  (Emax [j] - listV[max_index])

        Pabs[j] = eps_sol * Energy_flux(0., listEg1[j], Tsol, 0.) / np.pi


    PCE = [P_el, Pabs, Pth, P_rad_f,Pcol]

    PCE_list=[P_el / Pin, (P_el + Pcol) / Pin, (P_el + Pcol + P_rad_f) / Pin, (P_el + Pcol + P_rad_f + Pth) / Pin, (P_el + Pcol + P_rad_f + Pth + Pabs) / Pin]

    return np.array(PCE_list)
#, np.array(PCE) / Pin


def Shockley_Quesseir(listEg1, Tc, Tsol, eps_sol, eps_emit):
    #Tsol = 5800.
    Pin = eps_sol * Energy_flux(1.E-9, 10., Tsol, 0.) / np.pi
    P_el = np.zeros(len(listEg1))
    N_sun = np.zeros(len(listEg1))
    Prad = np.zeros(len(listEg1))
    Pabs = np.zeros(len(listEg1))
    Pcol = np.zeros(len(listEg1))
    Pth = np.zeros(len(listEg1))
    Vmax = np.zeros(len(listEg1))

    for j in range(len(listEg1)):
       # print(listEg1[j])


        listV = np.arange(0., 0.95 * listEg1[j], listEg1[j] / 200.)

        N_sun[j]= eps_sol * Photon_flux(listEg1[j],10., 5800., 0.) / np.pi


        current_density = np.zeros(len(listV))

        P1 =  np.zeros(len(listV))
        Tc = 300.

        for i in range(len(listV)):

            current_density[i] = sc.e * (N_sun[j] - eps_emit * Photon_flux(listEg1[j], 10., Tc, sc.e * listV[i]) / np.pi)
            P1[i] = current_density[i] * listV[i]

        max_index = np.argmax(P1)


        Vmax[j] = listV[max_index]
        Prad[j] = eps_emit * Energy_flux(listEg1[j], 10., Tc, sc.e * listV[max_index]) / np.pi
        Pabs[j] = eps_sol * Energy_flux(0., listEg1[j] , Tsol, 0.) / np.pi
        Pth[j]  = eps_sol * (Energy_flux(listEg1[j], 10., Tsol, 0.) - Photon_flux(listEg1[j], 10., Tsol, 0.) * sc.e * listEg1[j]) / np.pi
        Pcol[j] = (eps_sol * Photon_flux(listEg1[j], 10., Tsol, 0.) - eps_emit * Photon_flux(listEg1[j], 10., Tc, sc.e * listV[max_index]) ) * sc.e * (listEg1[j] - listV[max_index]) / np.pi
        P_el [j]= P1[max_index]
        PCE_list = [P_el / Pin, (P_el + Pcol) / Pin, (P_el + Pcol + Prad) / Pin, (P_el + Pcol + Prad + Pth) / Pin,(P_el + Pcol + Prad + Pth + Pabs) / Pin ]

    return np.array(PCE_list)

def IBSC(listEg1, Tc, Tsol, eps_sol, eps_emit):
    print('start')
    Pin = eps_sol * Energy_flux(1.E-9, 10., Tsol, 0.) / np.pi

    P_el = np.zeros(len(listEg1))
    N_sun = np.zeros(len(listEg1))
    Prad = np.zeros(len(listEg1))
    Pabs = np.zeros(len(listEg1))
    Pcol = np.zeros(len(listEg1))
    Pth = np.zeros(len(listEg1))
    Vmax = np.zeros(len(listEg1))

    for j in range(len(listEg1)):
        Eg1 = listEg1[j]
        listEg2 = np.arange(Eg1 / 2., Eg1, 0.05)
        Volt = np.arange(0, Eg1, 0.01)
        G10 = eps_sol * Photon_flux(Eg1, 10.0, Tsol, 0) / np.pi
        R10  = eps_emit * Photon_flux(Eg1, 10., 300., 0.) / np.pi
        Eg2_opt = np.zeros(len(listEg1))

        for i in range(len(listEg2)):
            Eg2 = listEg2[i]
            G2 = eps_sol * Photon_flux(Eg2, Eg1, Tsol, 0) / np.pi
            R20  = eps_emit * Photon_flux(Eg2,Eg1, Tc, 0.) / np.pi

            Eg3 = Eg1 - Eg2

            G3 = eps_sol * Photon_flux(Eg3, Eg2, Tsol,0.) / np.pi
            R30 = eps_emit * Photon_flux(Eg3,Eg2, Tc, 0.) / np.pi
            Pelec = np.zeros(len(Volt))
            Pelec_mpp = np.zeros(len(listEg2))
            mu2 = np.zeros(len(Volt))
            for nv in range(len(Volt)):
                mu1 = Volt[nv]
                P1 = ((G2 - G3) + np.sqrt((G2-G3) ** 2 - 4. * R20 * R30 * np.exp(sc.e * mu1 / sc.k / Tc)) ) / 2. / R20
                mu2[nv] = sc.k * Tc * np.log(P1) / sc.e

                Pelec[nv] = sc.e * mu1 * ( (eps_sol * Photon_flux(listEg1[j], 10., Tsol, 0.) - eps_emit * Photon_flux(listEg1[j], 10., Tc, sc.e * mu1) )  + (eps_sol * Photon_flux(listEg2[i], listEg1[j], Tsol, 0.) - eps_emit * Photon_flux(listEg2[i], listEg1[j], Tc, sc.e * mu2[nv]) ))
            Pelec_mpp[i] = np.argmax(Pelec)


        Eg2_opt[j] = listEg2[Pelec_mpp.argmax()]
      #  print('j=', j, 'Eg1=', Eg1, 'Eg2_opt=', Eg2_opt[j])
        P_el [j]= sc.e * Volt[Pelec.argmax()] * ( (eps_sol * Photon_flux(listEg1[j], 10., Tsol, 0.) - eps_emit * Photon_flux(listEg1[j], 10., Tc, sc.e * Volt[Pelec.argmax()]) )  + (eps_sol * Photon_flux(Eg2_opt[j], listEg1[j], Tsol, 0.) - eps_emit * Photon_flux(Eg2_opt[j], listEg1[j], Tc, sc.e * mu2[Pelec.argmax()]) ))
        print(P_el)

            # while (np.imag(P1.all()) == 0.) & P1.all() > 0.:
            #         mu2 = sc.k * Tc * np.log(P1) / sc.e
            #         mu3 = mu1 - mu2
            #         Power = mu1 * sc.e * ((G10- R10 *np.exp(sc.e * mu1/ sc.k / Tc)) \
            #                               +(G3 - R30 * np.exp(sc.e * mu3 / sc.k / Tc)) )
            #         print(Power)







    PCE_list =P_el
    return np.array(PCE_list)