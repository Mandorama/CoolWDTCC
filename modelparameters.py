#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:23:20 2023

@author: gabriel
"""

import time
import numpy as np
import pandas as pd
import Spectrum as spec

SpectralLines = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
ParametersGrid = pd.DataFrame(np.ones([260, 14]), columns=['Teff', 'logg',
                                                           'Alpha_Amplitude', 'Alpha_Std',
                                                           'Beta_Amplitude', 'Beta_Std',
                                                           'Gamma_Amplitude', 'Gamma_Std',
                                                           'Delta_Amplitude', 'Delta_Std',
                                                           'Epsilon_Amplitude', 'Epsilon_Std',
                                                           'Zeta_Amplitude', 'Zeta_Std'])
StartTime = time.time()


def assign_parameters(array, model):
    """Takes a model and returns an array with its stellar and line parameters.

    Parameters
    ----------
    array : ndarray
        Empty array to be fulfilled with parameters.
    model : <class 'Spectrum.ModelSpectrum'>
        A model from the module Spectrum that gives information about the 
        spectra of TLUSTY stellar model and its stellar parameters - in this
        case, the effective temperature and superficial gravity.

    Returns
    -------
    array : ndarray
        An array containing the effective temperature, the superficial gravity,
        and the amplitude and standard deviation of alpha, beta, gamma, delta,
        epsilon and zeta lines of the spectra of the given model.

    """

    array[0] = model.Teff
    array[1] = model.logg
    parameter_index = 2
    for line in SpectralLines:
        array[parameter_index] = model.LineParameters(line)[0]
        array[parameter_index+1] = model.LineParameters(line)[1]

        parameter_index += 2
    return array


ModelsList = [spec.ModelSpectrum(spec.Model(i)) for i in range(260)]

for Index, Model in enumerate(ModelsList):
    EmptyRow = np.ones(14)
    RowOfParameters = assign_parameters(EmptyRow, Model)
    ParametersGrid.iloc[Index] = RowOfParameters

LoopTime = time.time()

ParametersGrid.to_csv("ParametersGrid", index=False)

print(LoopTime-StartTime)
print(ParametersGrid)
