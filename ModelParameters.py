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


def AssignParameters(Array, Model):
    Array[0] = Model.Teff
    Array[1] = Model.logg
    ParameterIndex = 2
    for Line in SpectralLines:
        Array[ParameterIndex] = Model.LineParameters(Line)[0]
        Array[ParameterIndex+1] = Model.LineParameters(Line)[1]

        ParameterIndex += 2
    return Array


ModelsList = [spec.ModelSpectrum(spec.Model(i)) for i in range(260)]

for Index, Model in enumerate(ModelsList):
    EmptyRow = np.ones(14)
    RowOfParameters = AssignParameters(EmptyRow, Model)
    ParametersGrid.iloc[Index] = RowOfParameters

LoopTime = time.time()

ParametersGrid.to_csv("ParametersGrid", index=False)

print(LoopTime-StartTime)
print(ParametersGrid)
