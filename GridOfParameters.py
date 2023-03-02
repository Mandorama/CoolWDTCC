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
LinesParametersDataFrame = pd.DataFrame(np.ones([260, 12]), columns=['Alpha_Amplitude', 'Alpha_Std',
                                                                     'Beta_Amplitude', 'Beta_Std',
                                                                     'Gamma_Amplitude', 'Gamma_Std',
                                                                     'Delta_Amplitude', 'Delta_Std',
                                                                     'Epsilon_Amplitude', 'Epsilon_Std',
                                                                     'Zeta_Amplitude', 'Zeta_Std'])
StartTime = time.time()


def AssignParameters(Array, Model):
    ParameterIndex = 0
    for Line in SpectralLines:
        Array[ParameterIndex] = Model.getLineParameters(Line)[0]
        Array[ParameterIndex+1] = Model.getLineParameters(Line)[1]

        ParameterIndex += 2
    return Array


ModelsList = [spec.ModelSpectrum(spec.ModelData(i)) for i in range(260)]

for Index, Model in enumerate(ModelsList):
    EmptyRow = np.ones(12)
    RowOfParameters = AssignParameters(EmptyRow, Model)
    LinesParametersDataFrame.iloc[Index] = RowOfParameters

LoopTime = time.time()

ModelsGrid = spec.getModelsGrid()
StellarParameters = ModelsGrid.drop(labels="Arquivo", axis=1).copy()
ParametersGrid = StellarParameters.join(LinesParametersDataFrame).copy()

ParametersGrid.to_csv("ParametersGrid",index=False)

print(LoopTime-StartTime)
print(ParametersGrid)
