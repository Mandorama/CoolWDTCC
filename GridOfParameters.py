#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:23:20 2023

@author: gabriel
"""

import pandas as pd

ModelsGrid = pd.read_csv("GridCLEAN.csv")

import numpy as np
from astropy import units as u
from specutils.spectra import Spectrum1D

class ModelSpectrum:
    
    def __init__(self, ModelIndex):
        self.ModelData = pd.read_csv(ModelsGrid.Arquivo[ModelIndex],
                                     header=None,sep="\s+")
        self.FluxData = np.array(self.ModelData[1].copy())*u.Unit("erg / (Angstrom cm2 s)")
        self.SpectralAxis = np.array(self.ModelData [0].copy())*u.Angstrom
        
    def Spectrum1D(self, InferiorIndex,SuperiorIndex):
        return(Spectrum1D(flux=self.FluxData[InferiorIndex:SuperiorIndex], 
                          spectral_axis=self.SpectralAxis[InferiorIndex:SuperiorIndex]))

from specutils.spectra import SpectralRegion
from specutils.fitting import fit_generic_continuum
import warnings
import math
from astropy.modeling import models
from specutils.fitting import fit_lines

BalmerSeriesAngstrom = np.array([6562.79,4861.35,4340.472,4101.734,
                          3970.075,3889.064,3835.064,3797.909,3770.633])

def SpectrumFittedParameters(Model, Line):
    LineGaussianFitting = SpectrumLineFitting(Model, Line)
    LineAmplitude = LineGaussianFitting.amplitude.value
    LineStdDev = LineGaussianFitting.stddev.value
    
    Parameters = np.array([LineAmplitude,LineStdDev])
    
    return Parameters

def SpectrumLineFitting(Model, FittedLine):
    BalmerSeriesLineIndex = {'Alpha': 0,
                             'Beta': 1,
                             'Gamma': 2,
                             'Delta': 3,
                             'Epsilon': 4,
                             'Zeta': 5}
    
    SpectrumLimits = LineIndexLimits(FittedLine)
    LineBaseSpectrum = CreateBaseSpectrum(Model, SpectrumLimits[0], SpectrumLimits[1])
    
    NoiseRegion = NoiseRegionbyLine(FittedLine)
    LineContinuumFitted = FitContinuumSpectrum(LineBaseSpectrum, NoiseRegion)
    
    LineIndex = BalmerSeriesLineIndex[FittedLine]
    
    GaussianParameters = GaussianFitting(LineBaseSpectrum, LineContinuumFitted, LineIndex)
    
    return GaussianParameters

def LineIndexLimits(FittedLine):
    LineLimits = {'Alpha': [12550,17005],
                  'Beta': [12550,17005],
                  'Gamma': [12000,12550],
                  'Delta': [11650,12000],
                  'Epsilon': [11460,11650],
                  'Zeta': [11320,11460]}

    InferiorLimit = LineLimits[FittedLine][0]
    SuperiorLimit = LineLimits[FittedLine][1]
    
    Limits = np.array([InferiorLimit,SuperiorLimit])
    
    return Limits

def CreateBaseSpectrum(Model, InferiorLimitAngstrom, SuperiorLimitAngstrom):
    BaseSpectrum = Model.Spectrum1D(InferiorLimitAngstrom,SuperiorLimitAngstrom)
    
    return BaseSpectrum

def NoiseRegionbyLine(FittedLine):
    NoiseRegionsParameters = {'Alpha': [0,2],
                              'Beta': [1,2],
                              'Gamma': [2,1.5],
                              'Delta': [3,2],
                              'Epsilon': [4,1.8],
                              'Zeta': [5,1.1]}
    
    LineNoiseRegion = NoiseRegionsParameters[FittedLine]
    NoiseRegion = SetSubRegion(LineNoiseRegion[0], LineNoiseRegion[1])
    
    return NoiseRegion

def SetSubRegion(BalmerSeriesIndex, ScalingTolerenceFactor):
    BalmerToleranceAngstrom = np.array([300,300,150,100,60,30,30,20,20])
    
    SubRegion = SpectralRegion((BalmerSeriesAngstrom[BalmerSeriesIndex]-BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom, 
                          (BalmerSeriesAngstrom[BalmerSeriesIndex]+BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom)
    
    return SubRegion

def FitContinuumSpectrum(BaseSpectrum, RegionsToBeExcluded):
    ContinuumFit = CreateContinuumFitFunction(BaseSpectrum,RegionsToBeExcluded)
    ContinuumSpectrum = ContinuumFit(BaseSpectrum.spectral_axis)
    
    return ContinuumSpectrum

def CreateContinuumFitFunction(BaseSpectrum, RegionsToBeExcluded):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ContinuumFitFunction = fit_generic_continuum(BaseSpectrum, exclude_regions=RegionsToBeExcluded)
        
        return ContinuumFitFunction

def GaussianFitting(BaseSpectrum, ContinuumFittedSpectrum, BalmerSeriesIndex):
    NormalizedSpectrum = NormalizeSpectrum(BaseSpectrum, ContinuumFittedSpectrum)
    LinePeak = FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex)
    ShiftedSpectrum = PeakCentralizedSpectrum(NormalizedSpectrum, LinePeak)
    
    AmplitudeFirstGuess = ShiftedSpectrum.flux[LinePeak]
    InitialParameters = GaussianParameterFirstGuess(AmplitudeFirstGuess)
    GaussianFitResults = fit_lines(ShiftedSpectrum, InitialParameters)
    
    return GaussianFitResults

def NormalizeSpectrum(BaseSpectrum, ContinuumSpectrum):
    Normalization = BaseSpectrum/ContinuumSpectrum
    
    return Normalization

def FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex):
    LinePeak = np.where(NormalizedSpectrum.spectral_axis/u.Angstrom==FindNearestPointIndex(NormalizedSpectrum,
                                                                                           BalmerSeriesAngstrom[BalmerSeriesIndex]))[0][0]
    return LinePeak
    
def FindNearestPointIndex(NormalizedSpectrum, BalmerSeriesPeak):
    SpectralAxis = NormalizedSpectrum.spectral_axis/u.Angstrom
    index = np.searchsorted(SpectralAxis, BalmerSeriesPeak, side="left")
    
    if index > 0 and (index == len(SpectralAxis) or math.fabs(BalmerSeriesPeak - SpectralAxis[index-1]) < math.fabs(BalmerSeriesPeak - SpectralAxis[index])):
        NearestPointIndex = SpectralAxis[index-1]
    else:
        NearestPointIndex = SpectralAxis[index]
    
    return NearestPointIndex
    
def PeakCentralizedSpectrum(NormalizedSpectrum, Peak):
    CentralizedSpectrum = Spectrum1D(flux=NormalizedSpectrum.flux-1,
                      spectral_axis=NormalizedSpectrum.spectral_axis-NormalizedSpectrum.spectral_axis[Peak])
    return CentralizedSpectrum

def GaussianParameterFirstGuess(AmplitudeGuess):
    InitialGuess = models.Gaussian1D(amplitude=AmplitudeGuess, mean=0, stddev=20)
    
    return InitialGuess

SpectralLines = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
LinesParametersDataFrame = pd.DataFrame(np.ones([260,12]),columns=['Alpha_Amplitude','Alpha_Std',
                                                                  'Beta_Amplitude','Beta_Std',
                                                                  'Gamma_Amplitude','Gamma_Std',
                                                                  'Delta_Amplitude','Delta_Std',
                                                                  'Epsilon_Amplitude','Epsilon_Std',
                                                                  'Zeta_Amplitude','Zeta_Std'])

import time

StartTime = time.time()

for Index, Model in enumerate([ModelSpectrum(i) for i in range(260)]):
    Row = np.ones(12)
    ParameterIndex = 0
    
    for Line in SpectralLines:
        Row[ParameterIndex] = SpectrumFittedParameters(Model, Line)[0]
        Row[ParameterIndex+1] = SpectrumFittedParameters(Model, Line)[1]
        
        ParameterIndex += 2
        
    LinesParametersDataFrame.iloc[Index] = Row
    
LoopTime = time.time()
    
# print(ModelsGrid.drop(columns=["Arquivo"],axis=0))

StellarParameters = ModelsGrid.drop(labels="Arquivo",axis=1).copy()
ParametersGrid = StellarParameters.join(LinesParametersDataFrame).copy()

print(LoopTime-StartTime)
