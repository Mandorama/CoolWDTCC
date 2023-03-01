#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 08:58:11 2023

@author: gabriel
"""

import pandas as pd

ModelsGrid = pd.read_csv("GridCLEAN.csv")

import numpy as np
from astropy import units as u
from specutils.spectra import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.fitting import fit_generic_continuum
import warnings
import math
from astropy.modeling import models
from specutils.fitting import fit_lines

class Model:
    
    def __init__(self, ModelIndex):
        self.ModelData = pd.read_csv(ModelsGrid.Arquivo[ModelIndex],
                                     header=None,sep="\s+")
        self.FluxData = np.array(self.ModelData[1].copy())*u.Unit("erg / (Angstrom cm2 s)")
        self.SpectralAxis = np.array(self.ModelData [0].copy())*u.Angstrom
        self.Spectrum = Spectrum1D(flux=self.FluxData, spectral_axis=self.SpectralAxis)
        
    def SpectrumFittedParameters(self, Line):
        LineGaussianFitting = Model.SpectrumLineFitting(self.Spectrum, Line)
        LineAmplitude = LineGaussianFitting.amplitude.value
        LineStdDev = LineGaussianFitting.stddev.value
        
        Parameters = np.array([LineAmplitude,LineStdDev])
        
        return Parameters
    
    def SpectrumLineFitting(self, FittedLine):
        BalmerSeriesLineIndex = {'Alpha': 0,
                                 'Beta': 1,
                                 'Gamma': 2,
                                 'Delta': 3,
                                 'Epsilon': 4,
                                 'Zeta': 5}
        
        SpectrumLimits =Model. LineIndexLimits(FittedLine)
        LineBaseSpectrum = Model.CreateBaseSpectrum(self.Spectrum, SpectrumLimits[0], SpectrumLimits[1])
        
        NoiseRegion = Model.NoiseRegionbyLine(FittedLine)
        LineContinuumFitted = Model.FitContinuumSpectrum(LineBaseSpectrum, NoiseRegion)
        
        LineIndex = BalmerSeriesLineIndex[FittedLine]
        
        GaussianParameters = Model.GaussianFitting(LineBaseSpectrum, LineContinuumFitted, LineIndex)
        
        return GaussianParameters
    
    @staticmethod
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
    
    def CreateBaseSpectrum(self, InferiorLimitAngstrom, SuperiorLimitAngstrom):
        BaseSpectrum = self.Spectrum[InferiorLimitAngstrom,SuperiorLimitAngstrom]
        
        return BaseSpectrum
    
    @staticmethod
    def NoiseRegionbyLine(FittedLine):
        NoiseRegionsParameters = {'Alpha': [0,2],
                                  'Beta': [1,2],
                                  'Gamma': [2,1.5],
                                  'Delta': [3,2],
                                  'Epsilon': [4,1.8],
                                  'Zeta': [5,1.1]}
        
        LineNoiseRegion = NoiseRegionsParameters[FittedLine]
        NoiseRegion = Model.SetSubRegion(LineNoiseRegion[0], LineNoiseRegion[1])
        
        return NoiseRegion
    
    @staticmethod
    def SetSubRegion(BalmerSeriesIndex, ScalingTolerenceFactor):
        BalmerToleranceAngstrom = np.array([300,300,150,100,60,30,30,20,20])
        
        SubRegion = SpectralRegion((Model.BalmerSeriesAngstrom[BalmerSeriesIndex]-BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom, 
                              (Model.BalmerSeriesAngstrom[BalmerSeriesIndex]+BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom)
        
        return SubRegion
    
    @staticmethod
    def FitContinuumSpectrum(BaseSpectrum, RegionsToBeExcluded):
        ContinuumFit = Model.CreateContinuumFitFunction(BaseSpectrum,RegionsToBeExcluded)
        ContinuumSpectrum = ContinuumFit(BaseSpectrum.spectral_axis)
        
        return ContinuumSpectrum
    
    @staticmethod
    def CreateContinuumFitFunction(BaseSpectrum, RegionsToBeExcluded):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ContinuumFitFunction = fit_generic_continuum(BaseSpectrum, exclude_regions=RegionsToBeExcluded)
            
            return ContinuumFitFunction
        
    @staticmethod 
    def GaussianFitting(BaseSpectrum, ContinuumFittedSpectrum, BalmerSeriesIndex):
        NormalizedSpectrum = Model.NormalizeSpectrum(BaseSpectrum, ContinuumFittedSpectrum)
        LinePeak = Model.FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex)
        ShiftedSpectrum = Model.PeakCentralizedSpectrum(NormalizedSpectrum, LinePeak)
        
        AmplitudeFirstGuess = ShiftedSpectrum.flux[LinePeak]
        InitialParameters = Model.GaussianParameterFirstGuess(AmplitudeFirstGuess)
        GaussianFitResults = fit_lines(ShiftedSpectrum, InitialParameters)
        
        return GaussianFitResults
    
    @staticmethod 
    def NormalizeSpectrum(BaseSpectrum, ContinuumSpectrum):
        Normalization = BaseSpectrum/ContinuumSpectrum
        
        return Normalization
    
    @staticmethod 
    def FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex):
        LinePeak = np.where(NormalizedSpectrum.spectral_axis/u.Angstrom==Model.FindNearestPointIndex(NormalizedSpectrum,
                                                                                               Model.BalmerSeriesAngstrom[BalmerSeriesIndex]))[0][0]
        return LinePeak
    
    @staticmethod     
    def FindNearestPointIndex(NormalizedSpectrum, BalmerSeriesPeak):
        SpectralAxis = NormalizedSpectrum.spectral_axis/u.Angstrom
        index = np.searchsorted(SpectralAxis, BalmerSeriesPeak, side="left")
        
        if Model.IndexCorrectionNeeded(index, SpectralAxis, BalmerSeriesPeak):
            NearestPointIndex = SpectralAxis[index-1]
        else:
            NearestPointIndex = SpectralAxis[index]
        
        return NearestPointIndex
    
    @staticmethod
    def IndexCorrectionNeeded(index, SpectralAxis, Peak):
        return index == len(SpectralAxis) or math.fabs(Peak - SpectralAxis[index-1]) < math.fabs(Peak - SpectralAxis[index])
    
    @staticmethod     
    def PeakCentralizedSpectrum(NormalizedSpectrum, Peak):
        CentralizedSpectrum = Spectrum1D(flux=NormalizedSpectrum.flux-1,
                          spectral_axis=NormalizedSpectrum.spectral_axis-NormalizedSpectrum.spectral_axis[Peak])
        return CentralizedSpectrum
    
    @staticmethod 
    def GaussianParameterFirstGuess(AmplitudeGuess):
        InitialGuess = models.Gaussian1D(amplitude=AmplitudeGuess, mean=0, stddev=20)
        
        return InitialGuess
    
Teste = Model(0)
print(Teste.Spectrum)