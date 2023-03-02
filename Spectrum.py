#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:48:58 2023

@author: gabriel
"""

import math
import warnings
from astropy import units as u
import numpy as np
import pandas as pd
from specutils.spectra import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.fitting import fit_lines
from specutils.fitting import fit_generic_continuum
from astropy.modeling import models


def ModelData(Index, Gridfilepath="GridCLEAN.csv"):
    Grid = getModelsGrid(Gridfilepath)
    Data = pd.read_csv(
        f"./Espectros/{Grid.Arquivo[Index]}", header=None, sep="\s+")

    return Data

def getModelsGrid(Gridfilepath="GridCLEAN.csv"):
    Grid = pd.read_csv(Gridfilepath)
    
    return Grid


class ModelSpectrum:

    def __init__(self, ModelData):
        self.ModelData = ModelData
        self.FluxData = np.array(
            self.ModelData[1].copy())*u.Unit("erg / (Angstrom cm2 s)")
        self.SpectralAxis = np.array(self.ModelData[0].copy())*u.Angstrom
        self.Spectrum = Spectrum1D(flux=self.FluxData,
                                   spectral_axis=self.SpectralAxis)

    def getLineParameters(self, Line):
        LineGaussianFitting = SpectrumLineFitting(self, Line)
        LineAmplitude = LineGaussianFitting.amplitude.value
        LineStdDev = LineGaussianFitting.stddev.value

        Parameters = np.array([LineAmplitude, LineStdDev])

        return Parameters

    def getGaussianLineFit(self, Line):
        GaussianFit = SpectrumLineFitting(self, Line)

        return GaussianFit

    def getGaussianFits(self):
        GaussianFits = {'Alpha': SpectrumLineFitting(self, 'Alpha'),
                        'Beta': SpectrumLineFitting(self, 'Beta'),
                        'Gamma': SpectrumLineFitting(self, 'Gamma'),
                        'Delta': SpectrumLineFitting(self, 'Delta'),
                        'Epsilon': SpectrumLineFitting(self, 'Epsilon'),
                        'Zeta': SpectrumLineFitting(self, 'Zeta')}

        return GaussianFits

    def getNormalizedLineSpectrum(self, Line):
        CutSpectrum = LineSpectrum(self, Line)
        ContinuumSpectrum = FitContinuumSpectrum(CutSpectrum, Line)

        NormalizedSpectrum = CutSpectrum/ContinuumSpectrum

        return NormalizedSpectrum


BalmerSeriesAngstrom = np.array([6562.79, 4861.35, 4340.472, 4101.734,
                                 3970.075, 3889.064, 3835.064, 3797.909, 3770.633])


def SpectrumLineFitting(Model, FittedLine):
    LineBaseSpectrum = LineSpectrum(Model, FittedLine)
    LineContinuumFitted = FitContinuumSpectrum(
        LineBaseSpectrum, FittedLine)

    LineIndex = getBalmerSeriesLineIndex(FittedLine)

    GaussianParameters = GaussianFitting(
        LineBaseSpectrum, LineContinuumFitted, LineIndex)

    return GaussianParameters


def LineSpectrum(Model, FittedLine):
    SpectrumLimits = LineIndexLimits(FittedLine)
    LineSpectrum = CreateBaseSpectrum(
        Model, SpectrumLimits[0], SpectrumLimits[1])

    return LineSpectrum


def getBalmerSeriesLineIndex(FittedLine):
    BalmerSeriesLineIndex = {'Alpha': 0,
                             'Beta': 1,
                             'Gamma': 2,
                             'Delta': 3,
                             'Epsilon': 4,
                             'Zeta': 5}

    return BalmerSeriesLineIndex[FittedLine]


def LineIndexLimits(FittedLine):
    LineLimits = {'Alpha': [12550, 17005],
                  'Beta': [12550, 17005],
                  'Gamma': [12000, 12550],
                  'Delta': [11650, 12000],
                  'Epsilon': [11460, 11650],
                  'Zeta': [11320, 11460]}

    InferiorLimit = LineLimits[FittedLine][0]
    SuperiorLimit = LineLimits[FittedLine][1]

    Limits = np.array([InferiorLimit, SuperiorLimit])

    return Limits


def CreateBaseSpectrum(Model, InferiorLimitAngstrom, SuperiorLimitAngstrom):
    BaseSpectrum = Model.Spectrum[InferiorLimitAngstrom:SuperiorLimitAngstrom]

    return BaseSpectrum


def NoiseRegionbyLine(FittedLine):
    NoiseRegionsParameters = {'Alpha': [0, 2],
                              'Beta': [1, 2],
                              'Gamma': [2, 1.5],
                              'Delta': [3, 2],
                              'Epsilon': [4, 1.8],
                              'Zeta': [5, 1.1]}

    LineNoiseRegion = NoiseRegionsParameters[FittedLine]
    NoiseRegion = SetSubRegion(LineNoiseRegion[0], LineNoiseRegion[1])

    return NoiseRegion


def SetSubRegion(BalmerSeriesIndex, ScalingTolerenceFactor):
    BalmerToleranceAngstrom = np.array(
        [300, 300, 150, 100, 60, 30, 30, 20, 20])

    SubRegion = SpectralRegion((BalmerSeriesAngstrom[BalmerSeriesIndex]-BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom,
                               (BalmerSeriesAngstrom[BalmerSeriesIndex]+BalmerToleranceAngstrom[BalmerSeriesIndex]/ScalingTolerenceFactor)*u.Angstrom)

    return SubRegion


def FitContinuumSpectrum(BaseSpectrum, FittedLine):
    NoiseRegion = NoiseRegionbyLine(FittedLine)

    ContinuumFit = CreateContinuumFitFunction(
        BaseSpectrum, NoiseRegion)
    ContinuumSpectrum = ContinuumFit(BaseSpectrum.spectral_axis)

    return ContinuumSpectrum


def CreateContinuumFitFunction(BaseSpectrum, RegionsToBeExcluded):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ContinuumFitFunction = fit_generic_continuum(
            BaseSpectrum, exclude_regions=RegionsToBeExcluded)

        return ContinuumFitFunction


def GaussianFitting(BaseSpectrum, ContinuumFittedSpectrum, BalmerSeriesIndex):
    NormalizedSpectrum = NormalizeSpectrum(
        BaseSpectrum, ContinuumFittedSpectrum)
    LinePeak = FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex)
    ShiftedSpectrum = PeakCentralizedSpectrum(NormalizedSpectrum, LinePeak)

    GaussianFitResults = GaussianFit(ShiftedSpectrum, LinePeak)

    return GaussianFitResults


def NormalizeSpectrum(BaseSpectrum, ContinuumSpectrum):
    Normalization = BaseSpectrum/ContinuumSpectrum

    return Normalization


def FindLinePeak(NormalizedSpectrum, BalmerSeriesIndex):
    LinePeak = np.where(NormalizedSpectrum.spectral_axis/u.Angstrom == FindNearestPointIndex(NormalizedSpectrum,
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


def GaussianFit(Spectrum, Peak):
    AmplitudeFirstGuess = Spectrum.flux[Peak]
    InitialParameters = GaussianParameterFirstGuess(AmplitudeFirstGuess)
    GaussianFitResults = fit_lines(Spectrum, InitialParameters)

    return GaussianFitResults


def GaussianParameterFirstGuess(AmplitudeGuess):
    InitialGuess = models.Gaussian1D(
        amplitude=AmplitudeGuess, mean=0, stddev=20)

    return InitialGuess

if __name__ == "__main__":
    print(ModelSpectrum(ModelData(0)))
