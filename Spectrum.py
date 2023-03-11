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


class GenericModel:
    """ Simple model with parameters from grid file. """

    def __init__(self, index, grid_filepath="GridCLEAN.csv"):
        self.index = index
        self.grid = get_modelgrid(grid_filepath)

    @property
    def model_data(self):
        """ Flux as a function of wavelength in CGI units. """
        return get_model_data(self.grid, self.index)

    @property
    def effective_temperature(self):
        """" Effective Temperature (Teff) of the model in K. """
        return self.grid.Teff[self.index]

    @property
    def logarithmic_gravity(self):
        """ Logarithmic gravity (log g) of the model. """
        return self.grid.logg[self.index]


def get_modelgrid(grid_filepath="GridCLEAN.csv"):
    """ Create a pandas dataframe from a csv file of the grid. """
    grid = pd.read_csv(grid_filepath)

    return grid


def get_model_data(grid, index):
    """ Return the spectra data of a model from its index in the grid. """
    model_data = pd.read_csv(
        f"./Espectros/{grid.Arquivo[index]}", header=None, sep="\\s+")

    return model_data


class ModelSpectrum:
    """ 
    An object containing information about the flux and spectral data of a
    model and methods to obtain the parameters of its alpha, beta, gamma,
    delta, epsilon and zeta line parameters, calculated with gaussian fits.

    ---------------------------------------------------------------------------
    Properties:

    flux_data
        Contains an array (more specificaly an astropy.units.quantity.Quantity) 
        of the flux in CGS unit [erg / (Angstrom cm2 s)].

    spectral_data
        Contains an array (more specificaly an astropy.units.quantity.Quantity) 
        of the spectral space in CGS unit [Angstrom].

    spectrum
        A specutils.spectra.spectrum1d.Spectrum1D object, containing the flux
        and spectral axis data of a one dimensional spectrum.

    gaussianfits
        Dictionary containing the gaussian fit for each spectral line from
        alpha to zeta of the model spectrum.
    ---------------------------------------------------------------------------
    Methods:

    get_line_parameters
        Returns an array containing the amplitude and standard deviation of the
        gaussian fit of a given line for the sprectrum of the model.

    ---------------------------------------------------------------------------


    """

    def __init__(self, model_properties):
        self.model_data = model_properties.model_data
        self.teff = model_properties.effective_temperature
        self.logg = model_properties.logarithmic_gravity

    @property
    def flux_data(self):
        """ Array of the model's flux data in erg / (Angstrom cm2 s) """
        return np.array(
            self.model_data[1].copy())*u.Unit("erg / (Angstrom cm2 s)")

    @property
    def spectral_axis(self):
        """ Array of the model's spectral axis in Angstrom """
        return np.array(self.model_data[0].copy())*u.Angstrom

    @property
    def spectrum(self):
        """ One dimensional spectrum object containing flux and spectral data 
        """
        return Spectrum1D(flux=self.flux_data,
                          spectral_axis=self.spectral_axis)

    @property
    def gaussianfits(self):
        """ Dictionary containing all the gaussian fits of the model;s spectrum
        """
        gaussian_fits = {'Alpha': fit_spectrum_line(self, 'Alpha'),
                         'Beta': fit_spectrum_line(self, 'Beta'),
                         'Gamma': fit_spectrum_line(self, 'Gamma'),
                         'Delta': fit_spectrum_line(self, 'Delta'),
                         'Epsilon': fit_spectrum_line(self, 'Epsilon'),
                         'Zeta': fit_spectrum_line(self, 'Zeta')}

        return gaussian_fits

    def get_line_parameters(self, line):
        """ Returns the amplitude and standard deviation of a given line.

        Parameters
        ----------
        line : string
            Line whose parameters are to be informed. Lines accepted are:
            Alpha, Beta, Gamma, Delta, Epsilon and Zeta. Must be capitalized.

        Returns
        -------
        parameters : ndarray
            Array containing two values: Amplitude and standard deviation of
            the gaussian fit of the line.

        """
        line_gaussianfitting = fit_spectrum_line(self, line)
        line_amplitude = line_gaussianfitting.amplitude.value
        line_stddev = line_gaussianfitting.stddev.value

        parameters = np.array([line_amplitude, line_stddev])

        return parameters

    def normalized_linespectrum(self, line):
        """ Returns the normalized spectrum around a given line.

        Parameters
        ----------
        line : string
            Line whose parameters are to be informed. Lines accepted are:
            Alpha, Beta, Gamma, Delta, Epsilon and Zeta. Must be capitalized.

        Returns
        -------
        normalized_spectrum : specutils.spectra.spectrum1d.Spectrum1D 
            Returns a one dimensional spectrum object of the spectrum 
            normalized by its continumm around a given line.

        """
        cut_spectrum = line_spectrum(self, line)
        continuum_spectrum = fit_continuum_spectrum(cut_spectrum, line)

        normalized_spectrum = cut_spectrum/continuum_spectrum

        return normalized_spectrum


BalmerSeriesAngstrom = np.array([6562.79, 4861.35, 4340.472, 4101.734,
                                 3970.075, 3889.064, 3835.064, 3797.909, 3770.633])


def fit_spectrum_line(model, line):
    """ Returns the gaussian paramters for the fit of a given line"""
    spectrum = line_spectrum(model, line)
    line_continuum = fit_continuum_spectrum(
        spectrum, line)

    line_index = get_line_index(line)

    gaussian_parameters = GaussianFitting(
        spectrum, line_continuum, line_index)

    return gaussian_parameters


def line_spectrum(model, line):
    """ Returns the spectrum around a given line. """
    spectrum_limits = line_indexlimits(line)
    spectrum = get_cut_spectrum(
        model, spectrum_limits[0], spectrum_limits[1])

    return spectrum


def get_line_index(line):
    """ Returns an integer index corresponding to the given balmer line. """
    balmer_line_indices = {'Alpha': 0,
                           'Beta': 1,
                           'Gamma': 2,
                           'Delta': 3,
                           'Epsilon': 4,
                           'Zeta': 5}

    return balmer_line_indices[line]


def line_indexlimits(line):
    """ Returns the limits of each line in the TLUSTY model spectra. """
    line_limits = {'Alpha': [12550, 17005],
                   'Beta': [12550, 17005],
                   'Gamma': [12000, 12550],
                   'Delta': [11650, 12000],
                   'Epsilon': [11460, 11650],
                   'Zeta': [11320, 11460]}

    inferior_limit = line_limits[line][0]
    superior_limit = line_limits[line][1]

    limits = np.array([inferior_limit, superior_limit])

    return limits


def get_cut_spectrum(model, inferior_limit, superior_limit):
    """ Cut the model's spectrum in the desired region """
    cut_spectrum = model.spectrum[inferior_limit:superior_limit]

    return cut_spectrum


def fit_continuum_spectrum(spectrum, line):
    """ Returns the continuum spectrum around a given line. """
    noise_region = line_noise_region(line)

    continuum_fit = CreateContinuumFitFunction(
        spectrum, noise_region)
    continuum_spectrum = continuum_fit(spectrum.spectral_axis)

    return continuum_spectrum


def line_noise_region(line):
    """ Returns the spectral region of noise in Angstrom. """
    regions_paramters = {'Alpha': [0, 2],
                         'Beta': [1, 2],
                         'Gamma': [2, 1.5],
                         'Delta': [3, 2],
                         'Epsilon': [4, 1.8],
                         'Zeta': [5, 1.1]}

    line_noise_parameters = regions_paramters[line]
    noise_region = set_subregion(line_noise_parameters[0],
                                 line_noise_parameters[1])

    return noise_region


def set_subregion(balmer_series_index, scaling_tolerence_factor):
    """ Returns spectral subregion for a given balmer series index. """
    line_tolerance = np.array([300, 300, 150, 100, 60, 30, 30, 20, 20])
    lower_boundary = (BalmerSeriesAngstrom[balmer_series_index] -
                      line_tolerance[balmer_series_index])*u.Angstrom
    upper_boundary = (BalmerSeriesAngstrom[balmer_series_index] +
                      line_tolerance[balmer_series_index])*u.Angstrom
    sub_region = SpectralRegion((lower_boundary/scaling_tolerence_factor),
                                (upper_boundary/scaling_tolerence_factor))

    return sub_region


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


class ObservedSpectrum():
    pass


if __name__ == "__main__":
    TestModel = ModelSpectrum(GenericModel(0))

    print(TestModel.gaussianfits['Beta'])
