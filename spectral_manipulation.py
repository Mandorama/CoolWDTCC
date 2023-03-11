#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:48:58 2023

@author: gabriel
"""

import math
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.modeling import models
from specutils.spectra import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils.fitting import fit_lines
from specutils.fitting import fit_generic_continuum


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
                                 3970.075, 3889.064])


def fit_spectrum_line(model, line):
    """ Returns the gaussian paramters for the fit of a given line"""
    spectrum = line_spectrum(model, line)
    gaussian_parameters = gaussian_fitting(
        spectrum, line)

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

    continuum_fit = create_continuum_fitfunction(spectrum, noise_region)
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


def create_continuum_fitfunction(original_spectrum, exclude_regions):
    """ Returns the continuum fitted from a given spectrum. """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        continuum_fitfuction = fit_generic_continuum(original_spectrum,
                                                     exclude_regions=exclude_regions)

        return continuum_fitfuction


def gaussian_fitting(spectrum, line):
    """ Returns the results of a gaussian fit of the spectrum around a line. """
    line_continuum = fit_continuum_spectrum(
        spectrum, line)
    normalized_spectrum = normalize_spectrum(spectrum, line_continuum)

    line_peak = find_linepeak(normalized_spectrum, line)
    shifted_spectrum = peakshift_spectrum(normalized_spectrum, line_peak)

    amplitude_firstguess = shifted_spectrum.flux[line_peak]
    initial_parameters = gaussian_first_guess(amplitude_firstguess)

    gaussianfit_results = fit_lines(shifted_spectrum, initial_parameters)

    return gaussianfit_results


def normalize_spectrum(spectrum, continuum_spectrum):
    """ Returns a normalized spectrum using the continuum. """
    normalized_spectrum = spectrum/continuum_spectrum

    return normalized_spectrum


def find_linepeak(normalized_spectrum, line):
    """ Returns the peak of a given line for a given spectrum. """
    line_index = get_line_index(line)
    nearest_point = find_nearest_point(normalized_spectrum,
                                       BalmerSeriesAngstrom[line_index])
    spectral_axis = normalized_spectrum.spectral_axis/u.Angstrom

    line_peak = np.where(spectral_axis == nearest_point)[0][0]

    return line_peak


def find_nearest_point(normalized_spectrum, balmer_peak):
    """ Returns the index in the spectrum closest to the balmer value. """
    spectral_axis = normalized_spectrum.spectral_axis/u.Angstrom
    index = np.searchsorted(spectral_axis, balmer_peak, side="left")

    if previous_index_condition(index, spectral_axis, balmer_peak):
        nearest_index = spectral_axis[index-1]
    else:
        nearest_index = spectral_axis[index]

    return nearest_index


def previous_index_condition(index, spectral_axis, balmer_peak):
    """ Evaluates if the previous index is more adequate to be the peak. """
    positive_index_condition = index > 0
    length_condition = index == len(spectral_axis)

    equal_to_spectral_length = positive_index_condition and length_condition

    previous_difference = math.fabs(balmer_peak - spectral_axis[index-1])
    present_difference = math.fabs(balmer_peak - spectral_axis[index])

    close_value_condition = previous_difference < present_difference

    return equal_to_spectral_length or close_value_condition


def peakshift_spectrum(normalized_spectrum, peak):
    """ Shift the spectrum so the peak is in the center and the flux is close
    to 0.
    """
    spectral_axis = normalized_spectrum.spectral_axis
    centralized_spectrum = spectral_axis - spectral_axis[peak]

    shifted_spectrum = Spectrum1D(flux=normalized_spectrum.flux-1,
                                  spectral_axis=centralized_spectrum)
    return shifted_spectrum


def gaussian_first_guess(amplitude_guess):
    """ Returns a one-dimensional gaussinan model to be used as a first guess. """
    initial_guess = models.Gaussian1D(
        amplitude=amplitude_guess, mean=0, stddev=20)

    return initial_guess


class ObservedSpectrum():
    """ To be done. Will be the equivalent of ModelSpectrum but for real SDSS
    spectra manipulation.
    """


if __name__ == "__main__":
    TestModel = ModelSpectrum(GenericModel(0))

    print(TestModel.gaussianfits['Beta'])
