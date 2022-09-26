

import numpy as np

import matplotlib.pyplot as plt

from synthesizer.filters import SVOFilterCollection
from synthesizer.grid import SpectralGrid
from synthesizer.binned import SFH, ZH, generate_sfzh, SEDGenerator
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr

from astropy.cosmology import Planck18 as cosmo



if __name__ == '__main__':


    grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'

    grid = SpectralGrid(grid_name)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr }
    Z_p = {'log10Z': -2.0} # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p) # constant star formation
    sfh.summary() # print sfh summary
    Zh = ZH.deltaConstant(Z_p) # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass = stellar_mass)


    galaxy = SEDGenerator(grid, sfzh)


    # # --- simple dust and gas screen
    # galaxy.screen(tauV = 0.1)
    # galaxy.plot_spectra()

    # # --- should be identical to above
    # galaxy.pacman(tauV = 0.1)
    # galaxy.plot_spectra()

    # # --- half of light escapes without nebular reprocessing
    # galaxy.pacman(fesc = 0.5)
    # galaxy.plot_spectra()

    # --- no Lyman-alpha escapes
    # galaxy.pacman(fesc = 0.0, fesc_LyA = 0.0)
    # galaxy.plot_spectra()
    # galaxy.plot_spectra(spectra_to_plot = ['total'])

    # --- everything
    galaxy.pacman(fesc = 0.5, fesc_LyA = 0.5, tauV = 0.2)
    # galaxy.plot_spectra()


    # --- now calculate the observed frame spectra

    z = 4 # redshift
    sed = galaxy.spectra['total'] # choose total SED
    sed.get_fnu(cosmo, z) # generate observed frame spectra

    # --- calculate broadband luminosities
    filter_codes = [f'JWST/NIRCam.{f}' for f in ['F090W', 'F115W','F150W','F200W','F277W','F356W','F444W']] # define a list of filter codes
    filter_codes += [f'JWST/MIRI.{f}' for f in ['F770W']]
    fc = SVOFilterCollection(filter_codes, new_lam = sed.lamz)


    # --- measure broadband fluxes
    fluxes = sed.get_broadband_fluxes(fc)

    for filter, flux in fluxes.items(): print(f'{filter}: {flux:.2f}')  # print broadband fluxes

    galaxy.plot_observed_spectra(cosmo, z, fc = fc, spectra_to_plot = ['total'])  # make plot of observed including broadband fluxes (if filter collection object given)