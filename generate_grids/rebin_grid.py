
""" Create a rebinned grid for testing. This test grid should not be used for science """

import matplotlib.pyplot as plt
import argparse
import numpy as np
import h5py
from spectres import spectres


# define the original base grid
path_to_grids = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'
grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'

# open the original grid
original_grid = h5py.File(f'{path_to_grids}/{grid_name}.h5', 'r')

# open the new grid file
rebinned_grid = h5py.File(f'../tests/test_grid/test_grid.h5', 'w')

# copy various quantities (all excluding the spectra) from the original grid
for ds in ['metallicities', 'log10metallicities', 'log10ages', 'log10Q', 'star_fraction', 'remnant_fraction', 'lines']:
    original_grid.copy(original_grid[ds], rebinned_grid['/'], ds)

# define the length of the metallicities and ages
nZ = len(original_grid['metallicities'])  # number of metallicity grid points
na = len(original_grid['log10ages'])  # number of age grid points

# define the new wavelength grid
lmin, lmax, deltal = 100., 10000., 10.  # min wavelength, max wavelength, resolution
new_wavs = np.arange(lmin, lmax, deltal)


# alias
original_spectra = original_grid['spectra']
spectra_types = original_spectra.attrs['spec_names']

# create a group holding the spectra in the grid file
rebinned_spectra = rebinned_grid.create_group('spectra')
rebinned_spectra['wavelength'] = new_wavs
rebinned_spectra.attrs['spec_names'] = original_spectra.attrs['spec_names']

# loop over different spectra
for spectra_type in spectra_types:

    rebinned_spectra[spectra_type] = np.zeros((na, nZ, len(new_wavs)))

    # loop over ia and iZ
    for ia in range(na):
        for iZ in range(nZ):

            # loop over ia and iZ
            rebinned_spectra[spectra_type][ia, iZ] = spectres(
                new_wavs, original_spectra['wavelength'][:], original_spectra[spectra_type][ia, iZ, :])