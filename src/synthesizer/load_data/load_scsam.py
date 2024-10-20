"""A module for interfacing with the outputs of Semi Analytic Models.

Currently implemented are loading methods for
- SC-SAM (using a parametric method)
- SC-SAM (using a particle method)
"""

import numpy as np
from unyt import Msun, yr

from regrid_sfh import get_grid_map, rebin_grid

from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars as ParticleStars


def load_SCSAM(
        fname, method, redshift, cosmo, grid=None, grid_map=None, verbose=False
):
    """
    Read an SC-SAM star formation data file.

    Returns a list of galaxy objects, halo indices, and birth halo IDs.
    Adapted from code by Aaron Yung.

    Args:
        fname (str):
            The SC-SAM star formation data file to be read.
        method (str):
            How to model SFH. Options: 'particle' or 'parametric'
            'particle' treats each age-Z bin as a particle.
            'parametric' rebins the SFH grid to the SPS grid binning
        redshift (float):
            Redshift of the galaxies that will be loaded
        cosmo (object):
            Astropy cosmology object
        grid (grid object):
            Grid object to extract from (needed for parametric method).
        grid_map (dict):
            Dictionary that contains information for mapping the old,
            original SFH grid to the desired new grid (needed for
            parametric method). Obtained from get_grid_map function in
            regrid_sfh.py. 
        verbose (bool):
            Are we talking?

    Returns:
        tuple:
            galaxies (list):
                list of galaxy objects
            halo_ind_list (list):
                list of halo indices
            birthhalo_id_list (list):
                birth halo indices
    """
    # Prepare to read SFHist file
    sfhist = open(fname, "r")
    lines = sfhist.readlines()

    # Set up halo index, birth halo ID, redshift and galaxy object lists
    halo_ind_list = []
    birthhalo_id_list = []
    redshift_list = []
    galaxies = []

    # Line counter
    count = 0
    count_gal = 0

    # Read file line by line as it can be large
    for line in lines:
        # Get age-metallicity grid structure
        if count == 0:
            Z_len, age_len = [int(i) for i in line.split()]
            if verbose:
                print(
                    f"There are {Z_len} metallicity bins and "
                    f"{age_len} age bins."
                )

        # Get metallicity bins
        if count == 1:
            Z_lst = [float(i) for i in line.split()]  # logZ in solar units
            if verbose:
                print(f"Z_lst (log10Zsun): {Z_lst}")
            # Check that this agrees with the expected grid structure
            if len(Z_lst) != Z_len:
                print("Wrong number of Z bins.")
                break
            Z_sun = 0.02  # Solar metallicity
            Z_lst = 10 ** np.array(Z_lst) * Z_sun  # Unitless
            if verbose:
                print(f"Z_lst (unitless): {Z_lst}")

        # Get age bins
        if count == 2:
            age_lst = [float(i) for i in line.split()]  # Gyr
            if verbose:
                print(f"age_lst: {age_lst}")
            # Check that this agrees with the expected grid structure
            if len(age_lst) != age_len:
                print("Wrong number of age bins.")
                break

        # Get galaxy data
        # The data chunk for each galaxy consists of one header line,
        # followed by the age x Z grid.
        # Thus it takes up age_len+1 lines.
        if (count - 3) % (age_len + 1) == 0:
            # The line preceding each age x Z grid contains:
            halo_ind = int(line.split()[0])
            birthhalo_id = int(line.split()[1])
            redshift = float(line.split()[2])

            # Append this information to its respective list
            halo_ind_list.append(halo_ind)
            birthhalo_id_list.append(birthhalo_id)
            redshift_list.append(redshift)

            # Start a new SFH array, specific to the galaxy
            SFH = []

        # If the age x Z grid of a galaxy is being read:
        if (count - 3) % (age_len + 1) != 0 and count > 3:
            _grid = [float(i) for i in line.split()]
            SFH.append(_grid)

        # If the last line of an age x Z grid has been read:
        # (i.e we now have the full grid of a single galaxy)
        if (count - 3) % (age_len + 1) == age_len and count > 3:
            count_gal += 1

            # Convert SFH units
            SFH = np.array(SFH) * 10**9  # Msun

            # Convert age axis of SFH grid from age of the universe to stellar ages
            SFH = _universe_to_stellar_age(
                SFH, old_age, redshift, cosmo, check_bin=True, verbose=True
            )

            # Create galaxy object
            if method == "particle":
                galaxy = _load_SCSAM_particle_galaxy(
                    SFH, age_lst, Z_lst, verbose=verbose
                )
            elif method == "parametric":
                galaxy = _load_SCSAM_parametric_galaxy(
                    SFH, age_lst, Z_lst, grid, grid_map, verbose=verbose
                )
            else:
                raise ValueError("Method not recognised.")

            # Append to list of galaxy objects
            galaxies.append(galaxy)

        count += 1

    return galaxies, halo_ind_list, birthhalo_id_list


def _load_SCSAM_particle_galaxy(SFH, age_lst, Z_lst, verbose=False):
    """
    Treat each age-Z bin as a particle.
    PSA: From what I recall, this method does not work very well.

    Args:
    SFH: age x Z SFH array as given by SC-SAM for a single galaxy
    age_lst: age bins in the SFH array (yr)
    Z_lst: metallicity bins in the SFH array (unitless)
    """

    # Initialise arrays for storing particle information
    p_imass = []  # initial mass
    p_age = []  # age
    p_Z = []  # metallicity

    # Get length of arrays
    age_len = len(age_lst)
    Z_len = len(Z_lst)

    # Iterate through every point on the grid
    if verbose:
        print("Iterating through grid...")
    for age_ind in range(age_len):
        for Z_ind in range(Z_len):
            if SFH[age_ind][Z_ind] == 0:
                continue
            else:
                p_imass.append(SFH[age_ind][Z_ind])  # Msun
                p_age.append(age_lst[age_ind])  # Gyr
                p_Z.append(Z_lst[Z_ind])  # unitless

    # Convert units
    if verbose:
        print("Converting units...")
    p_imass = np.array(p_imass)  # Msun
    p_age = np.array(p_age) * 10**9  # yr
    p_Z = np.array(p_Z)  # unitless

    if verbose:
        print("Generating SED...")

    # Create stars object
    stars = ParticleStars(
        initial_masses=p_imass * Msun, ages=p_age * yr, metallicities=p_Z
    )

    if verbose:
        print("Creating galaxy object...")
    # Create galaxy object
    particle_galaxy = ParticleGalaxy(stars=stars)

    return particle_galaxy


def _load_SCSAM_parametric_galaxy(
        SFH, age_lst, Z_lst, grid, grid_map=None, verbose=False
):
    """
    Obtain galaxy SED using the parametric method.
    This is done by interpolating the grid.
    Returns a galaxy object.

    Args:
    SFH: age x Z SFH array as given by SC-SAM for a single galaxy
    age_lst: age bins in the SFH array (Gyr)
    Z_lst: metallicity bins in the SFH array (unitless)
    method: method of interpolating the grid
            'NNI' - scipy's nearest ND interpolator
            'RGI' - scipy's regular grid interpolator
    """

    # SPS grid that we want to regrid to
    new_age = 10**grid.log10age  # yr
    new_Z = np.log10(grid.metallicity)  # log10Z
    new_age_edges, new_Z_edges = _get_sps_bin_edges(grid)
    new_grid_dim = (len(new_age), len(new_Z))

    # Original SFH grid output by SAM
    old_age = np.array(age_lst) * 10**9  # yr
    old_Z = np.log10(Z_lst)  # log10Z
    old_age_edges, old_Z_edges = _get_scsam_bin_edges(
        old_age, old_Z, verbose=True
    )

    # If the grid_map object is not specified, create it
    if grid_map==None:
        grid_map = get_grid_map(
            old_age_edges, old_Z_edges, new_age_edges,
            new_Z_edges, old_age, old_Z,
        )

    # Rebin the SFH grid
    new_SFH = rebin_grid(SFH, grid_map, new_grid_dim)

    # Create Binned SFZH object
    stars = ParametricStars(
        log10ages=grid.log10age,
        metallicities=grid.metallicity,
        sfzh=new_SFH,
    )

    # Create galaxy object
    parametric_galaxy = ParametricGalaxy(stars)

    return parametric_galaxy


def _universe_to_stellar_age(
        SFH,
        age_bins,
        redshift,
        cosmo,
        check_bins=False,
        verbose=False,
):

    """
    Convert the age axis of an SFH grid from age of the universe
    to stellar ages. Meant for SC-SAM SFHs.
    
    Args:
        SFH (array): 
            SFH array for a single galaxy (Msun). Note that the age axis
            of this SFH array should be the age of the universe (yr).
        age_bins (array): 
            Bin centres of the SFH grid age axis in age of the universe (yr)
        redshift (float): 
            Redshift of the galaxy
        cosmo (object):
            Astropy FlatLambdaCDM object. Note that the conversion from
            age of the universe to stellar ages is sensitive to cosmology.
        check_bins (bool):
            Check whether the total stellar mass is conserved?
        verbose (bool):
            Increase output verbosity
                
    Returns:
        new_SFH (array):
            SFH array with the age axis in the age of the stellar 
            particles (Msun)
        new_age_bins (array):
            Bin centres of the new age axis in stellar years (yr)
    """
 
    # Get age of universe at the redshift of the galaxy
    snapshot_age = cosmo.age(redshift).value * 1e9 # yr
    if verbose:
        print(f'The universe was {snapshot_age/(1e9)} Gyr old at z={redshift}.')
        
    # Keep the age bin centres equal to or below the age of the
    # universe at the galaxy's redshift
    keep_bins = np.where(age_bins <= snapshot_age)[0]
    
    # As we are working with bin centres rather than bin edges, it is
    # possible that stellar mass is formed in an age bin with a centre
    # that is older than the given redshift
    keep_bins = np.append(keep_bins, keep_bins[-1]+1)

    # Mask the age bin centres
    new_age_bins = age_bins[keep_bins]
    if verbose:
        print(f'Relevant bin centres in age of the universe (yr): {new_age_bins}')
  
    # Check that stellar mass is conserved
    if check_bins:
        
        # Discard age bin centres greater than the age of the universe
        # at the galaxy's redshift
        exclude_bins = np.where(age_bins > snapshot_age)[0]
        # Exclude first bin in this list, which may contain some mass
        exclude_bins = exclude_bins[1:]
        print(f'Discarded bin centres are (yr): {age_bins[exclude_bins]}')
        
        # Check if total mass is conserved
        old_sfh_mass = np.sum(np.array(SFH))
        new_sfh_mass = np.sum(np.array(SFH)[keep_bins])
        print(f'Original SFH grid total mass: {new_sfh_mass:5e} Msun')
        print(f'New SFH grid total mass: {new_sfh_mass:5e} Msun')
        if np.abs(old_sfh_mass-new_sfh_mass)<1e-10:
            print('Mass in the SFH grid is conserved!')
        else:
            raise ValueError('Total mass is not conserved.')
        
        # The sum of the discarded SFH should be equal to zero
        discarded_mass = np.sum(np.array(SFH)[exclude_bins])
        print(f'Discarded SFH grid total mass: {discarded_mass} Msun')
        if np.sum(np.array(SFH)[exclude_bins]) == 0:
            print('Discarded bins are empty (as should be)!')
        else:
            raise ValueError('Discarded bins have non-zero mass.')
        
    # Convert age of the universe to stellar age
    new_age_bins = snapshot_age - new_age_bins
    
    # Make the final age bin equal to zero - we don't want negative ages
    # Note: final bin because the stellar ages are now in descending order
    # Technically, the bin centre should not be zero, but it does not
    # really matter so long as we get the bin edges right later
    new_age_bins = np.append(new_age_bins[:-1], 0.)
    if verbose:
        print(f'Bin centres in stellar ages (yr): {new_age_bins}')

    # Mask the SFH array
    new_SFH = np.array(SFH)[keep_bins]
    if verbose:
        print(f'The new SFH has shape {SFH.shape}.')

    # Flip arrays to be in order of ascending stellar age
    new_age_bins = np.flip(new_age_bins)
    new_SFH = np.flipud(new_SFH)

    return new_SFH, new_age_bins


def _get_scsam_bin_edges(
        age_bins,
        log10metal_bins,
        verbose=True,
):

    """
    Get bin edges of the age and metallicity axes.
    Specifically for SC-SAM SFH grids, which are uniformly-spaced along
    the age axis and irregularly-spaced along the metallicity axis.

    Args:
        age_bins:
            Bin centres of the SFH grid age axis in stellar years (yr)
            Note: must be uniformly-spaced!
        log10metal_bins:
            Bin centres of the SFH grid metallicity axis in log10Z
        verbose:
            Increase output verbosity

    Returns:
        age_edges:
            Bin edges of the SFH grid age axis in stellar years (yr)
        log10metal_edges:
            Bin edges of the SFH grid metallicity axis in log10Z  
    """

    # Get stellar age bin edges
    # Make the following assumptions:
    # 1. Lowest bin edge is 0
    # 2. Second lowest bin edge is equal to the second lowest bin
    #    centre minus half the bin width
    # 3. Highest bin edge is highest bin + half the bin width
    # 4. The remaining bin edges lie equidistant between bin centres
    binw = np.array(age_bins)[-1] - np.array(age_bins)[-2]
    age_edges = np.empty(len(age_bins)+1)
    age_edges[0] = 0.
    age_edges[1] = sam_age_stars[1]-binw/2
    age_edges[2:] = np.array(sam_age_stars)[1:] + binw/2
    if verbose:
        print(f'The stellar age bin edges are: {age_edges}')

    # Get log metallicity bin edges
    # Make the following assumptions:
    # 1. Lowest bin edge is the lowest bin centre
    # 2. Highest bin edge is the highest bin centre
    # 3. The remaining bin edges lie equidistant between bin centres
    log10metal_edges = np.empty(len(log10metal_bins)+1)
    log10metal_edges[0] = log10metal_bins[0]
    log10metal_edges[-1] = log10metal_bins[-1]
    log10metal_bins[1:-1] = np.array(log10metal_bins)[:-1] + \
        np.diff(log10metal_bins)/2
    if verbose:
        print(f'The log10Z bin edges are: {log10metal_edges}')

    return age_edges, log10metal_edges


def _get_sps_bin_edges(
        grid,
):

    """
    Get bin edges of an SPS grid.

    Args:
        grid (object):
            Synthesizer grid object

    Returns:
        age_edges:
            Bin edges of the SFH grid age axis in stellar years (yr)
        log10metal_edges:
            Bin edges of the SFH grid metallicity axis in log10Z
    """

    # Get stellar age bin centres
    age_bins = 10**grid.log10age # yr

    # Get stellar age bin edges
    # Make the following assumptions:
    # 1. Lowest bin edge is 0
    # 2. Highest bin edge is the highest bin centre
    # 3. The remaining bin edges lie equidistant between bin centres
    age_edges = np.empty(len(age_bins)+1)
    age_edges[0] = 0.
    age_edges[-1] = age_bins[-1]
    age_bins[1:-1] = np.array(age_bins)[:-1] + np.diff(age_bins)/2
    if verbose:
        print(f'The stellar age bin edges are: {age_edges}')

    # Get log metallicity bin centres
    log10metal_bins = np.log10(grid.metallicity)
        
    # Get log metallicity bin edges
    # Make the following assumptions:
    # 1. Lowest bin edge is the lowest bin centre
    # 2. Highest bin edge is the highest bin centre
    # 3. The remaining bin edges lie equidistant between bin centres
    log10metal_edges = np.empty(len(log10metal_bins)+1)
    log10metal_edges[0] = log10metal_bins[0]
    log10metal_edges[-1] = log10metal_bins[-1]
    log10metal_bins[1:-1] = np.array(log10metal_bins)[:-1] + \
        np.diff(log10metal_bins)/2
    if verbose:
        print(f'The log10Z bin edges are: {log10metal_edges}')
            
    return age_edges, log10metal_edges
