import numpy as np


def get_grid_map(
        old_age_edges,
        old_metal_edges,
        new_age_edges,
        new_metal_edges,
        old_age_centres=None,
        old_metal_centres=None,
):
    
    """
    Obtain a mapping to map mass from one SFH grid (old) to another (new).
    Typically, the old grid would be the SAM SFH grid and the new grid
    would be the SPS grid.

    Args:
        old_age_edges (array):
            Bin edges of the old SFH grid stellar age axis
        old_metal_edges (array):
            Bin edges of the old SFH grid metallicity axis
        new_age_edges (array):
            Bin edges of the new SFH grid stellar age axis
        new_metal_edges (array):
            Bin edges of the new SFH grid metallicity axis
        old_age_centres (array):
            Bin centres of the old SFH grid stellar age axis
            Optional - saved in the output dictionary for reference.
        old_metal_centres (array):
            Bin centres of the old SFH grid metallicity axis
            Optional - saved in the output dictionary for reference.

    Returns:
        map_dict (dict):
            Dictionary that contains information for mapping the old,
            original SFH grid to the desired new grid.
    """
            
    # Get lengths
    old_age_lengths = np.diff(old_age_edges)
    old_metal_lengths = np.diff(old_metal_edges)
    new_age_lengths = np.diff(new_age_edges)
    new_metal_lengths = np.diff(new_metal_edges)

    # Create dictionary to store mapping
    map_dict = {}
    
    # Loop over each bin on the metallicity axis of the old grid
    map_dict['metal_bin_index'] = {}
    for ii in range(len(old_metal_lengths)):

        map_dict['metal_bin_index'][ii] = {}

        # Get the upper and lower bound of this bin
        old_low = old_metal_edges[ii]
        old_up = old_metal_edges[ii+1]
        old_length = old_metal_lengths[ii]

        # To find overlapping bins in the new grid, identify bins in
        # the new grid with:
        # 1. A lower bound that falls below the upper bound of the old bin
        # 2. An upper bound that lies above the lower bound of the old bin
        ok_low = (new_metal_edges[:-1] < old_up)
        ok_high = (new_metal_edges[1:] > old_low)
        ok_ind = np.where(ok_low & ok_high)[0]

        # Along the metallicity axis, get the fraction of the old bin's
        # width that is occupied by each overlapping new bin
        # If there is only one bin in the new grid that overlaps, then
        # the fraction is simply 1
        if len(ok_ind)==1:
            frac = [1]
        # If there are at at least two bins in the new grid that overlap,
        # calculate the overlap of the two end bins that bracket the old bin 
        elif len(ok_ind)>=2:
            frac = np.ones(len(ok_ind))
            # Find the overlapping length of the lower bin 
            l1 = new_metal_edges[ok_ind[0]+1] - old_low
            l1_frac = l1/old_length
            frac[0] = l1_frac
            #  Find the overlapping length of the higher bin
            l2 = old_up - new_metal_edges[ok_ind[-1]]
            l2_frac = l2/old_length
            frac[-1] = l2_frac
            # Calculate the overlap of any remaining bins
            if len(ok_ind)>2:
                for iii, ind in enumerate(ok_ind[1:-1]):
                    frac[1+iii] = new_metal_lengths[ind]/old_length
        # If no bin in the new grid overlaps with this old bin...
        elif len(ok_ind)==0:
            # Could be that the old grid goes to higher values than the new grid
            if np.sum(ok_high)==0:
                # In this case, move all mass to highest bin in the new grid
                ok_ind = [len(new_metal_lengths)-1]
                frac = [1]
            # Or could be the old grid goes to lower values than the new grid
            elif np.sum(ok_low)==0:
                # In this case, move all mass to lowest bin in the new grid
                ok_ind = [0]
                frac = [1]
            else:
                print('There is an edge case that has not been considered.')
                break
        else:
            print('Something is very wrong.')
            break

        # print('sum of frac:', np.sum(frac))

        # Store values
        map_dict['metal_bin_index'][ii]['new_grid_indices'] = np.array(ok_ind)
        map_dict['metal_bin_index'][ii]['fraction'] = np.array(frac)
        if (old_metal_centres!=None).all():
            map_dict['metal_bin_index'][ii]['old_bin_value'] = old_metal_centres[ii]

    # Loop over each bin on the age axis of the old grid
    # Exactly the same procedure as before. To do: rewrite to loop over axes
    map_dict['age_bin_index'] = {}
    for ii in range(len(old_age_lengths)):

        map_dict['age_bin_index'][ii] = {}

        # Get the upper and lower bound of this bin
        old_low = old_age_edges[ii]
        old_up = old_age_edges[ii+1]
        old_length = old_age_lengths[ii]

        # To find overlapping bins in the new grid, identify bins in
        # the new grid with:
        # 1. A lower bound that falls below the upper bound of the old bin
        # 2. An upper bound that lies above the lower bound of the old bin
        ok_low = (new_age_edges[:-1] < old_up)
        ok_high = (new_age_edges[1:] > old_low)
        ok_ind = np.where(ok_low & ok_high)[0]

        # Along the age axis, get the fraction of the old bin's
        # width that is occupied by each overlapping new bin
        # If there is only one bin in the new grid that overlaps, then
        # the fraction is simply 1
        if len(ok_ind)==1:
            frac = [1]
        # If there are at at least two bins in the new grid that overlap,
        # calculate the overlap of the two end bins that bracket the old bin 
        elif len(ok_ind)>=2:
            frac = np.ones(len(ok_ind))
            # Find the overlapping length of the lower bin 
            l1 = new_age_edges[ok_ind[0]+1] - old_low
            l1_frac = l1/old_length
            frac[0] = l1_frac
            #  Find the overlapping length of the higher bin
            l2 = old_up - new_age_edges[ok_ind[-1]]
            l2_frac = l2/old_length
            frac[-1] = l2_frac
            # Calculate the overlap of any remaining bins
            if len(ok_ind)>2:
                for iii, ind in enumerate(ok_ind[1:-1]):
                    frac[1+iii] = new_age_lengths[ind]/old_length
        # If no bin in the new grid overlaps with this old bin...
        elif len(ok_ind)==0:
            # Could be that the old grid goes to higher values than the new grid
            if np.sum(ok_high)==0:
                # In this case, move all mass to highest bin in the new grid
                ok_ind = [len(new_age_lengths)-1]
                frac = [1]
            # Or could be the old grid goes to lower values than the new grid
            elif np.sum(ok_low)==0:
                # In this case, move all mass to lowest bin in the new grid
                ok_ind = [0]
                frac = [1]
            else:
                print('There is an edge case that has not been considered.')
                break
        else:
            print('Something is very wrong.')
            break

        # print('sum of frac:', np.sum(frac))

        # Store values
        map_dict['age_bin_index'][ii]['new_grid_indices'] = np.array(ok_ind)
        map_dict['age_bin_index'][ii]['fraction'] = np.array(frac)
        if (old_age_centres!=None).all():
            map_dict['age_bin_index'][ii]['old_bin_value'] = old_age_centres[ii]
        
    return map_dict


def rebin_grid(
        SFH,
        grid_map,
        new_grid_dim,
):
    
    """
    Rebin a parametric SFH grid from one set of age-metallicity bin
    centres to another.

    Args:
        SFH (array):
            Original SFH grid to be rebinned.
        grid_map (dict):
            Dictionary that contains information for mapping the old,
            original SFH grid to the desired new grid. Obtained from
            the get_grid_map function in this script.
        new_grid_dim (array):
            Dimensions of the new grid

    Returns:
       new_SFH
    """
        
    # Empty SPS grid, to be populated
    new_SFH = np.zeros(new_grid_dim)

    # Get i,j coordinates of old SFH
    xx = np.arange(0, np.array(SFH).shape[0], 1)
    yy = np.arange(0, np.array(SFH).shape[1], 1)
    iis, jjs = np.meshgrid(xx, yy, indexing='ij')

    # Only perform mass distribution for bins with non-zero mass
    keep = SFH > 0.
    iis = iis[keep].flatten()
    jjs = jjs[keep].flatten()
        
    # Loop over each bin in the old grid with non-zero mass
    for (i, j) in zip(iis, jjs):

        # Metallicity axis
        new_metal_ind = grid_map['metal_bin_index'][j]['new_grid_indices']
        new_metal_frac = grid_map['metal_bin_index'][j]['fraction']

        # Age axis
        new_age_ind = grid_map['age_bin_index'][i]['new_grid_indices']
        new_age_frac = grid_map[f'age_bin_index'][i]['fraction']

        # Loop over each relevant SPS bin
        for (new_i, frac_i) in zip(new_age_ind, new_age_frac):
            for (new_j, frac_j) in zip(new_metal_ind, new_metal_frac):

                # print(new_i, new_j)

                frac = frac_i * frac_j
                new_SFH[new_i,new_j] += SFH[i,j] * frac
        
    return new_SFH
