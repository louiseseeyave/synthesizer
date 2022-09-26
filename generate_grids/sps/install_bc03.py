"""
Download BC03 and convert to HDF5 synthesizer grid.
"""

import numpy as np
import os
import sys
import re
import wget
from utils import write_data_h5py, write_attribute
import tarfile
import glob
import gzip
import shutil



model_name = 'bc03_chabrier03'


def download_data():

    url = 'http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_2000_chabrier_imf.tar.gz'
    filename = wget.download(url)
    return filename

def untar_data():


    input_dir = f'{synthesizer_data_dir}/input_files/'
    fn = 'bc03.models.padova_2000_chabrier_imf.tar.gz'

    # --- untar main directory
    tar = tarfile.open(fn)
    tar.extractall(path = input_dir)
    tar.close()
    os.remove(fn)

    # --- unzip the individual files that need reading
    model_dir = f'{synthesizer_data_dir}/input_files/bc03/models/Padova2000/chabrier'
    files = glob.glob(f'{model_dir}/bc2003_hr_m*_chab_ssp.ised_ASCII.gz')

    for file in files:
        with gzip.open(file, 'rb') as f_in:
            with open('.'.join(file.split('.')[:-1]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)





def readBC03Array(file, lastLineFloat=None):
    """Read a record from bc03 ascii file. The record starts with the
       number of elements N and is followed by N numbers. The record may
       or may not start within a line, i.e. a line need not necessarily
       start with a record.
    Parameters:
    ----------------------------------------------------------------------
    file: handle on open bc03 ascii file
    lastLineFloat: still open line from last line read, in case of a
                   record starting mid-line.
    Returns array, lastLine, where:
    ----------------------------------------------------------------------
    array = The array values read from the file
    lastLine = The remainder of the last line read (in floating format),
               for continued reading of the file
    """

    if lastLineFloat is None or len(lastLineFloat) == 0:
        # Nothing in last line, so read next line
        line = file.readline()
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]
    # Read array 'header' (i.e. number of elements)
    arrayCount = int(lastLineFloat[0])    # Length of returned array
    array = np.empty(arrayCount)          # Initialise the array
    lastLineFloat = lastLineFloat[1:len(lastLineFloat)]
    iA = 0                                # Running array index
    while True:                           # Read numbers until array is full
        for iL in range(0, len(lastLineFloat)):  # Loop numbers in line
            array[iA] = lastLineFloat[iL]
            iA = iA+1
            if iA >= arrayCount:                # Array is full so return
                return array, lastLineFloat[iL+1:]
        line = file.readline()   # Went through the line so get the next one
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]


def convertBC03(files=None):
    """Convert BC03 outputs

    Parameters (user will be prompted for those if not present):
    ----------------------------------------------------------------------
    files: list of each BC03 SED ascii file, typically named
           bc2003_xr_mxx_xxxx_ssp.ised_ASCII
    """

    # Prompt user for files if not provided--------------------
    if files is None:
        print('Please write the model to read',)
        files = []
        while True:
            filename = input('filename >')
            if filename == '':
                break
            files.append(filename)
        print('ok checking now')
        if not len(files):
            print('No BC03 files given, nothing do to')
            return

    # Initialise ---------------------------------------------------------
    ageBins = None
    lambdaBins = None
    metalBins = [None] * len(files)
    seds = np.array([[[None]]])

    print('Reading BC03 files and converting...')
    # Loop SED tables for different metallicities
    for iFile, fileName in enumerate(files):
        print('Converting file ', fileName)
        file = open(fileName, 'r')
        # file = gzip.open(f'{fileName}.gz', 'rb')


        ages, lastLine = readBC03Array(file)  # Read age bins
        nAge = len(ages)
        print("Number of ages: %s" % nAge)
        if ageBins is None:
            ageBins = ages
            seds.resize((seds.shape[0], len(ageBins),
                         seds.shape[2]), refcheck=False)
        if not np.array_equal(ages, ageBins):  # check for consistency
            print('Age bins are not identical everywhere!!!')
            print('CANCELLING CONVERSION!!!')
            return
        # Read four (five ?) useless lines
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        # These last three lines are identical and contain the metallicity
        ZZ, = re.search('Z=([0-9]+\.?[0-9]*)', line).groups()
        metalBins[iFile] = eval(ZZ)
        seds.resize(
            (len(metalBins), seds.shape[1], seds.shape[2]), refcheck=False)
        # Read wavelength bins
        lambdas, lastLine = readBC03Array(file, lastLineFloat=lastLine)
        if lambdaBins is None:  # Write wavelengths to sed file
            lambdaBins = lambdas
            seds.resize((seds.shape[0], seds.shape[1],
                         len(lambdaBins)), refcheck=False)
        if not np.array_equal(lambdas, lambdaBins):  # check for consistency
            print('Wavelength bins are not identical everywhere!!!')
            print('CANCELLING CONVERSION!!!')
            return
        # Read luminosities
        for iAge in range(0, nAge):
            lums, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            if len(lums) != len(lambdaBins):
                print('Inconsistent number of wavelength bins in BC03')
                print('STOPPING!!')
                return
            # Read useless array
            tmp, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            seds[iFile, iAge] = lums
            progress = (iAge + 1) / nAge
            sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format(
                '#' * int(progress * 50), progress * 100))
        print(' ')
        lastLine = None


    return (np.array(seds, dtype=np.float64),
            np.array(metalBins, dtype=np.float64),
            np.array(ageBins, dtype=np.float64),
            np.array(lambdaBins, dtype=np.float64))





def make_grid():
    """ Main function to convert BC03 grids and
        produce grids used by synthesizer """

    # Define base path
    basepath = synthesizer_data_dir+"/input_files/bc03/models/Padova2000/chabrier/"

    # Define output
    fname = f'{synthesizer_data_dir}/grids/{model_name}.h5'

    # Define files
    files = ['bc2003_hr_m122_chab_ssp.ised_ASCII',
             'bc2003_hr_m132_chab_ssp.ised_ASCII',
             'bc2003_hr_m142_chab_ssp.ised_ASCII',
             'bc2003_hr_m152_chab_ssp.ised_ASCII',
             'bc2003_hr_m162_chab_ssp.ised_ASCII',
             'bc2003_hr_m172_chab_ssp.ised_ASCII']

    out = convertBC03([basepath + s for s in files])

    zsol = 0.0127

                    # Lsol / AA / Msol



    metallicities = out[1]
    log10metallicities = np.log10(metallicities)

    ages = out[2]
    log10ages = np.log10(ages)

    lam = out[3]
    nu = 3E8/(lam*1E-10)

    spec = out[0]

    spec = np.swapaxes(spec, 0,1) # make (age, metallicity, wavelength)

    print(spec.shape)
    print(metallicities.shape)
    print(ages.shape)



    spec *= (3.826e33)  # erg s^-1 AA^-1 Msol^-1
    spec *= lam/nu # erg s^-1 Hz^-1 Msol^-1


    write_data_h5py(fname, 'spectra/wavelength', data=lam, overwrite=True)
    write_attribute(fname, 'spectra/wavelength', 'Description',
            'Wavelength of the spectra grid')
    write_attribute(fname, 'spectra/wavelength', 'Units', 'AA')

    write_data_h5py(fname, 'ages', data=ages, overwrite=True)
    write_attribute(fname, 'ages', 'Description',
            'Stellar population ages years')
    write_attribute(fname, 'ages', 'Units', 'yr')

    write_data_h5py(fname, 'log10ages', data=log10ages, overwrite=True)
    write_attribute(fname, 'log10ages', 'Description',
            'Stellar population ages in log10 years')
    write_attribute(fname, 'log10ages', 'Units', 'log10(yr)')

    write_data_h5py(fname, 'metallicities', data=metallicities, overwrite=True)
    write_attribute(fname, 'metallicities', 'Description',
            'raw abundances')
    write_attribute(fname, 'metallicities', 'Units', 'dimensionless [Z]')

    write_data_h5py(fname, 'log10metallicities', data=log10metallicities, overwrite=True)
    write_attribute(fname, 'log10metallicities', 'Description',
            'raw abundances in log10')
    write_attribute(fname, 'log10metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(fname, 'spectra/stellar', data=spec, overwrite=True)
    write_attribute(fname, 'spectra/stellar', 'Description',
                    'Three-dimensional spectra grid, [age, metallicity, wavelength]')
    write_attribute(fname, 'spectra/stellar', 'Units', 'erg s^-1 Hz^-1')

# Lets include a way to call this script not via an entry point
if __name__ == "__main__":

    synthesizer_data_dir = os.getenv('SYNTHESIZER_DATA')

    # download_data()
    # untar_data()
    make_grid()

    # filename = f'{synthesizer_data_dir}/grids/{model_name}.h5'
    # add_log10Q(filename)