"""This module will download kernels and setup SPICE
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
from urllib.request import urlretrieve
import spiceypy as spice

import logging

# TODO Reduce the total amount of data that needs to be downloaded for testing spiceypy
# TODO Logging for downloading of kernels
# TODO Switch to hosting kernels on Github instead of downloading directly
# TODO Better handling of errors in downloading and verifying that kernels exist

# all kernels are saved to the astro package path
cwd = os.path.realpath(os.path.dirname(__file__))
directory = 'kernels'

if not os.path.isdir(os.path.join(cwd, directory)):
    os.mkdir(os.path.join(cwd, directory))


def getKernelNameFromUrl(url):
    r"""Parse URL to get the kernel name

    kernel_name = getKernelNameFromUrl(url)

    Parameters
    ----------
    url : str
        URL for kernel (direct link)

    Returns
    -------
    kernel_name : str
        Outputs the last part of the URL - the filename of the kernel

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """ 
    return str(url.split('/')[-1])


def getPathfromUrl(url):
    r"""Create the local path given a URL

    path = getPathfromUrl(url)

    Parameters
    ----------
    url : str
        URL location for the kernel file

    Returns
    -------
    path : str
        local path for kernel

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """
    return str(os.path.join(cwd, directory, getKernelNameFromUrl(url)))


def delete_file(path, logger=logging.getLogger(__name__)):
    r"""Delete a given kernel

    delete_file(path)

    Parameters
    ----------
    path : str
        Path to the kernel. It will be deleted

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """

    if os.path.exists(path):
        os.remove(path)
        logger.info('Deleted {}'.format(path))
    else:
        logger.info('{} does not exist'.format(path))


# TODO : Create a list for each type of kernel, a list for SPK, FK, BSP, etc.
# Then have logic setup to automatically download each specific type of kernel for the object
class NearKernels(object):
    r"""SPICE Kernels for NEAR mission

    NearKernels(path, logger)

    Parameters
    ----------
    path : str
        Path to store all kernels. Defaults to CWD of astro package
    logger: logging logger object
        Logger for writing logs. If not given will be instantiated

    Notes
    -----
    This only downloads data for 2001, not the whole mission.

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------
    More data on Near is available:
    https://pdssbn.astro.umd.edu/data_sb/missions/near/index.shtml
    """

    def __init__(self, path=cwd, logger=logging.getLogger(name=__name__)):
        r"""Instantiate the NearKernel object

        Notes
        -----
        This holds all the links and paths to the NEAR kernels.
        Also will create a metakernel which can be loaded by SPICE

        Author
        ------
        Shankar Kulumani		GWU		skulumani@gwu.edu

        """
        self.logger = logger
        self.logger.debug('Instantiating NearKernels')

        self.near_id = '-93'
        self.eros_id = '2000433'
        
        self.near_body_frame = 'NEAR_SC_BUS_PRIME'
        self.near_body_frame_id = -93000
        self.near_msi_frame = 'NEAR_MSI'
        self.near_msi_frame_id = -93001
        self.eros_body_frame = 'IAU_EROS'
        self.eros_body_frame_id = 2000433

        self.inertial_frame = 'J2000'

        # mission start from near_171.tsc in UTC
        self.start_et = -122138129.0388301
        self.start_utc = '1996-02-17 20:43:28.775 UTC'

        self.Lsk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0008.tls'
        self.Ck_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ck/near_20010101_20010228_v01.bc'
        self.Sclk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/sclk/near_171.tsc'

        self.PckEros1_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/erosatt_1998329_2001157_v01.bpc'
        self.PckEros2_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/erosatt_1999304_2001151.bpc'
        self.Pck3_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/pck00010.tpc'

        self.Fk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/fk/eros_fixed.tf'

        self.Ikgrs_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/grs12.ti'
        self.Ikmsi_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/msi15.ti'
        self.Iknis_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/nis14.ti'
        self.Iknlr_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/nlr04.ti'
        self.Ikxrs_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/xrs12.ti'

        self.SpkPlanet_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/de403s.bsp'
        self.SpkEros_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/eros80.bsp'
        self.SpkEros2_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/erosephem_1999004_2002181.bsp'
        self.SpkMath_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/math9749.bsp'
        self.SpkNearLanded_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/near_eroslanded_nav_v1.bsp'
        self.SpkNearOrbit_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/near_erosorbit_nav_v1.bsp'
        self.SpkStations_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/stations.bsp'
        
        # get the path for each kernel
        self.logger.info('Now converting all the URLs to a local path by parsing the URLs')

        self.urlList = [self.Lsk_url, self.Ck_url, self.Sclk_url, self.Pck3_url,
                        self.PckEros1_url, self.PckEros2_url, self.Fk_url,
                        self.Ikgrs_url, self.Ikmsi_url, self.Iknis_url, self.Iknlr_url,
                        self.Ikxrs_url, self.SpkPlanet_url, self.SpkEros_url,
                        self.SpkEros2_url, self.SpkMath_url, self.SpkNearLanded_url,
                        self.SpkNearOrbit_url, self.SpkStations_url]

        self.Lsk = getPathfromUrl(self.Lsk_url)
        self.Ck = getPathfromUrl(self.Ck_url)
        self.Sclk = getPathfromUrl(self.Sclk_url)

        self.PckEros1 = getPathfromUrl(self.PckEros1_url)
        self.PckEros2 = getPathfromUrl(self.PckEros2_url)
        self.Pck3 = getPathfromUrl(self.Pck3_url)

        self.Fk = getPathfromUrl(self.Fk_url)

        self.Ikgrs = getPathfromUrl(self.Ikgrs_url)
        self.Ikmsi = getPathfromUrl(self.Ikmsi_url)
        self.Iknis = getPathfromUrl(self.Iknis_url)
        self.Iknlr = getPathfromUrl(self.Iknlr_url)
        self.Ikxrs = getPathfromUrl(self.Ikxrs_url)

        self.SpkPlanet = getPathfromUrl(self.SpkPlanet_url)
        self.SpkEros = getPathfromUrl(self.SpkEros_url)
        self.SpkEros2 = getPathfromUrl(self.SpkEros2_url)
        self.SpkMath = getPathfromUrl(self.SpkMath_url)
        self.SpkNearLanded = getPathfromUrl(self.SpkNearLanded_url)
        self.SpkNearOrbit = getPathfromUrl(self.SpkNearOrbit_url)
        self.SpkStations = getPathfromUrl(self.SpkStations_url)


        self.kernelList = [self.Lsk, self.Ck, self.Sclk, self.Pck3, self.PckEros1,
                           self.PckEros2, self.Fk, self.Ikgrs, self.Ikmsi, self.Iknis,
                           self.Iknlr, self.Ikxrs, self.SpkPlanet, self.SpkEros,
                           self.SpkEros2, self.SpkMath, self.SpkNearLanded,
                           self.SpkNearOrbit, self.SpkStations]

        self.nameList = [getKernelNameFromUrl(url) for url in self.urlList]
        self.kernelDescription = 'Metal Kernel for 2001 NEAR orbit and landing'

        self.logger.info('Starting the download for NEAR kernels')
        getKernels(self, os.path.realpath(path))

        self.logger.info('Creating a NEAR metakernel')
        self.metakernel = writeMetaKernel(self, 'near2001.tm')

        self.info()

    def info(self):
        """Read and Output info about the loaded kernels
        """
        spice.kclear()

        spice.furnsh(self.metakernel)

        self.spkList = [self.SpkPlanet, self.SpkEros, self.SpkEros2,
                        self.SpkMath, self.SpkNearLanded, self.SpkNearOrbit,
                        self.SpkStations]

        self.ckList = [self.Ck]
        self.pckList = [self.PckEros1, self.PckEros2]

        # check SPK coverage
        self.bodies = {}
        self.bodies_coverage = {}
        for spk in self.spkList:
            idcell = spice.spkobj(spk)
            for code in idcell:
                cover = spice.stypes.SPICEDOUBLE_CELL(1000)
                spice.spkcov(spk, code, cover)
                self.bodies[str(code)] = spice.bodc2n(code)
                self.bodies[spice.bodc2n(code)] = code
                self.bodies_coverage[str(code)] = [x for x in cover]
                self.bodies_coverage[spice.bodc2n(code)] = [x for x in cover]

        # check CK coverage
        self.ckframes = {}
        self.ckframes_coverage = {}
        for ck in self.ckList:
            idcell = spice.ckobj(ck)
            for code in idcell:
                cover = spice.ckcov(ck, code, True, 'SEGMENT', 0.0, 'TDB')
                self.ckframes[str(code)] = spice.frmnam(code)
                self.ckframes[spice.frmnam(code)] = code
                self.ckframes_coverage[str(code)] = [x for x in cover]
                self.ckframes_coverage[spice.frmnam(code)] = [x for x in cover]

        # check pck coverage
        self.pckframes = {}
        for pck in self.pckList:
            ids = spice.stypes.SPICEINT_CELL(1000)
            spice.pckfrm(pck, ids)
            for code in ids:
                self.pckframes[str(code)] = spice.frmnam(code)
                self.pckframes[spice.frmnam(code)] = code

        spice.kclear()


class CassiniKernels(object):
    """List of urls and kernels for the Cassini mission

    More data on Cassini is available:
    https://naif.jpl.nasa.gov/pub/naif/CASSINI/
    """

    def __init__(self, path=cwd):
        """Initialize the Cassini kernel class
        """
        self.Lsk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0011.tls'
        self.Sclk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/sclk/cas00171.tsc'
        self.Pck_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/pck/cpck09May2017.tpc'
        self.Fk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/fk/release.11/cas_v37.tf'
        self.Ck_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ck/04135_04171pc_psiv2.bc'
        self.Spk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/981005_PLTEPH-DE405S.bsp'
        self.Ik_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/release.11/cas_iss_v09.ti'
        self.TourSpk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/030201AP_SK_SM546_T45.bsp'
        self.satSpk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/020514_SE_SAT105.bsp'

        self.Lsk = getPathfromUrl(self.Lsk_url)
        self.Sclk = getPathfromUrl(self.Sclk_url)
        self.Pck = getPathfromUrl(self.Pck_url)
        self.Fk = getPathfromUrl(self.Fk_url)
        self.Ck = getPathfromUrl(self.Ck_url)
        self.Spk = getPathfromUrl(self.Spk_url)
        self.Ik = getPathfromUrl(self.Ik_url)
        self.TourSpk = getPathfromUrl(self.TourSpk_url)
        self.satSpk = getPathfromUrl(self.satSpk_url)

        self.urlList = [self.Lsk_url, self.Sclk_url, self.Pck_url, self.Fk_url,
                        self.Ck_url, self.Spk_url, self.Ik_url, self.TourSpk_url,
                        self.satSpk_url]
        self.kernelList = [self.Lsk, self.Sclk, self.Pck, self.Fk, self.Ck,
                           self.Spk, self.Ik, self.TourSpk, self.satSpk]
        self.nameList = [getKernelNameFromUrl(url) for url in self.urlList]

        self.kernelDescription = 'Metal Kernel for Cassini Orbiter'
        getKernels(self, os.path.realpath(path))
        self.metakernel = writeMetaKernel(self, 'cassini.tm')


def cleanupKernels(kernelObj=CassiniKernels):
    """Delete all the Kernels
    """
    for kernel in kernelObj.kernelList:
        delete_file(kernel)

def attemptDownload(url, kernelName, targetFileName, num_attempts=5, logger=logging.getLogger(__name__)):
    """Download the file from a specific url
    """
    current_attempt = 0
    while current_attempt < num_attempts:
        try:
            logger.info("Attempting to download kernel: {}".format(kernelName))
            urlretrieve(url, targetFileName)
            break
        except: # TODO Catch URL exception ehre - just a warning also catch a real error if the download fails log it both
            pass

        current_attempt = current_attempt + 1
        logger.info("Attempting to download kernel again...")
        time.sleep(2 + current_attempt)

    if current_attempt >= num_attempts:
        # TODO Add an exception here for a real failure
        logger.error("Error downloading kernel: {}. Check if it exists at url: {}".format(
            kernelName, url))


def getKernels(kernelObj=CassiniKernels, path=cwd):
    """Download all the Kernels
    """

    for url in kernelObj.urlList:
        kernelName = getKernelNameFromUrl(url)
        kernelFile = os.path.join(path, directory, kernelName)

        if not os.path.isfile(kernelFile):
            attemptDownload(url, kernelName, kernelFile, 5)

    return 0


def writeMetaKernel(kernelObj, filename='testKernel.tm'):
    """Write a user defined meta kernel file
    """
    with open(os.path.join(cwd, directory, filename), 'w') as metaKernel:
        metaKernel.write('\\begintext\n\n')
        metaKernel.write('Created: Shankar Kulumani\n')
        if kernelObj.kernelDescription:
            metaKernel.write('Description: {}\n'.format(
                kernelObj.kernelDescription))

        metaKernel.write('\n')

        metaKernel.write('\\begindata\n\n')
        metaKernel.write('PATH_VALUES = (\n')
        metaKernel.write('\'{0}\'\n'.format(os.path.join(cwd, directory)))
        metaKernel.write(')\n\n')

        metaKernel.write('PATH_SYMBOLS = (\n')
        metaKernel.write('\'KERNELS\'\n')
        metaKernel.write(')\n\n')

        metaKernel.write('KERNELS_TO_LOAD = (\n')
        for kernel in kernelObj.nameList:
            metaKernel.write('\'$KERNELS/{0}\'\n'.format(kernel))

        metaKernel.write(')\n')
        metaKernel.close()

    return os.path.join(cwd, directory, filename)


if __name__ == '__main__':
    pass
