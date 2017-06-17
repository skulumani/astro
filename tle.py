"""Download and handle TLEs
"""
import numpy as np
import ephem
import pdb
from collections import defaultdict, namedtuple

# tuple to hold all the items from a single TLE 
TLE = namedtuple('TLE', [
    'satname', 'satnum', 'classification', 'id_year', 'id_launch',
    'id_piece', 'epoch_year', 'epoch_day', 'ndot_over_2', 'nddot_over_6',
    'bstar', 'ephtype', 'elnum', 'checksum1',
    'inc', 'raan', 'ecc', 'argp', 'ma', 'mean_motion', 'epoch_rev',
    'checksum2'])

def get_tle_ephem(filename):
    """Load TLEs from a file
    """
    satlist = []
    with open(filename,'r') as f:
        l1 = f.readline()
        while l1:
            l2 = f.readline()
            l3 = f.readline()
            sat = ephem.readtle(l1, l2, l3)
            satlist.append(sat)
            print(sat.name)
            l1 = f.readline()

    print("{} satellites loaded into list".format(len(satlist)))
    return satlist

# write my own TLE parser since ephem doesn't save all the values

def validtle(l0, l1, l2):
    """Quick check to make sure we have the three lines of a 3TLE
    """
    l0_valid_start = int(l0[0]) == 0
    l1_valid_start = int(l1[0]) == 1
    l2_valid_start = int(l2[0]) == 2

    l1_valid_length = len(l1) == 69
    l2_valid_length = len(l2) == 69

    l1_valid_checksum = int(l1[-1]) == checksum(l1)
    l2_valid_checksum = int(l2[-1]) == checksum(l2)
    return (l0_valid_start and l1_valid_start and l2_valid_start 
            and l1_valid_length and l2_valid_length and 
            l1_valid_checksum and l1_valid_checksum)

def checksum(line):
    """The checksums for each line are calculated by adding the all numerical
    digits on that line, including the line number. One is added to the
    checksum for each negative sign (-) on that line. All other non-digit
    characters are ignored.  @note this excludes last char for the checksum
    thats already there.
    """
    return sum(map(int, filter(
        lambda c: c >= '0' and c <= '9', line[:-1].replace('-','1')))) % 10

def stringScientificNotationToFloat(sn):
    """Specific format is 5 digits, a + or -, and 1 digit, ex: 01234-5 which is
    0.01234e-5
    """
    return 1e-5*float(sn[:6]) * 10**int(sn[6:]) 

def parsetle(l0, l1, l2):
    """Get all the elements out of a TLE
    """
    # parse line zero 
    satname = l0[0:23]
    
    # parse line one
    satnum = int(l1[2:7])
    classification = l1[7:8]
    id_year = int(l1[9:11])
    id_launch = int(l1[11:14])
    id_piece = l1[14:17]
    epoch_year = int(l1[18:20])
    epoch_day = float(l1[20:32])
    ndot_over_2 = float(l1[33:43])
    nddot_over_6 = stringScientificNotationToFloat(l1[44:52])
    bstar = stringScientificNotationToFloat(l1[53:61])
    ephtype = int(l1[62:63])
    elnum = int(l1[64:68])
    checksum1 = int(l1[68:69])

    # parse line 2
    # satellite        = int(line2[2:7])
    inc = float(l2[8:16])
    raan = float(l2[17:25])
    ecc = float(l2[26:33]) * 0.0000001
    argp = float(l2[34:42])
    ma = float(l2[43:51])
    mean_motion = float(l2[52:63])
    epoch_rev = float(l2[63:68])
    checksum2 = float(l2[68:69])
    
    return TLE(satnum=satnum, classification=classification, id_year=id_year,
            id_launch=id_launch, id_piece=id_piece, epoch_year=epoch_year,
            epoch_day=epoch_day, ndot_over_2=ndot_over_2, 
            nddot_over_6=nddot_over_6, bstar=bstar, ephtype=ephtype, 
            elnum=elnum, checksum1=checksum1, inc=inc, raan=raan, ecc=ecc,
            argp=argp, ma=ma, mean_motion=mean_motion, epoch_rev=epoch_rev,
            checksum2=checksum2,
            satname=satname)

def get_tle(filename):
    """Assuming a file with 3 Line TLEs is given this will parse the file
    and save all the elements to a list or something. 

    Only reading and error checking is done in this function, other
    transformations can happen later
    """
    with open(filename, 'r') as f:
        l0 = f.readline().strip()
        while l0:
            l1 = f.readline().strip()
            l2 = f.readline().strip()
            # error check to make sure we have read a complete TLE
            if validtle(l0, l1, l2):
                # now we parse the tle
                elements = parsetle(l0, l1, l2)
            else:
                print("Invalid TLE")

            l0 = f.readline().strip()

