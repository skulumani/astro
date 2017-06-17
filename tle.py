"""Download and handle TLEs
"""

import ephem

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

