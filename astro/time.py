"""Time

This module contains several useful time conversion functions.

Notes
-----
You can use this module by importing it into your code

from astro import time

Attributes
----------
No module level attributes

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from kinematics import attitude

# TODO Setup logging
def date2jd(yr, mon, day, hr, minute, sec):
    """Convert date to Julian Date

    Purpose:
        - Converts UTC date to julian date valid 1 Mar 1900 to 28 Feb 2100

    JD = date2jd(yr, mon, day, hr, min, sec)

    Inputs:
        - yr - 4 digit year (between 1900 and 2100)
        - mon - month (between 01 and 12)
        - day - day (between 1 and 31)
        - hr - UT hour (between 0 and 23)
        - min - UT min (between 0 and 59)
        - sec - UT sec (between 0 and 59.999)

    Outputs:
        - JD - julian date (days from 1 Jan 4713 BC 12 Noon)
        - MJD - modified julian date

    Dependencies:
        - none

    Author:
        - Shankar Kulumani 21 Oct 2012
            - list revisions
        - Shankar Kulumani 3 Dec 2016
            - convert to python

    References
        - Vallado
        - USAFA Astro 321
    """

    JD = 367.0 * yr - np.floor((7 * (yr + np.floor((mon + 9) / 12.0))) *
                               0.25) + np.floor(275 * mon / 9.0) + day + 1721013.5 + (
        (sec / 60.0 + minute) / 60.0 + hr) / 24.0

    MJD = JD - 2400000.5

    return (JD, MJD)


def dayofyr2mdhms(yr, days):
    """This function converts the day of the year, days, to the month
        day, hour, minute and second.

    Algorithm     : Set up array for the number of days per month
                    loop through a temp value while the value is < the days
                    Perform integer conversions to the correct day and month
                    Convert remainder into H M S using type conversions

    Author        : Capt Dave Vallado  USAFA/DFAS     719-472-4109  26 Feb 1990
    In Ada        : Dr Ron Lisowski    USAFA/DFAS     719-472-4110  17 May 1995
    In MATLAB     : LtCol Thomas L. Yoder USAFA/DFAS  719-472-4110  Spring 00
    In Python     : Shankar Kulumani   GWU            630-336-6257  2017 06 15

    Inputs        :
        Yr          - Year                                 1900 .. 2100
        Days        - Julian Day of the year               1.0  .. 366.0

    OutPuts       :
        Mon         - Month                                   1 .. 12
        D           - Day                                     1 .. 28,29,30,31
        H           - Hour                                    0 .. 23
        M           - Minute                                  0 .. 59
        Sec         - Second                                0.0 .. 59.999

    Locals        :
        DayOfyr     - Day of year
        Temp        - Temporary Long_Float value
        IntTemp     - Temporary 16 bit Integer value
        i           - Index

    Constants     :
        LMonth(12)  - Integer Array containing the number of days per month

    Coupling      : finddays is the inverse

    References    :
        None.
    """
    dayofyr = np.fix(days)

    # find month and day of month
    lmonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (yr - 1900) % 4 == 0:
        lmonth[1] = 29

    ii = 0
    temp = 0
    while (dayofyr > temp + lmonth[ii]) and (ii < 11):
        temp += lmonth[ii]
        ii += 1

    mon = ii + 1
    day = dayofyr - temp

    # find hour, min, and seconds
    temp = (days - dayofyr) * 24
    hour = np.fix(temp) - 0
    temp = (temp - hour) * 60.0
    minute = np.fix(temp) - 0
    sec = (temp - minute) * 60.0

    return (mon, day, hour, minute, sec)


def jd2date(jd):
    """This function finds the Year, month, day, hour, minute and second
    given the Julian date.

    Algorithm     : Set up starting values
                    Find the elapsed days through the year in a loop
                    Call routine to find each individual value

    Author        : Capt Dave Vallado  USAFA/DFAS  719-472-4109  26 Feb 1990
    In Ada        : Dr Ron Lisowski    USAFA/DFAS  719-472-4110  17 May 1995
    In Matlab     : LtCol Thomas Yoder USAFA/DFAS  719-333-4110  Spring 2000
    In Python     : Shankar Kulumani   GWU         630-336-6257  2017 06 15

    Inputs        :
        JD          - Julian Date                          days from 4713 B.C.

    OutPuts       :
        Yr          - Year                                 1900 .. 2100
        Mon         - Month                                   1 .. 12
        D           - Day                                     1 .. 28,29,30,31
        H           - Hour                                    0 .. 23
        M           - Minute                                  0 .. 59
        S           - Second                                0.0 .. 59.999

    Locals        :
        days        - Day of year plus fraction of a day   days
        Tu          - Julian Centuries from 1 Jan 1900
        Temp        - Temporary Long_Float value
        LeapYrs     - Number of Leap years from 1900

    Constants     :
        None.

    Coupling      :
        DayofYr2MDHMS  - Finds Month, day, hour, minute and second given Days
        and Yr

    References    :
        1988 Almanac for Computers  pg. B2
        Escobal       pg. 17-19
        Kaplan        pg. 329-330
    """

    temp = jd - 2415019.5
    tu = temp / 365.25
    yr = 1900 + np.fix(tu)
    leapyrs = np.fix((yr - 1900 - 1) * 0.25)
    days = temp - ((yr - 1900) * 365.0 + leapyrs)

    # check for beginning of year
    if days < 1.0:
        yr = yr - 1
        leapyrs = np.fix((yr - 1900 - 1) * 0.25)
        days = temp - ((yr - 1900) * 365.0 + leapyrs)

    mon, day, h, m, s = dayofyr2mdhms(yr, days)

    return (yr, mon, day, h, m, s)


def gsttime0(yr):
    """This function finds the Greenwich Sidereal time at the beginning of a
    year.  This formula is derived from the Astronomical Almanac and is good
    only for 0 hr UT, 1 Jan of a year.

    Algorithm     : Find the Julian Date Ref 4713 BC
                    Perform expansion calculation to obtain the answer
                    Check the answer for the correct quadrant and size

    Author        : Capt Dave Vallado  USAFA/DFAS  719-472-4109   12 Feb 1989
    In Ada        : Dr Ron Lisowski    USAFA/DFAS  719-472-4110   17 May 1995
    In MatLab     : LtCol Thomas Yoder USAFA/DFAS  719-333-4110   Spring 2000
    Remove Global : Shankar Kulumani 7 Dec 2014
    In Python     : Shankar Kulumani   GWU         630-336-6257   10 Jun 2017

    Inputs        :
        Yr          - Year                                 1988, 1989, etc.

    OutPuts       :
        GST0        - Returned Greenwich Sidereal Time     0 to 2Pi rad

    Locals        :
        JD          - Julian Date                          days from 4713 B.C.
        Tu          - Julian Centuries from 1 Jan 2000

    Constants     :
        TwoPI         Two times Pi (DFASmath.m constant)

    Coupling      :
        mod			  MatLab modulus function

    References    :
        1989 Astronomical Almanac pg. B6
        Escobal       pg. 18 - 21
        Explanatory Supplement pg. 73-75
        Kaplan        pg. 330-332
        BMW           pg. 103-104
    """

    TwoPI = 2 * np.pi

    JD = (367.0 * yr - np.fix(7.0 * (yr + np.fix(10.0 / 12.0)) * 0.25) +
          np.fix(275.0 / 9.0) + 1721014.5)
    Tu = (np.fix(JD) + 0.5 - 2451545.0) / 36525.0
    GST0 = 1.753368559 + 628.3319705 * Tu + 6.770708127E-06 * Tu * Tu
    GST0 = attitude.normalize(GST0, 0, 2 * np.pi)
    return GST0


def gstlst(jd, site_lon):
    """This program calculates GST and LST given the Julian Day and site
    longitude.

    Author:  Shankar Kulumani GWU 18 Jun 2017

    Inputs:
        jd : float
            Julian Day
        sitlon : site longitude (radians)
    Outputs:
        gst - Greenwich Sidereal Time
        lst - Local Sidereal Time

    Constants: None

    Coupling: 

    Modifications:
        18 Jun 2017 - use algorithm 15 from Vallado

    References:
        Astro 321 PREDICT
        Vallado Algorithm 15 
    """
    deg2rad = np.pi / 180
    hour2sec = 3600
    sec2deg = 15 / 3600

    Tut1 = (jd - 2451545.0) / 36525

    gst = (- 6.2e-6 * Tut1 * Tut1 * Tut1 + 0.093104 * Tut1 * Tut1
           + (876600.0 * 3600.0 + 8640184.812866) * Tut1 + 67310.54841)
    gst = (gst % 86400) * sec2deg

    gst = attitude.normalize(gst * deg2rad, 0, 2 * np.pi)
    lst = attitude.normalize(gst[0] + site_lon, 0, 2 * np.pi)

    return gst[0], lst[0]


def finddays(yr, mo, day, hr, m, sec):
    """This function finds the fractional days through a year given the year,
        month, day, hour, minute and second. Midnight New Year's is 0.0

    Algorithm     : Set up array for the number of days per month
                    Check for a leap year
                    Loop to find the elapsed days in the year

    Author        : Capt Dave Vallado  USAFA/DFAS  719-472-4109  11 Dec 1990
    In Ada        : Dr Ron Lisowski    USAFA/DFAS  719-472-4110  17 May 1995
    In MatLab     : LtCol Thomas Yoder USAFA/DFAS  719-333-4110  Spring 2000

    Inputs        :
        Year        - Year                                 1900 .. 2100
        Month       - Month                                   1 .. 12
        Day         - Day                                     1 .. 28,29,30,31
        Hr          - Hour                                    0 .. 23
        Min         - Minute                                  0 .. 59
        Sec         - Second                                0.0 .. 59.999

    OutPuts       :
        DDays        - Fractional elapsed day of year    days

    Locals        :
        i           - Index
            LMonth	  - array holding number of days in each month

    Constants     :
        None.

    Coupling      :
        dayofyr2mdhms is the inverse

    """
    # shift index values to be 1 - 12
    LMonth = np.array([ 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
   
    # TODO: Add logger ehre for leap vs. non leap year
    if yr % 4 == 0 :
        if yr % 100 == 0:
            if yr % 400 == 0:
                LMonth[2] = 29
            else:
                LMonth[2] = 28
        else:
            LMonth[2] = 29
    else:
        LMonth[2] = 28
    
    
    ii = 1
    DDays = 0
    while ii < mo and ii < 12:
        DDays = DDays + LMonth[ii]
        ii += 1

    DDays = DDays + day + hr / 24.0 + m / 1440.0 + sec / 86400.0

    return DDays


