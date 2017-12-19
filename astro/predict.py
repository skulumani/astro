#!/usr/bin/env python3

"""Satellite pass prediction module

This module implements PREDICT for command line use

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
import numpy as np
from astro import geodetic, time, planets, tle, constants, transform, satellite
import os
import tempfile
import sys
import datetime
import argparse
import multiprocessing
import functools
import logging

deg2rad = constants.deg2rad
rad2deg = constants.rad2deg

mu = constants.earth.mu
re = constants.earth.radius
eesqrd = constants.earth.eesqrd
ee = constants.earth.ee

# TODO Make the time step user selctable (optional)


def predict(site_location, date_start, date_end, step_sec=60,
            ifile='./tle.txt', ofile='./output.txt'):
    r"""PREDICT satellite passes for a given Earth location

    sats, all_passes, site = predict(site_location, date_start, date_end, 
                                        ifile, ofile)

    Parameters
    ----------
    site_location : array_like
        latitude (deg), longitude (deg), altitude (km) of obs site
    date_start : array_like
        yr, month, day, hr, min, sec for window start
    date_end : array_like
        yr, month, day, hr, min, sec for window end
    step_sec : float
        Step size in seconds for prediction window
    ifile : str
        path to input TLE file
    ofile : str
        path to output file

    Returns
    -------
    sats : array of Satellite objects
        Holds the data from TLE and updating functions
    all_passes : array of Pass objects
        holds the data for each visible pass
    site : dictionary
        dictionary holding the site information

    See Also
    --------
    see the Satellite module which holds much of the required functions

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu

    References
    ----------
    Look at Astro 321 or MAE3145 or Vallado
    """
    print("""
    ____________ ___________ _____ _____ _____ 
    | ___ \ ___ \  ___|  _  \_   _/  __ \_   _|
    | |_/ / |_/ / |__ | | | | | | | /  \/ | |
    |  __/|    /|  __|| | | | | | | |     | |
    | |   | |\ \| |___| |/ / _| |_| \__/\ | |
    \_|   \_| \_\____/|___/  \___/ \____/ \_/
          \n""")

    logger = logging.getLogger(__name__)

    ifile = os.path.abspath(ifile)
    ofile = os.path.abspath(ofile)

    site_lat = np.deg2rad(site_location[0])
    site_lon = np.deg2rad(site_location[1])
    site_alt = site_location[2]

    jd_start, _ = time.date2jd(date_start[0], date_start[1], date_start[2], date_start[3],
                               date_start[4], date_start[5])
    jd_end, _ = time.date2jd(date_end[0], date_end[1], date_end[2], date_end[3],
                             date_end[4], date_end[5])
    jd_step = step_sec / 86400
    jd_span = np.arange(jd_start, jd_end, jd_step)

    # build the site dictionary
    logger.info('Starting to create site dicitonary')
    site = build_site(jd_span, np.deg2rad(site_location[0]),
                      np.deg2rad(site_location[1]), float(site_location[2]))

    # echo check some data
    logger.info('Write some input data to the output file {}'.format(ofile))
    with open(ofile, 'w') as f:
        f.write('PREDICT RESULTS : Shankar Kulumani\n\n')
        f.write('Check Input Data : \n\n')
        f.write('Site Latitude    = {:9.6f} rad = {:9.6f} deg\n'.format(site_lat,
                                                                        np.rad2deg(site_lat)))
        f.write('Site Longitude   = {:9.6f} rad = {:9.6f} deg\n'.format(site_lon,
                                                                        np.rad2deg(site_lon)))
        f.write('Site Altitude    = {:9.6f} km  = {:9.6f} m\n'.format(site_alt,
                                                                      site_alt * 1000))

        f.write('\nObservation Window :\n\n')
        f.write('Start Date : {} UTC\n'.format(datetime.datetime(int(date_start[0]),
                                                                 int(date_start[1]),
                                                                 int(date_start[2]),
                                                                 int(date_start[3]),
                                                                 int(date_start[4]),
                                                                 int(date_start[5])).isoformat()))
        f.write('End Date   : {} UTC\n\n'.format(datetime.datetime(int(date_end[0]),
                                                                   int(date_end[1]),
                                                                   int(date_end[2]),
                                                                   int(date_end[3]),
                                                                   int(date_end[4]),
                                                                   int(date_end[5])).isoformat()))

        f.write('Start Julian Date   = {}\n'.format(jd_start))
        f.write('End Julian Date     = {}\n\n'.format(jd_end))

    # now we have to read the TLE
    logger.info('Read TLEs from {}'.format(ifile))
    sats = tle.get_tle(ifile)
    logger.debug('Read {} sats'.format(len(sats)))

    parallel_predict = functools.partial(
        satellite.parallel_predict, jd_span=jd_span, site=site)

    logger.info('Starting parallel processing for satellites')

    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        sats_passes = p.map(parallel_predict, sats)

    logger.info('Now outputting visible data to textfile')

    for sat, pass_vis in zip(sats, sats_passes):
        satellite.output(sat, pass_vis, ofile)

    # for sat in sats:
    #     sat.tle_update(jd_span)
    #     sat.visible(site)
    #     sat.output(ofile)

    print("Output saved to : \n{}".format(ofile))

    return sats, sats_passes, site


def build_site(jd_span, site_lat, site_lon, site_alt):
    r"""Create the site dictionary

    site = build_site(jd_span, site_lat, site_lon, site_alt)

    Parameters
    ----------
    jd_span : array
        Array of Julian date times for teh window
    site_lat : float
        site latitude in radians
    site_lon : float
        site longitude in radians
    site_alt : float
        site altitude in kilometers

    Returns
    -------
    site : dictionary
        site dicitonary holding data on the observation site

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """
    site_ecef = geodetic.lla2ecef(site_lat, site_lon, site_alt)

    # loop over jd span
    site = defaultdict(list)
    site['lat'] = site_lat
    site['lon'] = site_lon
    site['alt'] = site_alt
    site['ecef'] = site_ecef

    for jd in jd_span:
        gst, lst = time.gstlst(jd, site_lon)
        # site_eci = geodetic.site2eci(site_lat, site_alt, lst)
        # site_eci = attitude.rot3(gst).dot(site_ecef)
        site_eci = transform.dcm_ecef2eci(jd).dot(site_ecef)

        # test if site is in the dark
        sun_eci, _, _ = planets.sun_earth_eci(jd)

        # append site information to list
        site['eci'].append(site_eci)
        site['sun_eci'].append(sun_eci)
        site['gst'].append(gst)
        site['lst'].append(lst)

    # turn into numpy arrays
    for key, item in site.items():
        site[key] = np.squeeze(np.array(item))

    return site


def parse_args(args):
    r"""Parse arguments from PREDICT 

    (args.latitude, args.longitude, args.altitude), args.start, args.end,
    ifile, args.output = parse_args(args)

    Parameters
    ----------
    args : system command line arguments

    Returns
    -------
    Predict data from the system arguments

    Author
    ------
    Shankar Kulumani		GWU		skulumani@gwu.edu
    """

    # default start and end times if none are given
    start_utc = list(datetime.datetime.today().timetuple()[0:6])
    end_utc = list((datetime.datetime.today() +
                    datetime.timedelta(days=7)).timetuple()[0:6])

    # default output filename
    output_name = os.path.join(tempfile.gettempdir(), 'predict_output.txt')

    # parse inputs
    # TODO Figure out command line arguments which should be positional vs optional
    # TODO Use logging in this funciton

    parser = argparse.ArgumentParser(description='PREDICT satellite passes')

    parser.add_argument(
        'latitude', help='Latitude (deg) of Observation site', type=float)
    parser.add_argument(
        'longitude', help='Longitude (deg) of Observation site', type=float)
    parser.add_argument(
        'altitude', help='Altitude (km) of Observation site', type=float)

    parser.add_argument('-s', '--start', help='UTC Space separated list of year month day of start of prediction window.',
                        default=start_utc, action='store', nargs=3, type=int)
    parser.add_argument('-e', '--end', help='UTC Space separated list of year month day of end of prediction window.',
                        default=end_utc, action='store', nargs=3, type=int)
    # option to use saved tle
    parser.add_argument(
        '-i', '--input', help='Path to input tle file', action='store', type=str)

    parser.add_argument('-o', '--output', help='Path to output file', action='store', type=str,
                        default=output_name)

    parser.add_argument('-st', '--step', help='Step size in seconds for propogation',
                        action='store', type=int, default=60)

    args = parser.parse_args(args)

    # option to download spacetrack tle and then run predict
    if not args.input:
        # logging here
        ifile = os.path.join(tempfile.gettempdir(), 'predict_input.txt')
        tle.get_tle_visible(ifile)
    else:
        # different logging
        ifile = args.input

    start_date = (args.start[0], args.start[1], args.start[2], 0, 0, 0)
    end_date = (args.end[0], args.end[1], args.end[2], 0, 0, 0)

    return ((args.latitude, args.longitude, args.altitude), start_date,
            end_date, args.step, ifile, args.output)


if __name__ == "__main__":
    site_location, start, end, step_sec, input_file, output_file = parse_args(
        sys.argv[1:])
    predict(site_location, start, end, step_sec, input_file, output_file)
