import numpy as np
from .. import time

def test_julian_day_j2000():
    jd, mjd = time.date2jd(2000, 1, 1, 12, 0, 0)
    np.testing.assert_almost_equal(jd,2451545)

def test_julian_day_current():
    jd, mjd = time.date2jd(2017, 6, 10, 0, 0, 0)
    np.testing.assert_equal(jd, 2457914.5)

def test_julian_day_upper_limit():
    jd, mjd = time.date2jd(2100, 2, 28, 0, 0, 0)
    np.testing.assert_equal(jd, 2488127.5)

def test_julian_day_lower_limit():
    jd, mjd = time.date2jd(1900, 3, 1, 0, 0, 0)
    np.testing.assert_equal(jd, 2415079.5)

def test_finddays_newyears():
    ddays = time.finddays(2017, 1, 1, 0, 0, 0)
    np.testing.assert_equal(ddays, 0)

def test_dayofyr2mdhms_january():
    yr = 2017
    days = 1.5

    expected_output = (1, 1, 12, 0, 0)
    actual_output = time.dayofyr2mdhms(yr, days)
    np.testing.assert_array_almost_equal(actual_output, expected_output)

