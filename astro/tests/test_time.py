import numpy as np
from astro import time

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

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
    np.testing.assert_equal(ddays, 1)

def test_finddays_leapday():
    dday = time.finddays(2000, 2, 29, 0, 0, 0)
    np.testing.assert_equal(dday, 31+29)

def test_finddays_vallado_311():
    expected_days = 129.0
    actual_days = time.finddays(1992, 5, 8, 0, 0, 0)
    np.testing.assert_allclose(actual_days, expected_days)

def test_dayofyr2mdhms_vallado_311():
    actual_yr, actual_days = 1992, 129.0
    actual_mo, actual_day, actual_hr, actual_min, actual_sec = 5, 8, 0, 0, 0

    actual_mdhms = time.dayofyr2mdhms(1992, 129)
    np.testing.assert_allclose(actual_mdhms, (actual_mo, actual_day, actual_hr, actual_min, actual_sec))

def test_vallado_312_converting_fractional_days():
    expected_days = 77.5097222
    actual_days = time.finddays(2001, 3, 18, 12, 14, 0)
    np.testing.assert_allclose(actual_days, expected_days)

def test_vallado_312_dayofyear():
    expected_mdhms = (3, 18, 12, 13, 59.99808)
    actual_mdhms = time.dayofyr2mdhms(2001, 77.5097222)
    np.testing.assert_allclose(actual_mdhms, expected_mdhms)

def test_finddays_special_year():
    expected_days = 365
    actual_days = time.finddays(1800, 12, 31, 0, 0, 0)
    np.testing.assert_allclose(actual_days, expected_days)

# TODO Finish adding dates from table 3-3
class TestDayOfLeapYear():
    # Table 3-3 from Vallado
    def test_jan_1(self):
        yrmdhms = (2000, 1, 1, 0, 0, 0)
        days = time.finddays(yrmdhms[0], yrmdhms[1], yrmdhms[2], yrmdhms[3], yrmdhms[4], yrmdhms[5])
        mdhms = time.dayofyr2mdhms(yrmdhms[0], 1)

        np.testing.assert_allclose(days, 1)
        np.testing.assert_allclose(mdhms, yrmdhms[1:])
    def test_jan_31(self):
        days = time.finddays(2000, 1, 31, 0, 0, 0)
        np.testing.assert_allclose(days, 31)

def test_date2jd_vallado_p409():
    # example from 7-1 Vallado pg. 409
    jd, mjd = time.date2jd(1995, 5, 20, 3, 17, 2)
    np.testing.assert_allclose(jd, 2449857.636829, rtol=1e-5)

def test_jd2date_vallado_p409():
    expected_date = (1995, 5, 20, 3, 17, 2.0255)
    actual_date = time.jd2date(2449857.636829)
    np.testing.assert_allclose(actual_date, expected_date, rtol=1e-3)

def test_jd2date_1900():
    expected_date = (1899, 12, 31, 0, 0, 0)
    actual_date = time.jd2date(2415019.5)
    np.testing.assert_allclose(actual_date, expected_date, rtol=1e-4)

def test_dayofyr2mdhms_leap():
    yr = 2000
    days = 31+29
    expected_date = (2, 29, 0, 0, 0)
    actual_output = time.dayofyr2mdhms(yr, days)
    np.testing.assert_allclose(actual_output, expected_date)

def test_dayofyr2mdhms_january():
    yr = 2017
    days = 1.5

    expected_output = (1, 1, 12, 0, 0)
    actual_output = time.dayofyr2mdhms(yr, days)
    np.testing.assert_array_almost_equal(actual_output, expected_output)

def test_dayofyr2mdhms_start():
    yr = 2006
    day = 170
    expected_output = (6, 19, 0, 0, 0)
    actual_output = time.dayofyr2mdhms(yr, day)
    np.testing.assert_allclose(actual_output, expected_output)

def test_dayofyr2mdhms_end():
    yr = 2006
    day = 180
    expected_output = (6, 29, 0, 0, 0)
    actual_output = time.dayofyr2mdhms(yr, day)
    np.testing.assert_allclose(actual_output, expected_output)

def test_date2jd_start():
    expected_output = 2453905.5
    actual_output = time.date2jd(2006, 6, 19, 0, 0, 0)
    np.testing.assert_allclose(actual_output[0], expected_output)

def test_date2jd_end():
    expected_output = 2453915.5
    actual_output = time.date2jd(2006, 6, 29, 0, 0, 0)
    np.testing.assert_allclose(actual_output[0], expected_output)

def test_date2jd_vallado():
    expected_jd = 2450383.09722222
    actual_output, _ = time.date2jd(1996, 10, 26, 14, 20, 0)
    np.testing.assert_allclose(actual_output, expected_jd)

def test_gst_vallado():
    expected_gst = 152.578787886
    actual_gst, _ = time.gstlst(2448855.009722, -104*deg2rad)
    np.testing.assert_allclose(actual_gst * rad2deg, expected_gst, rtol=1e-3)

def test_lst_vallado():
    expected_lst = 48.578787886
    _, actual_lst = time.gstlst(2448855.009722, -104 * deg2rad)
    np.testing.assert_allclose(actual_lst * rad2deg, expected_lst, rtol=1e-3)

class TestTimeGSTValladoEx3_3():
    dateexp = (1995, 2, 24, 12, 0, 0)
    jdexp = 2449773.0
    gstexp = np.deg2rad(333.893486)
    jd, mjd = time.date2jd(dateexp[0], dateexp[1], dateexp[2], dateexp[3], dateexp[4], dateexp[5])
    date = time.jd2date(jdexp)
    gst, _ = time.gstlst(jd, 0)

    def test_jd(self):
        np.testing.assert_allclose(self.jd, self.jdexp)

    def test_date(self):
        np.testing.assert_allclose(self.date, self.dateexp)

    def test_gst(self):
        np.testing.assert_allclose(self.gst, self.gstexp)

