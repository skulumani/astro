"""Test SPICEYPY
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import spiceypy as spice
import numpy as np
import sys
import os
import pdb

from astro import kernels
import pytest
# pytestmark = pytest.mark.skip('WIP')

def test_spiceypy_installation_correct():
    spice_version = 'CSPICE_N0066'
    np.testing.assert_equal(spice.tkvrsn('TOOLKIT'),spice_version)

class TestSpiceyPyFunctions():
    cass = kernels.CassiniKernels()
    spice.furnsh(cass.metakernel) 
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    step = 4000
    times = np.linspace(etOne, etTwo, step) 
    spice.kclear()
    def test_spiceypy_cassini(self):
        spice.furnsh(self.cass.metakernel)
        true_initial_pos = [-5461446.61080924 ,-4434793.40785864 ,-1200385.93315424]
        positions, lightTimes = spice.spkpos('Cassini', self.times, 'J2000', 'None', 'SATURN BARYCENTER')
        np.testing.assert_array_almost_equal(positions[0], true_initial_pos)
        spice.kclear()

    def test_spicepy_rotation_matrix_identity(self):
        spice.furnsh(self.cass.metakernel)
        R_E2E = spice.pxform('IAU_EARTH', 'IAU_EARTH', self.etOne)
        np.testing.assert_array_almost_equal(R_E2E, np.eye(3))
        spice.kclear()

    # @pytest.mark.skip('failing')
    def test_spicepy_state_transformation(self):
        spice.furnsh(self.cass.metakernel)
        T = spice.sxform('IAU_EARTH', 'IAU_SATURN', self.etOne)
        R = spice.pxform('IAU_EARTH', 'IAU_SATURN', self.etOne)
        (Rout, wout) = spice.xf2rav(T)
        np.testing.assert_array_almost_equal(Rout, R)
        spice.kclear()

class TestNEARKernels():
    near = kernels.NearKernels()
    
    def test_msi_frame_transformation(self):
        """Check the frame at a specific time
        """
        spice.kclear()
        spice.furnsh(self.near.metakernel)

        utc = ['Feb 10, 2001 12:00:00 UTC']
        et = spice.str2et(utc)
        
        R_b2c = spice.pxform(self.near.near_body_frame,
                self.near.near_msi_frame, et)
        R_b2c_actual = np.array([
            [.9999988429, -.0004838414, .0014422523],
            [.0004838419, .9999998829, .0000000000],
            [-.0014422522, .0000006978, .9999989600]])
        spice.kclear()
        np.testing.assert_array_almost_equal(R_b2c, R_b2c_actual)

    def test_msi_frame_transformation_fixed(self):
        spice.kclear()
        spice.furnsh(self.near.metakernel)
        etone = spice.str2et('Jan 1, 2001 10:04:02.2 UTC')
        ettwo = spice.str2et('Feb 12, 2001 18:03:02.6 UTC')

        Rone = spice.pxform(self.near.near_body_frame, 
                self.near.near_msi_frame, etone)
        Rtwo = spice.pxform(self.near.near_body_frame,
                self.near.near_msi_frame, ettwo)
        np.testing.assert_array_almost_equal(Rone, Rtwo)
        spice.kclear()

    def test_msi_frame_transpose(self):
        spice.kclear()
        spice.furnsh(self.near.metakernel)
        utc = 'Feb 2, 2001 14:20:10 UTC'
        et = spice.str2et(utc)
        Rb2c = spice.pxform(self.near.near_body_frame, 
                self.near.near_msi_frame, et)
        Rc2b = spice.pxform(self.near.near_msi_frame,
                self.near.near_body_frame, et)
        np.testing.assert_array_almost_equal_nulp(Rb2c, Rc2b.T)
        spice.kclear()

    def test_msi_boresight_vector(self):
        spice.kclear()
        spice.furnsh(self.near.metakernel)
        bs_near_body = np.array([.9999988429, -.0004838414, .0014422523])
        bs_msi_frame = np.array([1, 0, 0])
        et = spice.str2et('Feb 12, 2001 UTC')
        Rc2b = spice.pxform(self.near.near_msi_frame,
                self.near.near_body_frame, et)
        np.testing.assert_array_almost_equal(Rc2b.dot(bs_msi_frame),
                bs_near_body)
        spice.kclear()

    def test_near_landing_coverage(self):
        spice.furnsh(self.near.metakernel)
        utc = ['Feb 12, 2001 12:00:00 UTC', 'Feb 12, 2001 20:05:00 UTC']
        etone = 35251264.18507089
        ettwo = 35280364.18507829
        np.testing.assert_almost_equal(spice.str2et(utc[0]), etone)
        np.testing.assert_almost_equal(spice.str2et(utc[1]), ettwo)
        spice.kclear()

    def test_near_body_frames(self):
        """Transformation from Body fixed frame to prime frame

        There is a constant rotation of 135 deg about the Z/Third axis
        """
        spice.furnsh(self.near.metakernel)

        ckid = spice.ckobj(self.near.Ck)[0]
        cover = spice.ckcov(self.near.Ck, ckid, False, 'INTERVAL', 0.0, 'SCLK') 
        R, av, clkout = spice.ckgpav(ckid, cover[0], 0, 'NEAR_SC_BUS')
        ang = 135*np.pi/180
        R_act = np.array([[np.cos(ang), -np.sin(ang), 0], 
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1]])

        np.testing.assert_array_almost_equal(R, R_act)
        np.testing.assert_array_almost_equal(av, np.zeros(3))
        np.testing.assert_almost_equal(clkout, cover[0])
        spice.kclear()

    def test_earth_spk_coverage(self):
        spice.furnsh(self.near.metakernel)
        spkids = spice.spkobj(self.near.SpkPlanet)
        cover = spice.stypes.SPICEDOUBLE_CELL(1000)
        spice.spkcov(self.near.SpkPlanet, 399, cover)
        result = [x for x in cover]
        expected_result = [-633873600.0, 347630400.0] 
        np.testing.assert_array_almost_equal(result, expected_result)
        spice.kclear()

    def test_eros_spk1_coverage(self):
        spice.furnsh(self.near.metakernel)
        code = spice.bodn2c('EROS')
        cover = spice.stypes.SPICEDOUBLE_CELL(1000)
        spice.spkcov(self.near.SpkEros, code, cover)
        result = [x for x in cover]
        expected_result = [-126273600.0, 37886400.0]
        np.testing.assert_array_almost_equal(result, expected_result)
        spice.kclear()

    def test_eros_spk2_coverage(self):
        code = spice.bodn2c('EROS')
        cover = spice.stypes.SPICEDOUBLE_CELL(1000)
        spice.spkcov(self.near.SpkEros2, code, cover)
        result = [x for x in cover]
        expected_result = [-31334400.0, 81432000.0] 
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_near_spk_landed_coverage(self):
        code = spice.bodn2c('NEAR')
        cover = spice.stypes.SPICEDOUBLE_CELL(1000)
        spice.spkcov(self.near.SpkNearLanded, code, cover)
        result = [x for x in cover]
        expected_result = [35279120.0, 315576000.0]
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_near_spk_orbit_coverage(self):
        code = spice.bodn2c('NEAR')
        cover = spice.stypes.SPICEDOUBLE_CELL(1000)
        spice.spkcov(self.near.SpkNearOrbit, code, cover)
        result = [x for x in cover]
        expected_result = [-43200.0, 35279120.0] 
        np.testing.assert_array_almost_equal(result, expected_result)
