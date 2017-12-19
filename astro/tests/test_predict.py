"""Test out the predict module and function
"""

import numpy as np
from astro import predict
import os, pytest, filecmp


class TestPredict():
    
    @pytest.fixture(autouse=True)
    def setup(self, tmpdir):
        lat = 39.006
        lon = -104.883
        alt = 2.184
        
        site_location = (lat, lon, alt)
        start_date = (2006, 6, 19, 0, 0, 0)
        end_date = (2006, 6, 29, 0, 0, 0)
        
        self.input_file = './astro/tests/Predict.dat'
        self.output_file = os.path.join(str(tmpdir), 'output.txt')

        self.sats, self.sats_passes, self.site = predict.predict(site_location, start_date,
                                                end_date, 10, self.input_file,
                                                self.output_file)
    # just check the first one
    
    def test_first_visible_pass(self):

        np.testing.assert_allclose(self.sats_passes[0][0].jd,
                                   [2453905.8365736026])

    def test_output_file(self):
        np.testing.assert_equal(filecmp.cmp(self.output_file, './astro/tests/output.txt'), True)

class TestParserMinimumInput():
    
    latitude = 39.006
    longitude = -104.2
    altitude = 2.184

    site_location, start, end, step_sec, input_file, output_file = predict.parse_args([str(latitude), str(longitude), str(altitude)])

    def test_latitude(self):
        np.testing.assert_allclose(self.site_location[0], self.latitude)

    def test_longitude(self):
        np.testing.assert_allclose(self.site_location[1], self.longitude)

    def test_altitude(self):
        np.testing.assert_allclose(self.site_location[2], self.altitude)



