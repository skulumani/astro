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
                                                end_date, self.input_file,
                                                self.output_file)
    # just check the first one
    
    def test_first_visible_pass(self):

        np.testing.assert_allclose(self.sats_passes[0][0].jd,
                                   [2453905.8327541635,
                                    2453905.8328699041,
                                    2453905.8329856447,
                                    2453905.8331013853,
                                    2453905.8332171259,
                                    2453905.8333328664,
                                    2453905.833448607,
                                    2453905.8335643476,
                                    2453905.8336800882,
                                    2453905.8337958287,
                                    2453905.8339115693,
                                    2453905.8340273099,
                                    2453905.8341430505,
                                    2453905.8342587911,
                                    2453905.8343745316,
                                    2453905.8344902722,
                                    2453905.8346060128,
                                    2453905.8347217534,
                                    2453905.834837494,
                                    2453905.8349532345,
                                    2453905.8350689751,
                                    2453905.8351847157,
                                    2453905.8353004563,
                                    2453905.8354161968,
                                    2453905.8355319374,
                                    2453905.835647678,
                                    2453905.8357634186,
                                    2453905.8358791592,
                                    2453905.8359948997,
                                    2453905.8361106403,
                                    2453905.8362263809,
                                    2453905.8363421215,
                                    2453905.8364578621,
                                    2453905.8365736026])

    def test_output_file(self):
        np.testing.assert_equal(filecmp.cmp(self.output_file, './astro/tests/output.txt'), True)

class TestParserMinimumInput():
    
    latitude = 39.006
    longitude = -104.2
    altitude = 2.184

    site_location, start, end, input_file, output_file = predict.parse_args([str(latitude), str(longitude), str(altitude)])

    def test_latitude(self):
        np.testing.assert_allclose(self.site_location[0], self.latitude)

    def test_longitude(self):
        np.testing.assert_allclose(self.site_location[1], self.longitude)

    def test_altitude(self):
        np.testing.assert_allclose(self.site_location[2], self.altitude)



