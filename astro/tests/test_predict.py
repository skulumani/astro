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
                                   (2453905.8333333,
                                    2453905.8347222,
                                    2453905.8361110))

    def test_output_file(self):
        np.testing.assert_equal(filecmp.cmp(self.output_file, './astro/tests/output.txt'), True)



