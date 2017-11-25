"""Satellite Module

Define satellite object from a TLE

Notes
-----

Attributes
----------
module_level_variable1 : int
    Descrption of the variable

Author
------
Shankar Kulumani		GWU		skulumani@gwu.edu
"""
import numpy as np
from collections import namedtuple
from astro import constants, time

deg2rad = np.pi / 180
rad2deg = 180 / np.pi
sec2day = 1 / (24 * 3600)
day2sec = 24 * 3600

COE = namedtuple('COE', ['n', 'ecc', 'raan', 'argp', 'mean', 'E', 'nu', 'a',
                         'p', 'inc'])

PASS = namedtuple('PASS', ['jd', 'az', 'el', 'site_eci', 'sat_eci', 'gst', 'lst',
                           'sun_alt', 'alpha', 'beta', 'sun_eci', 'rho'])

RADAR_PASS = namedtuple('RADAR_PASS', ['jd', 'site_eci', 'sat_r_eci', 'sat_v_eci', 
                                       'gst', 'lst', 'sun_eci', 'sun_alt', 'alpha', 
                                       'beta', 'rho', 'az', 'el', 'drho', 'daz', 'dele'])

class Satellite(object):
    """Satellite class storing all the necessary information for a single
    satellite extracted from a TLE from space-track.org
    """

    def __init__(self, elements):
        # now do some conversions of the TLE components
        epoch_year = (2000 + elements.epoch_year
                      if elements.epoch_year < 70
                      else 1900 + elements.epoch_year)
        # working units
        ndot2 = elements.ndot_over_2 * (2 * np.pi) * (sec2day**2)
        inc0 = elements.inc * deg2rad
        raan0 = elements.raan * deg2rad
        argp0 = elements.argp * deg2rad
        mean0 = elements.ma * deg2rad
        n0 = elements.mean_motion * 2 * np.pi * sec2day
        ecc0 = elements.ecc
        epoch_day = elements.epoch_day

        # calculate perturbations raan, argp, ecc
        raandot, argpdot, eccdot = j2dragpert(inc0, ecc0, n0, ndot2)

        # calculate epoch julian date
        mo, day, hour, mn, sec = time.dayofyr2mdhms(epoch_year, epoch_day)
        epoch_jd, _ = time.date2jd(epoch_year, mo, day, hour, mn, sec)

        # store with class
        self.satname = elements.satname
        self.satnum = elements.satnum
        self.tle = elements
        self.ndot2 = ndot2
        self.inc0 = inc0
        self.raan0 = raan0
        self.ecc0 = ecc0
        self.argp0 = argp0
        self.mean0 = mean0
        self.n0 = n0
        self.raandot = raandot
        self.argpdot = argpdot
        self.eccdot = eccdot
        self.epoch_jd = epoch_jd
        self.epoch_year = epoch_year
        self.epoch_day = epoch_day

    def tle_update(self, jd_span, mu=398600.5):
        """Update the state of satellite given a JD time span

        This procedure uses the method of general perturbations to update
        classical elements from epoch to a future time for inclined elliptical
        orbits.  It includes the effects due to first order secular rates
        (second order for mean motion) caused by drag and J2.

        Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741

        Inputs:
            deltat - elapsed time since epoch (sec)
            n0 - mean motion at epoch (rad/sec)
            ndot2 - mean motion rate divided by 2 (rad/sec^2)
            ecc0 - eccentricity at epoch (unitless)
            eccdot - eccentricity rate (1/sec)
            raan0 - RAAN at epoch (rad)
            raandot - RAAN rate (rad/sec)
            argp0 - argument of periapsis at epoch (rad)
            argdot - argument of periapsis rate (rad/sec)
            mean0 - mean anomaly at epoch

        Outputs:
            n - mean motion at epoch + deltat (rad/sec)
            ecc -  eccentricity at epoch + deltat (unitless)
            raan - RAAN at epoch + deltat (rad)
            argp - argument of periapsis at epoch + deltat (rad)
            nu -  true anomaly at epoch + deltat (rad)

        Globals: None

        Constants: None

        Coupling:
            newton.m

        References:
            Astro 321 PREDICT LSN 24-25
        """

        deltat = (jd_span - self.epoch_jd) * day2sec
        n0 = self.n0
        ecc0 = self.ecc0
        ndot2 = self.ndot2
        eccdot = self.eccdot
        raan0 = self.raan0
        raandot = self.raandot
        argp0 = self.argp0
        argpdot = self.argpdot
        mean0 = self.mean0

        # mean motion at future time
        n = n0 + ndot2 * 2 * deltat
        ecc = ecc0 + eccdot * deltat
        raan = raan0 + raandot * deltat
        argp = argp0 + argpdot * deltat

        # get true anomaly at future time
        mean = mean0 + n0 * deltat + ndot2 * deltat**2
        mean = attitude.normalize(mean, 0, 2 * np.pi)

        E, nu, count = kepler.kepler_eq_E(mean, ecc)

        a = (mu / n**2)**(1 / 3)
        p = a * (1 - ecc**2)
        inc = self.inc0 * np.ones_like(p)

        # convert to ECI
        r_eci, v_eci, _, _ = kepler.coe2rv(p, ecc, inc, raan, argp, nu, mu)

        # save all of the variables to the object
        self.jd_span = jd_span
        self.coe = COE(n=n, ecc=ecc, raan=raan, argp=argp, mean=mean,
                       E=E, nu=nu, a=a, p=p, inc=inc)
        self.r_eci = r_eci
        self.v_eci = v_eci
        return 0

    def visible(self, site):
        """Check if current sat is visible from the site
        """
        sat_eci = self.r_eci
        site_eci = site['eci']
        sun_eci = site['sun_eci']

        alpha = np.arccos(np.einsum('ij,ij->i', site_eci, sun_eci) /
                          np.linalg.norm(site_eci, axis=1) / np.linalg.norm(sun_eci,
                                                                            axis=1))
        beta = np.arccos(np.einsum('ij,ij->i', sat_eci, sun_eci) /
                         np.linalg.norm(sat_eci, axis=1) / np.linalg.norm(sun_eci,
                                                                          axis=1))
        sun_alt = np.linalg.norm(sun_eci, axis=1) * np.sin(beta)

        jd_vis = []
        rho_vis = []
        az_vis = []
        el_vis = []

        # create array to store all the passes
        all_passes = []
        current_pass = PASS(jd=[], az=[], el=[], site_eci=[], sat_eci=[],
                            gst=[], lst=[], sun_alt=[], alpha=[], beta=[],
                            sun_eci=[], rho=[])

        jd_last = self.jd_span[0]
        for (jd, a, b, sa, si, su, su_alt, lst, gst) in zip(self.jd_span, alpha,
                                                            beta, sat_eci,
                                                            site_eci, sun_eci,
                                                            sun_alt, site['lst'],
                                                            site['gst']):

            if a > np.pi / 2:
                rho, az, el = geodetic.rhoazel(sa, si, site['lat'], lst)

                if rho < 1500 and el > 10 * deg2rad:
                    if b < np.pi / 2 or su_alt > 6378.137:
                        jd_vis.append(jd)
                        rho_vis.append(rho)
                        az_vis.append(az)
                        el_vis.append(el)

                        # we have a visible state - save to current pass if
                        # time since last visible is small enough if not then
                        # we have to create another pass namedtuple
                        if np.absolute(jd_last - jd) > (10 / 24 / 60):
                            if current_pass.jd:  # new pass
                                all_passes.append(current_pass)
                                current_pass = PASS(jd=[], az=[], el=[],
                                                    site_eci=[], sat_eci=[],
                                                    gst=[], lst=[], sun_alt=[],
                                                    alpha=[], beta=[],
                                                    sun_eci=[], rho=[])

                        current_pass.jd.append(jd)
                        current_pass.az.append(az)
                        current_pass.el.append(el)
                        current_pass.site_eci.append(si)
                        current_pass.sat_eci.append(sa)
                        current_pass.gst.append(gst)
                        current_pass.lst.append(lst)
                        current_pass.sun_alt.append(su_alt)
                        current_pass.alpha.append(a)
                        current_pass.beta.append(b)
                        current_pass.sun_eci.append(su)
                        current_pass.rho.append(rho)
                        jd_last = jd

        all_passes.append(current_pass)

        self.jd_vis = jd_vis
        self.rho_vis = rho_vis
        self.az_vis = az_vis
        self.el_vis = el_vis
        self.pass_vis = all_passes

    # TODO: Update documentation
    def visible_radar(self, site):
        """Check if current sat is visible from the site given an example radar sensor
        """
        sat_r_eci = self.r_eci
        sat_v_eci = self.v_eci
        site_eci = site['eci']
        sun_eci = site['sun_eci']

        alpha = np.arccos(np.einsum('ij,ij->i', site_eci, sun_eci) /
                          np.linalg.norm(site_eci, axis=1) / np.linalg.norm(sun_eci,
                                                                            axis=1))
        beta = np.arccos(np.einsum('ij,ij->i', sat_r_eci, sun_eci) /
                         np.linalg.norm(sat_r_eci, axis=1) / np.linalg.norm(sun_eci,
                                                                          axis=1))
        sun_alt = np.linalg.norm(sun_eci, axis=1) * np.sin(beta)

        jd_vis = []
        rho_vis = []
        az_vis = []
        el_vis = []
        drho_vis = []
        daz_vis = []
        dele_vis = []

        # create array to store all the passes
        all_passes = []
        current_pass = RADAR_PASS(jd=[], site_eci=[], sat_r_eci=[],
                                  sat_v_eci=[], gst=[], lst=[], sun_alt=[],
                                  alpha=[], beta=[], sun_eci=[], 
                                  rho=[], az=[], el=[], drho=[], daz=[], dele=[])

        jd_last = self.jd_span[0]
        for (jd, a, b, sar, sav, si, su, su_alt, lst, gst) in zip(self.jd_span, alpha,
                                                            beta, sat_r_eci, sat_v_eci,
                                                            site_eci, sun_eci,
                                                            sun_alt, site['lst'],
                                                            site['gst']):
            # deterministic radar measurement
            rho, az, el, drho, daz, dele = geodetic.rv2rhoazel(sar, sav, site['lat'],
                                                               site['lon'], site['alt'], jd)
            if el > 10 * deg2rad:
                jd_vis.append(jd)
                rho_vis.append(rho)
                az_vis.append(az)
                el_vis.append(el)
                drho_vis.append(drho)
                daz_vis.append(daz)
                dele_vis.append(dele)

                # we have a visible state - save to current pass if
                # time since last visible is small enough if not then
                # we have to create another pass namedtuple
                if np.absolute(jd_last - jd) > (10 / 24 / 60):
                    if current_pass.jd:  # new pass
                        all_passes.append(current_pass)
                        current_pass = RADAR_PASS(jd=[], site_eci=[], sat_r_eci=[],
                                  sat_v_eci=[], gst=[], lst=[], sun_alt=[],
                                  alpha=[], beta=[], sun_eci=[], 
                                  rho=[], az=[], el=[], drho=[], daz=[], dele=[])

                current_pass.jd.append(jd)
                current_pass.site_eci.append(si)
                current_pass.sat_r_eci.append(sar)
                current_pass.sat_v_eci.append(sav)
                current_pass.gst.append(gst)
                current_pass.lst.append(lst)
                current_pass.sun_alt.append(su_alt)
                current_pass.alpha.append(a)
                current_pass.beta.append(b)
                current_pass.sun_eci.append(su)
                current_pass.rho.append(rho)
                current_pass.az.append(az)
                current_pass.el.append(el)
                current_pass.drho.append(drho)
                current_pass.daz.append(daz)
                current_pass.dele.append(dele)
                jd_last = jd

        all_passes.append(current_pass)

        self.jd_vis = jd_vis
        self.rho_vis = rho_vis
        self.az_vis = az_vis
        self.el_vis = el_vis
        self.drho_vis = drho_vis
        self.daz_vis = daz_vis
        self.dele_vis = dele_vis
        self.pass_vis = all_passes
    

    def output(self, filename):
        """Write to output file
        """
        space = '    '
        with open(filename, 'w') as f:

            f.write('%s %05.0f\n' % (self.satname, self.satnum))
            f.write(
                'PASS    MON/DAY    HR:MIN(UT)    RHO(KM)    AZ(DEG)    EL(DEG)    SAT          \n')
            f.write(
                '-------------------------------------------------------------------------------\n\n')

            for ii, cur_pass in enumerate(self.pass_vis):
                for jd, rho, az, el in zip(cur_pass.jd, cur_pass.rho, cur_pass.az, cur_pass.el):
                    # convert julian day to normal date
                    yr, mo, day, hr, mn, sec = time.jd2date(jd)

                    # if sec > 30:
                    #     mn = mn + 1
                    #     if mn == 60:
                    #         hr = hr + 1
                    #         mn = 0
                    f.write('%4.0f%s' % (ii, space))
                    f.write('%3.0f/%3.0f%s' % (mo, day, space))
                    f.write('%02.0f:%02.0f:%3.1f%s' % (hr, mn, sec, space))
                    f.write('%7.2f%s' % (rho, space))
                    f.write('%7.2f%s' % (az * 180 / np.pi, space))
                    f.write('%7.2f%s' % (el * 180 / np.pi, space))
                    f.write('%13s\n' % (self.satname))

    def plot_pass(self, pass_num):
        """Try and plot a pass on a polar plot
        """
        def mapr(r):
            return 90 - r

        def add_arrow_to_line2D(
                axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
                arrowstyle='-|>', arrowsize=1, transform=None):
            """
            Add arrows to a matplotlib.lines.Line2D at selected locations.

            Parameters:
            -----------
            axes:
            line: Line2D object as returned by plot command
            arrow_locs: list of locations where to insert arrows, % of total length
            arrowstyle: style of the arrow
            arrowsize: size of the arrow
            transform: a matplotlib transform instance, default to data coordinates

            Returns:
            --------
            arrows: list of arrows
            """
            if not isinstance(line, mlines.Line2D):
                raise ValueError("expected a matplotlib.lines.Line2D object")
            x, y = line.get_xdata(), line.get_ydata()

            arrow_kw = {
                "arrowstyle": arrowstyle,
                "mutation_scale": 10 * arrowsize,
            }

            color = line.get_color()
            use_multicolor_lines = isinstance(color, np.ndarray)
            if use_multicolor_lines:
                raise NotImplementedError("multicolor lines not supported")
            else:
                arrow_kw['color'] = color

            linewidth = line.get_linewidth()
            if isinstance(linewidth, np.ndarray):
                raise NotImplementedError("multiwidth lines not supported")
            else:
                arrow_kw['linewidth'] = linewidth

            if transform is None:
                transform = axes.transData

            arrows = []
            for loc in arrow_locs:
                s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
                n = np.searchsorted(s, s[-1] * loc)
                arrow_tail = (x[n], y[n])
                arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
                p = mpatches.FancyArrowPatch(
                    arrow_tail, arrow_head, transform=transform,
                    **arrow_kw)
                axes.add_patch(p)
                arrows.append(p)
            return arrows

        jd = self.pass_vis[pass_num].jd
        az = self.pass_vis[pass_num].az
        el = self.pass_vis[pass_num].el

        sy, smo, sd, sh, smn, ss = time.jd2date(jd[0])
        ey, emo, ed, eh, emn, es = time.jd2date(jd[-1])
        fig = plt.figure()
        ax = plt.subplot(111, projection='polar')
        line = ax.plot(az, mapr(np.rad2deg(el)))[0]
        ax.set_yticks(range(0, 90, 10))
        ax.set_yticklabels(map(str, range(90, 0, -10)))
        ax.set_theta_zero_location("N")
        fig.suptitle("%s" %
                     (self.satname), y=1.05)
        plt.title('%2d/%2d' % (smo, sd))
        plt.title('Start\n%2d:%2d:%3.1f' % (sh, smn, ss), loc='left')
        plt.title('End\n%2d:%2d:%3.1f' % (eh, emn, es), loc='right')
        add_arrow_to_line2D(ax, line, arrow_locs=[0.25, 0.5, 0.75],
                            arrowstyle='->')
        plt.show()

def j2dragpert(inc0, ecc0, n0, ndot2, mu=398600.5, re=6378.137, J2=0.00108263):
    """
    This file calculates the rates of change of the right ascension of the
    ascending node(raandot), argument of periapsis(argdot), and
    eccentricity(eccdot).  The perturbations are limited to J2 and drag only.

    Author:   C2C Shankar Kulumani   USAFA/CS-19   719-333-4741
            12 5 2014 - modified to remove global varaibles
            17 Jun 2017 - Now in Python for awesomeness

    Inputs:
    inc0 - initial inclination (radians)
    ecc0 - initial eccentricity (unitless)
    n0 - initial mean motion (rad/sec)
    ndot2 - mean motion rate divided by 2 (rad/sec^2)

    Outputs:
    raandot - nodal rate (rad/sec)
    argdot - argument of periapsis rate (rad/sec)
    eccdot - eccentricity rate (1/sec)

    Globals: None

    Constants:
    mu - 398600.5 Earth gravitational parameter in km^3/sec^2
    re - 6378.137 Earth radius in km
    J2 - 0.00108263 J2 perturbation factor

    Coupling: None

    References:
    Vallado
    """

    # calculate initial semi major axis and semilatus rectum
    a0 = (mu / n0**2) ** (1 / 3)
    p0 = a0 * (1 - ecc0**2)

    # mean motion with J2
    nvec = n0 * (1 + 1.5 * J2 * (re / p0)**2 *
                 np.sqrt(1 - ecc0**2) * (1 - 1.5 * np.sin(inc0)**2))

    # eccentricity rate
    eccdot = (-2 * (1 - ecc0) * 2 * ndot2) / (3 * nvec)

    # calculate nodal rate
    raandot = (-1.5 * J2 * (re / p0)**2 * np.cos(inc0)) * nvec

    # argument of periapsis rate
    argdot = (1.5 * J2 * (re / p0)**2 * (2 - 2.5 * np.sin(inc0)**2)) * nvec

    return (raandot, argdot, eccdot)
