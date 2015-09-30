# [[file:alba-orion-west.org::*Module%20to%20find%20heliocentric%20correction:%20helio_utils.py][Module\ to\ find\ heliocentric\ correction:\ helio_utils\.py:1]]
import numpy as np
from astropy.io import fits
from astropy import coordinates as coord
from astropy import units as u
from astropy import constants as const
from astropy.wcs import WCS
from pyslalib.slalib import sla_dcs2c, sla_evp, sla_rverot, sla_obs

def ra_dec_from_header(hdr):
    ra = coord.Longitude(hdr['RA'], u.hour)
    dec = coord.Latitude(hdr['DEC'], u.deg)
    return ra, dec

def ra_dec_from_header_wcs(hdr):
    w = WCS(hdr, key='A').celestial
    ra = coord.Longitude(w.wcs.crval[0], u.deg)
    dec = coord.Latitude(w.wcs.crval[1], u.deg)
    return ra, dec

def mjd_from_header(hdr):
    return float(hdr.get('MJD-OBS'))

def obs_lat_SPM():
    obs_id, obs_name, obs_long, obs_lat, obs_height = sla_obs(0, "SANPM83")
    return coord.Latitude(obs_lat, u.radian)

def st_from_header(hdr):
    return coord.Longitude(hdr['ST'], u.hour)

def helio_topo_from_header(hdr):
    ra, dec = ra_dec_from_header_wcs(hdr)
    mjd = mjd_from_header(hdr)
    st = st_from_header(hdr)
    obs_lat = obs_lat_SPM()
    return helio_topo_correction(ra, dec, mjd, st, obs_lat)

def helio_topo_correction(ra, dec, mjd, st, obs_lat):
    """Find radial velocity correction in km/s due to transformation
    between topocentric and heliocentric frame.  Positive when
    observatory is moving away from source in heliocentric frame.
 
    Parameters
    ----------
    ra : :class:`~astropy.coordinates.Longitude` 
        Right ascension of source
    dec : :class:`~astropy.coordinates.Latitude` 
        Declination of source
    mjd : float
        Modified Julian Date of observation
    st : :class:`~astropy.coordinates.Angle`
        Sideral Time of observation
    obs_lat : :class:`~astropy.coordinates.Latitude` 
        Latitude of observatory

    """
    return helio_geo_correction(ra, dec, mjd, st) + geo_topo_correction(ra, dec, st, obs_lat)

def helio_geo_correction(ra, dec, mjd, st):
    """Motion of earth's center in heliocentric frame"""
    # line-of-sight unit vector to astronomical object
    k_los = sla_dcs2c(ra.radian, dec.radian)
    # Velocity and position of earth in barycentric and heliocentric frames
    # Units are AU and AU/s
    vel_bary, pos_bary, vel_hel, pos_hel = sla_evp(mjd, 2000.0)
    # Radial velocity correction (km/s) due to helio-geocentric transformation  
    # Positive when earth is moving away from object
    return u.AU.to(u.km, -np.dot(vel_hel, k_los))

def geo_topo_correction(ra, dec, st, obs_lat):
    """Motion of telescope in geocentric frame"""
    return sla_rverot(obs_lat.radian, ra.radian, dec.radian, st.radian)


LIGHT_SPEED_KMS = const.c.to('km/s').value
def vels2waves(vels, restwav, hdr):
    """Heliocentric radial velocity (in km/s) to observed wavelength (in
    m, or whatever units restwav is in)

    """
    # Heliocentric correction
    vels = np.array(vels) + helio_topo_from_header(hdr)
    waves = restwav*(1.0 + vels/LIGHT_SPEED_KMS)
    return waves


def waves2vels(waves, restwav, hdr):
    """Observed wavelength to Heliocentric radial velocity (in m/s) 

    """
    vels = const.c*(waves - restwav)/restwav
    # Heliocentric correction
    vels -= helio_topo_from_header(hdr)*u.km/u.s

    return vels
# Module\ to\ find\ heliocentric\ correction:\ helio_utils\.py:1 ends here
