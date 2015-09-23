# [[nil][Program\ to\ generate\ spectral\ map:\ spectral-map\.py:1]]
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import astropy.units as u

if len(sys.argv) > 1:
    line_id = sys.argv[1]
    wavrest = float(sys.argv[2])
else:
    line_id, wavrest = 'ha', 6562.79

def waves2pixels(waves, w):
    n = len(waves)
    pixels, _, _ = w.all_world2pix(waves, [0]*n, [0]*n, 0)
    return pixels

#
# First set up WCS for the output image
#
pixel_scale = 0.5               # arcsec
NX, NY = 2048, 2048
dRA, dDec = -pixel_scale/3600., pixel_scale/3600.
RA0, Dec0 = 83.69, -5.429
w = WCS(naxis=2)
w.wcs.crpix = [0.5*(1 + NX), 0.5*(1 + NY)]
w.wcs.cdelt = [dRA, dDec]
w.wcs.crval = [RA0, Dec0]
w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
w.wcs.cunit = ['deg', 'deg']

# Arrays to hold the output image
outimage = np.zeros((NY, NX))
outweights = np.zeros((NY, NX))

slit_width = 2.0                # width in arcsec of 150 micron slit
slit_pix_width = slit_width/pixel_scale

speclist = glob.glob('Calibrated/*-{}.fits'.format(line_id))

# Window widths for line and BG
dwline = 7.0

# Limits of non-vignetted portion of the slit
good_pixels = {
    '2006-02': (5, None),
    '2007-01': (10, None),
    '2010-01': (10, None),
    '2013-02': (None, -20),
    '2013-12': (None, -10),
    '2015-02-0003': (None, -80),
    '2015-02-0012': (None, -15),
}

for fn in speclist:
    print('Processing', fn)
    spechdu, = fits.open(fn)
    wspec = WCS(spechdu.header, key='A')

    # Trim to good portion of the slit
    goodslice = slice(None, None)
    for k, v in good_pixels.items():
        if k in fn:
            goodslice = slice(*v)
          
    # Find per-slit weight
    slit_weight = spechdu.header['WEIGHT']

    # Find sign of delta wavelength
    dwav = wspec.wcs.get_cdelt()[0]*wspec.wcs.get_pc()[0, 0]
    sgn = np.sign(dwav)         # Need to take slices backwards if this is negative

    # Eliminate degenerate 3rd dimension from data array and trim off bad bits
    spec2d = spechdu.data[0]

    # Find pixel indices for line 
    waves = [wavrest-dwline/2, wavrest+dwline/2]
    # Convert to SI to conform to stupid FITS standards
    waves = (waves*u.Angstrom).to(u.m).value
    i1, i2 = waves2pixels(waves, wspec)

    # Just the entire line for now
    profile = spec2d[:, i1:i2:sgn].sum(axis=-1)

    # Find celestial coordinates for each pixel along the slit
    NS = len(profile)
    slit_coords = pixel_to_skycoord(range(NS), [0]*NS, wspec, 0)

    # Trim off bad parts of slit
    profile = profile[goodslice]
    slit_coords = slit_coords[goodslice]
  
    # Convert to pixel coordinates in output image
    xp, yp = skycoord_to_pixel(slit_coords, w, 0)

    for x, y, bright in zip(xp, yp, profile):
        # Find output pixels corresponding to corners of slit pixel
        # (approximate as square)
        i1 = int(0.5 + x - slit_pix_width/2)
        i2 = int(0.5 + x + slit_pix_width/2)
        j1 = int(0.5 + y - slit_pix_width/2)
        j2 = int(0.5 + y + slit_pix_width/2)
        # Make sure we don't go outside the output grid
        i1, i2 = max(0, i1), max(0, i2)
        i1, i2 = min(NX, i1), min(NX, i2)
        j1, j2 = max(0, j1), max(0, j2)
        j1, j2 = min(NY, j1), min(NY, j2)
        # Fill in the square
        outimage[j1:j2, i1:i2] += bright*slit_weight
        outweights[j1:j2, i1:i2] += slit_weight

# Save everything as different images in a single fits file:
# 1. The sum of the raw slits 
# 2. The weights
# 3. The slits normalized by the weights
label = line_id + '-allvels'
fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(header=w.to_header(), data=outimage, name='slits'),
    fits.ImageHDU(header=w.to_header(), data=outweights, name='weight'),
    fits.ImageHDU(header=w.to_header(), data=outimage/outweights, name='scaled'),
    ]).writeto('new-slits-{}.fits'.format(label), clobber=True)
# Program\ to\ generate\ spectral\ map:\ spectral-map\.py:1 ends here
