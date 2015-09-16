# [[nil][Program\ to\ generate\ spectral\ map:\ spectral-map\.py:1]]
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

#
# First set up WCS for the output image
#
pixel_scale = 0.5               # arcsec
NX, NY = 4096, 4096
dRA, dDec = -pixel_scale/3600., pixel_scale/3600.
RA0, Dec0 = 83.61, -5.423
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

speclist = glob.glob('Calibrated/*-ha.fits')

for fn in speclist:
    print('Processing', fn)
    spechdu, = fits.open(fn)
    wspec = WCS(spechdu.header, key='A')

    # Just the entire spectrum for now
    profile = spechdu.data[0].sum(axis=-1)

    # Find celestial coordinates for each pixel along the slit
    NS = len(profile)
    slit_coords = pixel_to_skycoord(range(NS), [0]*NS, wspec, 0)

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
        outimage[j1:j2, i1:i2] += bright
        outweights[j1:j2, i1:i2] += 1.0

# Save everything as different images in a single fits file:
# 1. The sum of the raw slits 
# 2. The weights
# 3. The slits normalized by the weights
label = 'ha-allvels'
fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(header=w.to_header(), data=outimage, name='slits'),
    fits.ImageHDU(header=w.to_header(), data=outweights, name='weight'),
    fits.ImageHDU(header=w.to_header(), data=outimage/outweights, name='scaled'),
    ]).writeto('new-slits-{}.fits'.format(label), clobber=True)
# Program\ to\ generate\ spectral\ map:\ spectral-map\.py:1 ends here
