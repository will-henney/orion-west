import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

try:
    fname = sys.argv[1]
except IndexError:
    sys.exit('Usage: {} FILTERNAME'.format(sys.argv[0]))

acs_dir = '/Volumes/SSD-1TB/OrionTreasury/acs/'
acs_fmt = acs_dir + 'hlsp_orion_hst_acs_strip{field}_{fname}_v1_drz.fits'

for field in '1r', '0r', '2r':
    hdulist = fits.open(acs_fmt.format(field=field, fname=fname))
    # print(hdulist.info())
    hdu = hdulist['SCI']
    w = WCS(hdu.header)
    print(w)
