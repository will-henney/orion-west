# [[file:alba-orion-west.org::*Program%20to%20combine%20different%20grids:%20multibin-combine.py][Program\ to\ combine\ different\ grids:\ multibin-combine\.py:1]]
import sys
from astropy.io import fits
import numpy as np

try: 
    prefix, minw_scale = sys.argv[1], float(sys.argv[2])
except:
    print('Usage:', sys.argv[0], 'FITSFILE_PREFIX MINIMUM_WEIGHT')
    sys.exit()

nlist = [1, 2, 4, 8, 16, 32]
minweights = [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
outim = np.zeros((2048, 2048))
for n, minw in reversed(list(zip(nlist, minweights))):
    fn = '{}-bin{:03d}.fits'.format(prefix, n)
    hdulist = fits.open(fn)
    im = hdulist['scaled'].data
    hdr = hdulist['scaled'].header
    w = hdulist['weight'].data
    m = w*im >= minw*minw_scale
    outim[m] = im[m]
fits.PrimaryHDU(header=hdr, data=outim).writeto(prefix + '-multibin.fits', clobber=True)
# Program\ to\ combine\ different\ grids:\ multibin-combine\.py:1 ends here
