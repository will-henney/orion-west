# [[file:alba-orion-west.org::*Program%20to%20do%20multigridding%20of%20new%20spectral%20maps:%20multibin-map.py][Program\ to\ do\ multigridding\ of\ new\ spectral\ maps:\ multibin-map\.py:1]]
import sys
sys.path.append('/Users/will/Work/RubinWFC3/Tsquared')
from rebin_utils import downsample, oversample
from astropy.io import fits

nlist = [1, 2, 4, 8, 16, 32, 64]
mingoods = [2, 2, 2, 1, 1, 1, 2]

try: 
    infile = sys.argv[1]
except:
    print('Usage:', sys.argv[0], 'FITSFILE')
    sys.exit()

hdulist = fits.open(infile)
hdr = hdulist['scaled'].header
im = hdulist['scaled'].data
w = hdulist['weight'].data
m = w > 0.0

for n, mingood in zip(nlist, mingoods):
    im[~m] = 0.0
    outfile = infile.replace('.fits', '-bin{:03d}.fits'.format(n))
    print('Saving', outfile)
    # Save both the scaled image and the weights, but at the full resolution
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=oversample(im, n), header=hdr, name='scaled'),
        fits.ImageHDU(data=oversample(w, n), header=hdr, name='weight'),
    ]).writeto(outfile, clobber=True)
    # Now do the rebinning by a factor of two
    [im,], m, w = downsample([im,], m, weights=w, mingood=mingood)
# Program\ to\ do\ multigridding\ of\ new\ spectral\ maps:\ multibin-map\.py:1 ends here
