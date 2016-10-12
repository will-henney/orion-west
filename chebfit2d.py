# [[file:alba-orion-west.org::*Program%20to%20fit%202D%20Chebyshev%20to%20so-called%20continuum:%20chebfit2d.py][Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:1]]
import sys
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
# Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:1 ends here

# [[file:alba-orion-west.org::*Program%20to%20fit%202D%20Chebyshev%20to%20so-called%20continuum:%20chebfit2d.py][Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:2]]
def fit_background(data, mask, npx=4, npy=4):
    """Fit a polynomial surface to all elements of a 2D `data` array where
the corresponding `mask` is True.  Return the fit evaluated at each
point of the original data array.

    """
    assert data.shape == mask.shape
    ny, nx = data.shape
    # y = np.arange(ny).reshape((ny,1))
    # x = np.arange(nx).reshape((1,nx))
    y, x = np.mgrid[:ny, :nx]
    p_init = models.Chebyshev2D(x_degree=npx, y_degree=npy)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, x[mask], y[mask], data[mask])
    return p(x, y)
# Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:2 ends here

# [[file:alba-orion-west.org::*Program%20to%20fit%202D%20Chebyshev%20to%20so-called%20continuum:%20chebfit2d.py][Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:3]]
from astropy.wcs import WCS
from astropy import units as u
from helio_utils import vels2waves

def find_mask_for_hdu(hdu, threshold=0.001, v1=-100.0, v2=100.0):
    mask = hdu.data < threshold

    # Cut out a window around line center
    w = WCS(hdu.header, key='A')
    waves = vels2waves([v1, v2], w.wcs.restwav, hdu.header)
    [i1, i2], _, _ = w.all_world2pix(waves, [0, 0], [0, 0], 0)
    i1, i2 = int(i1), int(i2) + 1
    mask[:, :, i1:i2] = False

    return mask
# Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:3 ends here

# [[file:alba-orion-west.org::*Program%20to%20fit%202D%20Chebyshev%20to%20so-called%20continuum:%20chebfit2d.py][Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:4]]
def estimate_pixel_noise(data, size=50):
    # Slices for each corner of the image and one in the middle somewhere
    lo, hi, mid = slice(None, 50), slice(-50, None), slice(300, 350)
    corners = [data[lo, lo], data[lo, hi],
               data[hi, lo], data[hi, hi],
               data[mid, lo], data[mid, hi]]
    sigmas = [np.nanstd(corner) for corner in corners]
    means = [np.nanmean(corner) for corner in corners]
    sigma = np.nanmedian(sigmas)
    mean = np.nanmedian(means)
    print('Corner sigmas:', sigmas, 'Median =', sigma)
    print('Corner means:', means, 'Median =', mean)
    return sigma

def remove_bg(filename, olddir='Calibrated/', newdir='Calibrated/BGsub/'):
    assert olddir in filename
    hdu = fits.open(filename)[0]      # always use first HDU in file


    sigma = estimate_pixel_noise(hdu.data[0])
    mask = find_mask_for_hdu(hdu, threshold=2*sigma)
    if '2015-02-0003-nii' in filename:
        # Low order of x-polynomial for this one
        bg = fit_background(hdu.data[0], mask[0], npx=1, npy=4)
    else:
        # FITS data is 3d, so take 2d slice for fitting...
        bg = fit_background(hdu.data[0], mask[0])
    # ...and then add back 3rd dimension
    bg = bg[None, :, :]

    # Save BG-subtracted data
    hdu.data -= bg
    subfilename = filename.replace(olddir, newdir)
    hdu.writeto(subfilename, clobber=True)

    # And save fitted BG itself
    hdu.data = bg
    bgfilename = subfilename.replace('.fits', '-bg.fits')
    hdu.writeto(bgfilename, clobber=True)

    # And save the mask
    hdu.data = mask.astype(float)
    maskfilename = subfilename.replace('.fits', '-mask.fits')
    hdu.writeto(maskfilename, clobber=True)
# Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:4 ends here

# [[file:alba-orion-west.org::*Program%20to%20fit%202D%20Chebyshev%20to%20so-called%20continuum:%20chebfit2d.py][Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:5]]
if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except:
        print('Usage:', sys.argv[0], 'FITSFILE')
        sys.exit()

    remove_bg(fn)
# Program\ to\ fit\ 2D\ Chebyshev\ to\ so-called\ continuum:\ chebfit2d\.py:5 ends here
