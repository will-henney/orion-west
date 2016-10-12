import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
from astropy.wcs import WCS
import aplpy
import numpy as np
from matplotlib import cm, colors

def extract_window_hdu(hdu, x1=0.0, x2=0.6, y1=0.3, y2=1.0):
    """Extract a window from the image in `hdu`

    The window is specified by corners `x1`, `x2`, `y1`, `y2` in
    fractional coordinates. 
    Returns a new `astropy.io.fits.ImageHDU`

    """
    ny, nx = hdu.data.shape
    xslice = slice(int(x1*nx), int(x2*nx))
    yslice = slice(int(y1*ny), int(y2*ny))
    w = WCS(hdu.header)
    newdata = hdu.data[yslice, xslice]
    newheader = w.slice((yslice, xslice)).to_header()
    return fits.ImageHDU(data=newdata, header=newheader)



fn = 'WFI-Images/Orion_H_A_deep.fits'
slit_fn = 'new-slits-ha-allvels.fits'
# cmap = cm.PuRd
# cmap = cm.magma_r
cmap = cm.copper_r
slit_hdu = fits.open(slit_fn)['scaled']
shallow_hdu = fits.open(fn.replace('deep', 'shallow'))[0]
m = np.isfinite(slit_hdu.data)
slit_hdu.data[m] = 1.0
slit_hdu.data[~m] = 0.0
figfile = 'fov-with-slits.pdf'
f = aplpy.FITSFigure(fn)
f.recenter(83.7375, -5.4167, width=0.35, height=0.25)
f.show_grayscale(pmin=65.0, pmax=95, stretch='sqrt')
f.show_contour(extract_window_hdu(shallow_hdu),
               levels=[20.0, 30.0, 40.0, 50.0,
                       70.0, 100.0, 200.0, 400.0, 800.0],
               norm=colors.LogNorm(), vmin=0.3, vmax=1000.0,
               cmap=cmap, filled=True, alpha=0.5, overlap=True)
f.show_contour(slit_hdu,
               levels=[0.01, 10.0],
               filled=True, alpha=0.4, colors='#00a0ff', overlap=True)
f.add_grid()
f.grid.set_color('white')
f.grid.set_alpha(0.2)
f.save(figfile)
f.save(figfile.replace('.pdf', '.jpg'), dpi=300, format='jpeg')
print(figfile)
