import sys
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy import coordinates as coord
import astropy.units as u
import owutil
import pyregion
import matplotlib
matplotlib.use("Agg")
import aplpy
from matplotlib import pyplot as plt
from matplotlib import cm, colors

try: 
    dataset_id = sys.argv[1]
except IndexError:
    sys.exit('Usage: {} DATASET_ID'.format(sys.argv[0]))

glob_patterns = {
    'll2': 'XX1[123]*',
    'horiz': 'YY[01][019]*',
    'vert': 'XX1[56]??-2010-01-*',
    'east': 'YY1[234]*',
    'll1': 'XX0[45]*',
}

hfiles = glob.glob('Calibrated/BGsub/' + glob_patterns[dataset_id] + '-ha-vhel.fits')
nfiles = glob.glob('Calibrated/BGsub/' + glob_patterns[dataset_id] + '-nii-vhel.fits')
map_fn = 'new-slits-ha-allvels.fits'

slit_region_dir = 'Alba-Regions-2016-10/blue_knots_final-SLITS'

figwidth, figheight = 12, 12
subplot_windows = { 
    # x0, y0, dx, dy in fractions of figure size
    'ha': [0.08, 0.4, 0.44, 0.58],
    'nii': [0.54, 0.4, 0.44, 0.58],
    'map': [0.15, 0.06, 0.7, 0.28]
}

XYcenter = {
    'll2': 1290.0,
    'horiz': 1475.0,
    'vert': 1050.0,
    'east': 800.0,
    'll1': 925.0,
}
XYlength = {
    'll2': 720.0,
    'horiz': 720.0,
    'vert': 720.0,
    'east': 1200.0,
    'll1': 720.0,
}

pv_contour_style = {
    'levels': [0.05, 0.0707, 0.1, 0.141, 0.2, 0.282, 0.4, 0.564],
    'colors': 'k',
}
pv_colorscale_style = {
    'aspect': 'auto', 'cmap': 'CMRmap', 'stretch': 'sqrt',
    'vmin': -0.0003, 'vmax': 0.05}

blackbox = {'facecolor': 'black', 'alpha': 0.7}

def fix_pv_wcs(hdr, use_celestial=False):
    newhdr = hdr.copy()

    newhdr['CTYPE1'] = 'offset'

    if use_celestial:
        for k in 'CTYPE', 'CRVAL', 'CRPIX', 'CDELT':
            newhdr[k+'2'] = hdr[k+'3A']
        newhdr['CDELT2'] *= hdr['PC3_2A']

    return newhdr

def invert_second_fits_axis(hdu):
    """Flip the second (Y) axis of a FITS image in `hdu`

Modifies the HDU in place.  Does not return a value

    """
    # Flip the y-axis of the data array, which is the first python axis
    hdu.data = hdu.data[::-1, :]
    # We need to also operate on the alternative 'A' WCS because we
    # use it for placing the OW labels
    for key in '', 'A':
        # Move reference pixel
        hdu.header['CRPIX2' + key] = (1 + hdu.header['NAXIS2']
                                      - hdu.header['CRPIX2' + key])
        # Change sign of pixel scale
        hdu.header['CDELT2' + key] *= -1.0

    return None

def invert_second_region_axis(regions, ny):
    """Flip the second (Y) axis of RegionList `regions`.  

Second argument, `ny` is length of the y-axis.  

All y coordinates are transformed to 1 + ny - y

Modifies the HDU in place.  Does not return a value

    """
    for region in regions:
        region.coord_list[1] = 1 + ny - region.coord_list[1]
    return None

def get_specmask(specwcs, imshape, slit_pix_width=4):
    """Find image mask that corresponds to a given slit

    `specwcs` is a WCS for the slit spectrum, which should have the second pixel dimension along the slit and the X, Y coords as the second and third world coordinates.  `imshape` is the shape (ny, nx) of the desired mask
    """
    
    # Length of slit in slit pixels
    ns = specwcs._naxis2
    # Shape of image mask
    ny, nx = imshape

    # Coord arrays along the slit
    V, X, Y = specwcs.all_pix2world([0]*ns, range(ns), [0]*ns, 0)

    # Initialize empty mask
    specmask = np.zeros(imshape).astype(bool)

    # Fill in the mask pixel-by-pixel along the slit
    for x, y in zip(X, Y):
        # Find output pixels corresponding to corners of slit pixel
        # (approximate as square)
        i1 = int(0.5 + x - slit_pix_width/2)
        i2 = int(0.5 + x + slit_pix_width/2)
        j1 = int(0.5 + y - slit_pix_width/2)
        j2 = int(0.5 + y + slit_pix_width/2)
        # Make sure we don't go outside the output grid
        i1, i2 = max(0, i1), max(0, i2)
        i1, i2 = min(nx, i1), min(nx, i2)
        j1, j2 = max(0, j1), max(0, j2)
        j1, j2 = min(ny, j1), min(ny, j2)

        specmask[j1:j2, i1:i2] = True

    return specmask

for hfn, nfn in zip(hfiles, nfiles):
    fig = plt.figure(figsize=(figwidth, figheight))

    hhdu = fits.open(hfn)[0]
    nhdu = fits.open(nfn)[0]
    map_hdu = fits.open(map_fn)['scaled']
    hregfile = os.path.basename(hfn).replace('.fits', '.reg')
    try:
        regions = pyregion.open(os.path.join(slit_region_dir, hregfile))
    except FileNotFoundError:
        regions = None

    all_slits = np.isfinite(map_hdu.data)
    this_slit = get_specmask(WCS(hhdu.header, key='V'), map_hdu.data.shape)
    map_hdu.data[all_slits] = 1.0
    map_hdu.data[this_slit] = 10.0

    hhdu.header = fix_pv_wcs(hhdu.header)
    nhdu.header = fix_pv_wcs(nhdu.header)

    if hhdu.header['CDELT2'] < 0.0:
        invert_second_fits_axis(hhdu)
        invert_second_fits_axis(nhdu)
        if regions is not None:
            invert_second_region_axis(regions, ny=hhdu.header['NAXIS2'])

    hf = aplpy.FITSFigure(data=hhdu, figure=fig, subplot=subplot_windows['ha'])
    nf = aplpy.FITSFigure(data=nhdu, figure=fig, subplot=subplot_windows['nii'])
    mf = aplpy.FITSFigure(data=map_hdu, figure=fig, subplot=subplot_windows['map'])

    for f in hf, nf:
        f.recenter(25.0, XYcenter[dataset_id],
                   width=300.0, height=XYlength[dataset_id])
        f.show_colorscale(**pv_colorscale_style)
        f.show_contour(**pv_contour_style)
        f.add_grid()
        f.grid.set_alpha(0.3)

    if regions is not None:
        hf.show_regions(regions, text_offset=0.0)

    nf.hide_yaxis_label()
    nf.hide_ytick_labels()

    # Add labels for OW coords at y tick points
    w = WCS(nhdu.header, key='A').celestial
    yticks = nf._ax1.get_yticks()
    cc = coord.SkyCoord.from_pixel(yticks, np.zeros_like(yticks), w)
    ows = [owutil.ow_from_coord(c) for c in cc]
    x = nf._ax1.get_xticks()[-1] 
    for y, ow in zip(yticks, ows):
        nf._ax1.text(x, y, ow,
                     bbox=blackbox, color='orange', ha='right', va='center')

    # Add labels for each emission line
    hf.add_label(0.95, 0.95, 'H alpha', relative=True,
                 bbox=blackbox, size='large',
                 horizontalalignment='right', color='yellow')
    nf.add_label(0.05, 0.95, '[N II]', relative=True,
                 bbox=blackbox, size='large',
                 horizontalalignment='left', color='yellow')

    # Deal with the slit position map at the bottom
    mf.recenter(83.6875, -5.4167, width=0.15, height=0.15)
    mf.show_colorscale(aspect='equal', vmin=0.0, vmax=5.0, cmap='RdPu')
    mf.add_grid()
    mf.grid.set_color('black')
    mf.grid.set_alpha(0.1)
    # Include the WFI map for orientation
    mf.show_contour('WFI-Images/Orion_H_A_deep.fits',
                    levels=[3, 4.5, 6, 9, 12, 15, 20, 40, 80, 160, 320],
                    filled=True, alpha=0.3, cmap='Blues',
                    norm=colors.LogNorm(), vmin=3.0, vmax=400.0, overlap=True,
    )

    figfile = hfn.replace('-ha-vhel.fits', '-plot.jpg')
    fig.savefig(figfile, dpi=300)
    print(figfile)
