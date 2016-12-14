import sys
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import pyregion

try: 
    knot_region_file = sys.argv[1]
    line_id = sys.argv[2]
    region_frame = sys.argv[3]
except IndexError:
    sys.exit('Usage: {} KNOT_REGION_FILE (ha|nii) (linear|image)'.format(sys.argv[0]))

with open(knot_region_file) as f:
    knot_region_string = f.read()
# Workaround for bug in pyregion.parse when color is of form '#fff'
knot_region_string = knot_region_string.replace('color=#', 'color=')
knot_regions = pyregion.parse(knot_region_string)

knot_dict = {knot.attr[1]['text']: pyregion.ShapeList([knot])
             for knot in knot_regions
             if 'text' in knot.attr[1] and knot.name == 'ellipse'}

tab = Table.read('alba-knots-frompdf.tab',
                 format='ascii', delimiter='\t')
vcol = {'ha': 'V(Ha)', 'nii': 'V([N II])'}
wcol = {'ha': 'W(Ha)', 'nii': 'W([N II])'}

imhdu = fits.open('new-slits-{}-allvels.fits'.format(line_id))['scaled']
imwcs = WCS(imhdu.header)

ny, nx = imhdu.data.shape
Y, X = np.mgrid[0:ny, 0:nx]

speclist = glob.glob('Calibrated/BGsub/*-{}-vhel.fits'.format(line_id))
specdict = {fn.split('/')[-1].split('.')[0]: fits.open(fn)[0] for fn in speclist}

region_template = 'ellipse({1:.1f},{2:.1f},{3:.1f},{4:.1f},0) # text={{{0}}}'
region_header_lines = [
    'global color=green font="helvetica 10 normal"', 
    region_frame,
]
slit_region_dir = knot_region_file.replace('.reg', '-SLITS')
if not os.path.isdir(slit_region_dir):
    os.mkdir(slit_region_dir)

def look_for_velocity(knot_id, line_id):
    '''Try to parse something like "4299-524 (-70)"
    Returns (velocity, width)
    '''
    # Try and get something like "(-70)"
    maybe_parens = knot_id.split()[-1]
    if maybe_parens.startswith('(') and maybe_parens.endswith(')'):
        vstring = maybe_parens[1:-1]
        try:
            v0 = float(vstring)
            w = 30.0 if line_id == 'ha' else 20.0
        except ValueError:
            v0, w = None, None
    else:
        v0, w = None, None
    return v0, w

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

for specname, spechdu in specdict.items():
    print(specname)
    pvregions = []
    # WCS transform for the slit
    specwcs = WCS(spechdu.header, key='V')
    specmask = get_specmask(specwcs, imhdu.data.shape)
    if (X[specmask].max() - X[specmask].min()
        > Y[specmask].max() - Y[specmask].min()):
        orient = 'horizontal'
    else:
        orient = 'vertical'

    slit_region_file =  slit_region_dir + '/' + specname + '.reg'
    for knot_id, knot_region in knot_dict.items():
        # Find mask for knot and overlap with the slit mask
        knotmask = knot_region.get_mask(imhdu)
        overlap = knotmask & specmask
        # Number of pixels in overlap region
        n = overlap.sum()
        if n > 0:
            # New Will knots have velocity encoded in knot_id
            v0, dv = look_for_velocity(knot_id, line_id)
            if v0 is None:
                # But fall back on table look-up for the original Alba knots
                if not knot_id in tab['knot']:
                    # If not there either, then skip this region 
                    print('Warning: Knot', knot_id, 'not found in table!')
                    continue
                # Extract row from table
                knotrow = tab[tab['knot'] == knot_id][0]
                v0, dv = knotrow[vcol[line_id]], knotrow[wcol[line_id]]
            if region_frame == 'image':
                # Find j-pixel coordinates along the slit that correspond to this knot
                _, jslit, _ = specwcs.all_world2pix([0]*n, X[overlap], Y[overlap], 0)
                j1, j2 = jslit.min(), jslit.max()
                # Find i-pixel coordinates coresponding to knot velocity +/- width
                # Make sure to convert from km/s -> m/s since wcs is in SI
                v1, v2 = 1000*(v0 - dv/2), 1000*(v0 + dv/2)

                [i1, i2], _, _ = specwcs.all_world2pix([v1, v2], [0, 0], [0, 0], 0)
                i0, w = 0.5*(i1 + i2), i2 - i1
                j0, h = 0.5*(j1 + j2), j2 - j1
                pvregions.append([knot_id, i0, j0, w, h])
            elif region_frame == 'linear':
                # Regions written in x = km/s and y = map X or Y,
                # depending on orientation
                S = X if orient == 'horizontal' else Y
                s1, s2 = S[overlap].min(), S[overlap].max()
                s0, ds = 0.5*(s1 + s2), (s2 - s1)
                pvregions.append([knot_id, v0, s0, dv, ds])    
            else:
                raise NotImplementedError('Region frame must be "linear" or "image"')
    # If there are any knot regions for this slit, then write them out
    if pvregions:
        print(len(pvregions), 'regions found')
        region_lines = [region_template.format(*data) for data in pvregions]
        with open(slit_region_file, 'w') as f:
            f.write('\n'.join(region_header_lines + region_lines))
