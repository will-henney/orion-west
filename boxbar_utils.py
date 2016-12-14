import os
import glob
import numpy as np
import pyregion
import skimage
import skimage.morphology
import skimage.draw
import rasterio
import rasterio.features
import shapely
import shapely.geometry
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

def load_regions(region_file):
    with open(region_file) as f:
        region_string = f.read()
    # Workaround for bug in pyregion.parse when color is of form '#fff'
    region_string = region_string.replace('color=#', 'color=')
    regions = pyregion.parse(region_string)
    return regions

def sort_bars_into_knots(shapelist):
    """Make a dict of knots, each with a list of bar parameters"""
    knots = {}
    for shape in shapelist:
        if shape.name == 'line' and shape.coord_format == 'image':
            _, shape_dict = shape.attr
            tags = shape_dict['tag']
            assert len(tags) == 1, 'Each bar should belong to one knot only'
            for knot_id in tags:
                if not knot_id in knots:
                    knots[knot_id] = {'coords': [], 'width': [], 'vel': []}
                knots[knot_id]['coords'].append(shape.coord_list)
                knots[knot_id]['width'].append(int(shape_dict['width']))
                knots[knot_id]['vel'].append(int(shape_dict['text']))
    return knots

MAP_SHAPE = 2048, 2048
def blank_mask(shape=MAP_SHAPE):
    """Make a blank mask"""
    mask = np.zeros(MAP_SHAPE, dtype=bool)
    return mask


def paint_line_on_mask(x1, y1, x2, y2, mask):
    """Paint a single line on an image mask"""
    # Draw line between endpoints
    # (skimage always puts rows before columns)
    rr, cc = skimage.draw.line(y1, x1, y2, x2)
    mask[rr, cc] = True
    return mask


def nint(x):
    """Nearest integer value"""
    return int(x + 0.5)


def find_hull_mask(line_coord_list, min_size=4.0):
    """Given a list of line regions return an image mask of enclosing hull"""
    # Start with all blank
    mask = blank_mask()
    for x1, y1, x2, y2 in line_coord_list:
        # Add on each bar
        mask = paint_line_on_mask(nint(x1), nint(y1), nint(x2), nint(y2), mask)
    # Find the convex hull that encloses all the bars
    mask = skimage.morphology.convex_hull_image(mask)
    if min_size > 0.0:
        selem = skimage.morphology.disk(min_size/2)
        mask = skimage.morphology.dilation(mask, selem=selem)
    return mask

def vector_polygon_from_mask(mask, tolerance=2):
    """Find vertices from a polygonal image mask r
    Return vertices as two arrays: x, y"""
    # Use rasterio to get corners of polygon
    shapes_generator = rasterio.features.shapes(mask.astype(np.uint8), mask=mask)
    # There should be only one of them, and we throw away the image value
    shape_dict, _ = next(shapes_generator)
    # Now import it into shapely (note that asPolygon does not work)
    polygon = shapely.geometry.asShape(shape_dict)
    # And simplify the boundary 
    boundary = polygon.boundary.simplify(tolerance)
    # Return array of x values, array of y values
    return boundary.xy

def polygon_region_string(x, y, color=None, text=None):
    """Return pyregion polygon region as string"""
    coords = []
    for xx, yy in zip(x, y):
        coords.extend([xx, yy])
    string = 'polygon({})'.format(','.join(['{:.1f}'.format(v) for v in coords]))
    string += ' # '
    if color is not None:
        string += 'color={{{}}} '.format(color)
    if text is not None:
        string += 'text={{{}}} '.format(text)
    return string

BAR_REGION_HEADER = """# Region file format: DS9 version 4.1
global color=yellow dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image
"""

def convert_bars_to_knots(bar_region_file, knot_region_file):
    """Write DS9 region file of polygonal knots

    The knots enclose various bars, which are read from another region
    file in which each bar is tagged with the knot that it belongs to

    """

    bars = load_regions(bar_region_file)
    knots = sort_bars_into_knots(bars)
    coord_ids = find_knot_coord_ids(knots)
    region_strings = []
    for knot_id, knot_data in knots.items():
        m = find_hull_mask(knot_data['coords'])
        x, y = vector_polygon_from_mask(m)
        region_strings.append(polygon_region_string(
            x, y, text=coord_ids[knot_id]))
    with open(knot_region_file, 'w') as f:
        f.write(BAR_REGION_HEADER + '\n'.join(region_strings))

def radec2ow(ra, dec):
    """Implement the O'Dell & Wen coordinate designation

    Note (G1): Sources identified as <[RRS2008] NNNN-NNNN> in Simbad:
         * NNNN-NNN  : MSSs-MSS   (position: 5 3M SS.s -5 2M SS)
         * NNN-NNN   : SSs-MSS    (position: 5 35 SS.s -5 2M SS)
         * NNN-NNNN  : SSs-MMSS   (position: 5 35 SS.s -5 MM SS)
         * NNNN-NNNN : MSSs-MMSS  (position: 5 3M SS.s -5 MM SS)
    """
    c = SkyCoord(ra, dec, unit='deg')
    assert c.ra.hms.h == 5.0
    assert abs(c.ra.hms.m - 35) < 5.0
    rastring = '{:03d}'.format(int(0.5 + 10*c.ra.hms.s))
    if c.ra.hms.m != 35.0:
        rastring = str(int(c.ra.hms.m - 30.0)) + rastring
    assert c.dec.dms.d == -5.0
    decstring = '{:02d}'.format(int(-c.dec.dms.m))
    decstring += '{:02d}'.format(int(0.5 - c.dec.dms.s))
    if decstring.startswith('2'):
        decstring = decstring[1:]
    return '-'.join([rastring, decstring])


def find_knot_coord_ids(knots):
    """Find coordinate ID for each knot"""
    coord_ids = {}
    imhdu = fits.open('new-slits-ha-allvels.fits')['scaled']
    imwcs = WCS(imhdu.header)
    for knot_id, knot_data in knots.items():
        x = [0.5*(x1 + x2) for x1, _, x2, _ in knot_data['coords']]
        y = [0.5*(y1 + y2) for _, y1, _, y2 in knot_data['coords']]
        weights = knot_data['width']
        x0 = np.average(x, weights=weights)
        y0 = np.average(y, weights=weights)
        [ra], [dec] = imwcs.all_pix2world([x0], [y0], 0)
        coord_ids[knot_id] = radec2ow(ra, dec)
        v0 = np.average(knot_data['vel'], weights=weights)
        coord_ids[knot_id] += ' ({})'.format(int(round(v0/5.0)*5.0))
    return coord_ids

def find_bar2knot_map(shapelist, coord_ids):
    """Create a mapping between bar and knot coordinate ID.
    Bar is specified by tuple: (x1, y1, x2, y2)"""
    map_ = {}
    for shape in shapelist:
        if shape.name == 'line' and shape.coord_format == 'image':
            _, shape_dict = shape.attr
            knot_id, = shape_dict['tag']
            key = tuple(['{:.1f}'.format(_) for _ in shape.coord_list])
            map_[key] = coord_ids[knot_id]
    return map_

# This is largely copied from up above
FITS_DIR = 'Calibrated/BGsub'
BOX_FMT = 'box({:.1f},{:.1f},{:.1f},{:.1f},{:.1f}) # text={{{}}}'
BOX_HEADER = """global color=white font="helvetica 5 normal"
image
"""
def update_box_file(box_file, bar2knot_map):
    """Add the knot coordinate ID into all the boxes"""
    # Each box_file has the boxes for one slit
    slit_boxes = pyregion.open(box_file)
    # Also open the fits file associated with this slit
    slit_name = box_file.replace(
        os.path.join(REGION_DIR, 'pvboxes-'), '').replace('.reg', '')
    fits_name = os.path.join(FITS_DIR, slit_name) + '-ha-vhel.fits'
    hdu, = fits.open(fits_name)
    # Get the normal WCS together with the 'V' alternative WCS
    w = WCS(hdu)
    ww = WCS(hdu, key='V')
    newboxes = []
    for b in slit_boxes:
        # Check that it really is a box and that coordinates are in
        # the correct format
        if b.name == 'box' and b.coord_format == 'image':
            # Extract slit pixel coordinates
            # ii is along velocity axis
            # jj is along slit length
            ii, jj, dii, djj, angle = b.coord_list
            # Find the start/end coordinate along the slit
            jj1, jj2 = jj - 0.5*djj, jj + 0.5*djj
            # Then use alt WCS to find velocity plus both x and y
            [v, _], [x1, x2], [y1, y2] = ww.all_pix2world(
                [ii, ii], [jj1, jj2], [0, 0], 0)
            # Convert velocity from m/s -> km/s
            v /= 1000.0
            # Use tuple of rounded coordinates as the key
            key = tuple(['{:.1f}'.format(_) for _ in [x1, y1, x2, y2]])
            try: 
                coord_id = bar2knot_map[key]
                bars_remaining.remove(key)
            except KeyError:
                print('  '*2, 'Failed to match key', key)
                print('  '*3, ii, jj, dii, djj)
                if v > 0.0:
                    coord_id = 'RED KNOT ({:+.0f})'.format(5.0*round(v/5))
                else:
                    coord_id = 'LOST KNOT ({:+.0f})'.format(5.0*round(v/5))
                print('  '*3, coord_id)

            newbox = BOX_FMT.format(ii, jj, dii, djj, angle, coord_id)
            newboxes.append(newbox)


    newbox_file = box_file.replace('pvboxes', 'pvboxes-knots')
    with open(newbox_file, 'w') as f:
        f.write(BOX_HEADER)
        f.write('\n'.join(newboxes))
    return None

REGION_DIR = 'Will-Regions-2016-12'
bars_remaining = []
def retrofit_knots_on_boxes():
    boxfiles = glob.glob(os.path.join(REGION_DIR, 'pvboxes-[XY]*.reg'))
    barfiles = glob.glob(os.path.join(REGION_DIR, 'bars-from-boxes-*-groups.reg'))

    # Get list of all knots with data
    knots = {}
    bar2knot_map = {}
    print('Creating Bar -> Knot map ...')
    for barfile in barfiles:
        print('  ', barfile)
        bars = load_regions(barfile)
        knots.update(sort_bars_into_knots(bars))
        coord_ids = find_knot_coord_ids(knots)
        bar2knot_map.update(find_bar2knot_map(bars, coord_ids))

    print('Updating boxes with knot info ...')
    bars_remaining[:] = list(bar2knot_map.keys())
    for boxfile in boxfiles:
        print('  ', boxfile)
        update_box_file(boxfile, bar2knot_map)

    if bars_remaining:
        print('Bars remaining:')
        for bar in bars_remaining:
            print('  ', bar)
