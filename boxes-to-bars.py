import pyregion
from astropy.io import fits
from astropy.wcs import WCS
import glob
import os
import seaborn as sns

DEBUG = True
REGION_DIR = 'Will-Regions-2016-12'
FITS_DIR = 'Calibrated/BGsub'
BOX_PATTERN = 'pvboxes-*.reg'
BAR_HEADER = '''# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image
'''
BAR_FMT = ('line({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) # '
           + 'line=0 0 color={color} width={width} '
           + 'text={{{v:d}}} dash={dashed}')
BAR_FILE = 'bars-from-boxes.reg'

BRIGHT_LEVELS = [0.001, 0.003, 0.009, 0.027]
def find_width(b, hdu):
    shapelist = pyregion.ShapeList([b])
    m = shapelist.get_mask(hdu=hdu)
    box_bright = hdu.data[m].mean()
    width = 1
    dashed = 1
    for ib, blevel in enumerate(BRIGHT_LEVELS):
        if box_bright >= blevel:
            width = ib + 1
            dashed = 0
    return width, dashed


VMIN, VMAX = -110.0, 0.0
NC = int(VMAX - VMIN) + 1
rgblist = sns.hls_palette(NC)
def find_color(v):
    ic = int(VMAX - v)
    ic = max(0, min(ic, NC-1))
    r, g, b = rgblist[ic]
    return '#{:01x}{:01x}{:01x}'.format(int(16*r), int(16*g), int(16*b))


box_files = glob.glob(os.path.join(REGION_DIR, BOX_PATTERN))

VLIMITS = {
    'all': [-200.0, 200.0],
    'slow': [-45.0, 0.0],
    'fast': [-80.0, -35.0],
    'ultra': [-150.0, -70.0]}

bar_lists = {'all': [], 'slow': [], 'fast': [], 'ultra': []}
for box_file in box_files:
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
    # Check if horizontal or vertical
    is_horizontal = slit_name.startswith('YY')
    if DEBUG:
        print('Extracting boxes from', slit_name)
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

            width, dashed = find_width(b, hdu)
            color = find_color(v)

            bar_region = BAR_FMT.format(
                x1=x1, y1=y1, x2=x2, y2=y2,
                v=int(v), width=width, dashed=dashed, color=color)

            for vclass, (v1, v2) in VLIMITS.items():
                if v1 <= v <= v2:
                    bar_lists[vclass].append(bar_region)


for vclass, bar_list in bar_lists.items():
    bar_file = BAR_FILE.replace('.reg', '-' + vclass + '.reg')
    with open(os.path.join(REGION_DIR, bar_file), 'w') as f:
        f.write(BAR_HEADER + '\n'.join(bar_list))
