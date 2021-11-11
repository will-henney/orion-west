import sys
import logging
logging.disable(logging.INFO)
import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
import aplpy
import numpy as np
from matplotlib import cm, colors

try:
    vclass = sys.argv[1]
except IndexError:
    vclass = 'fast'

fn = 'WFI-Images/Orion_H_A_deep.fits'
figfile = sys.argv[0].replace('.py', '-{}.pdf'.format(vclass))
# regfile = 'Will-Regions-2016-12/knots-{}-wcs.reg'.format(vclass)
regfile = 'Will-Regions-2016-12/bars-from-boxes-{}-groups-wcs.reg'.format(vclass)
f = aplpy.FITSFigure(fn)
f.recenter(83.6458, -5.4167, width=0.15, height=0.15)
f.show_grayscale(pmin=65.0, pmax=95, stretch='sqrt')
f.add_grid()
f.grid.set_color('red')
f.grid.set_alpha(0.2)
f.show_regions(regfile)
f.save(figfile)
print(figfile, end='')
