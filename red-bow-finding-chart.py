import sys
import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
import aplpy
import numpy as np
from matplotlib import cm, colors

fn = 'WFI-Images/Orion_H_A_deep.fits'
figfile = sys.argv[0].replace('.py', '.pdf')
regfile = 'Alba-Regions-2016-10/bowshocks_arcs.reg'
f = aplpy.FITSFigure(fn)
f.recenter(83.6458, -5.4167, width=0.15, height=0.15)
f.show_grayscale(pmin=65.0, pmax=95, stretch='sqrt')
f.add_grid()
f.grid.set_color('white')
f.grid.set_alpha(0.2)
f.show_regions(regfile)
f.save(figfile)
print(figfile)
