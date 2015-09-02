# [[file:alba-orion-west.org::*Imports][Imports:1]]
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import os
# Imports:1 ends here

# [[file:alba-orion-west.org::*Read%20in%20the%20table%20of%20all%20slits][Read\ in\ the\ table\ of\ all\ slits:1]]
tab = Table.read('all-slits-input.tab', format='ascii.tab')
# Read\ in\ the\ table\ of\ all\ slits:1 ends here

# [[file:alba-orion-west.org::*Construct%20the%20synthetic%20slit%20from%20the%20reference%20image][Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1]]
wfi_dir = '/Users/will/Work/OrionTreasury/wfi'
photom, = fits.open(os.path.join(wfi_dir, 'Orion_H_A_deep.fits'))
wphot = WCS(photom.header)
# Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1 ends here

# [[file:alba-orion-west.org::*Test%20what%20is%20going%20on][Test\ what\ is\ going\ on:1]]
print(wphot.wcs)
# for row in tab:
#     print([row[x] for x in ('Dataset', 'imid', 'specid', 'Notes')])
# Test\ what\ is\ going\ on:1 ends here
