# [[file:alba-orion-west.org::*Imports][Imports:1]]
import os
import numpy as np
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
# Imports:1 ends here

# [[file:alba-orion-west.org::*Read%20in%20the%20table%20of%20all%20slits][Read\ in\ the\ table\ of\ all\ slits:1]]
converters = {'imid': [astropy.io.ascii.convert_numpy(np.str)]}
tab = Table.read('all-slits-input.tab',
                 format='ascii.tab', converters=converters)
# Read\ in\ the\ table\ of\ all\ slits:1 ends here

# [[file:alba-orion-west.org::*Fits%20files%20for%20the%20spectra%20and%20image+slit][Fits\ files\ for\ the\ spectra\ and\ image+slit:1]]
file_templates = {
    'fullspec' : {
        '2006-02': 'Work/SPM2005/pp{}.fits',
        '2007-01b': 'Work/SPM2007/Reduced/HH505/slits/reducciones/spec{}.fits',
        '2007-01': 'Work/SPM2007/Reduced/spec{}-transf.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/spm{}h.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spm{}_bcr.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spm{}_bcrx.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}o_bcrx.fits',
    },
    'ha' : {
        '2006-02': 'Work/SPM2007/Reduced/HH505/slits/SPMha/spec{}-halpha.fits',
        '2007-01b': 'Work/SPM2007/Reduced/HH505/slits/reducciones/spec{}-ha.fits',
        '2007-01': 'Work/SPM2007/Reduced/spec{}-ha.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/spec{}-ha.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spec{}-ha.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spec{}-ha.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}o-ha.fits',
    },
    'nii' : {
        '2006-02': 'Work/SPM2007/Reduced/HH505/slits/SPMnii/spec{}-nii.fits',
        '2007-01b': 'Work/SPM2007/Reduced/HH505/slits/reducciones/spec{}-nii.fits',
        '2007-01': 'Work/SPM2007/Reduced/spec{}-nii.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/spec{}-nii.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spec{}-nii.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spec{}-nii.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}o-nii.fits',
    },
    'image' : {
        '2006-02': 'Dropbox/Papers/LL-Objects/feb2006/pp{}-ardec.fits',
        '2007-01b': 'Work/SPM2007/Reduced/HH505/slits/reducciones/spm{}-ardec.fits',
        '2007-01': 'Work/SPM2007/Reduced/spm{}-ardec.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/posiciones/spm{}-ardec.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spm{}_ardec.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spm{}-ardec.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}-ardec.fits',
    },
}

def find_fits_filepath(db, filetype):
    """Return path to the FITS file for an image or spectrum 
    """
    id_ = db['imid'] if filetype == 'image' else db['specid']
    id_ = str(id_)
    if filetype in ('ha', 'nii') and db['Dataset'] not in ['2013-12']:
        id_ = id_.split('-')[0]
    template = file_templates[filetype][db['Dataset']]
    print('ID =', id_, 'Template =', template)
    path = template.format(id_)
    homedir = os.path.expanduser('~')
    return os.path.join(homedir, path)
# Fits\ files\ for\ the\ spectra\ and\ image+slit:1 ends here

# [[file:alba-orion-west.org::*Construct%20the%20synthetic%20slit%20from%20the%20reference%20image][Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1]]
def slit_profile(ra, dec, image, wcs):
    """
    Find the image intensity for a list of positions (ra and dec)
    """
    xi, yj = wcs.all_world2pix(ra, dec, 0)
    ii, jj = int(xi + 0.5), int(yi + 0.5)
    return np.array([image[j, i] for i, j in zip(ii, jj)])
# Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1 ends here

# [[file:alba-orion-west.org::*Construct%20the%20synthetic%20slit%20from%20the%20reference%20image][Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1]]
wfi_dir = '/Users/will/Work/OrionTreasury/wfi'
photom, = fits.open(os.path.join(wfi_dir, 'Orion_H_A_deep.fits'))
wphot = WCS(photom.header)
# Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1 ends here

# [[file:alba-orion-west.org::*Loop%20over%20the%20slit%20positions%20and%20do%20the%20stuff][Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1]]
for row in tab:
    print(row)
    imslitfile = find_fits_filepath(row, 'image')
    specfile = find_fits_filepath(row, 'fullspec')
    hafile = find_fits_filepath(row, 'ha')
    fits.open(imslitfile).info()
    fits.open(specfile).info()
    fits.open(hafile).info()
# Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1 ends here
