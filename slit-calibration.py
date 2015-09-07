# [[file:alba-orion-west.org::*Imports][Imports:1]]
import os
import numpy as np
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
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
    # Find nearest integer pixel
    ii, jj = np.floor(xi + 0.5), np.floor(yj + 0.5)
    print(ra[::100], dec[::100])
    print(ii[::100], jj[::100])
    return np.array([image[j, i] for i, j in zip(ii, jj)])
# Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1 ends here

# [[file:alba-orion-west.org::*Construct%20the%20synthetic%20slit%20from%20the%20reference%20image][Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1]]
wfi_dir = '/Users/will/Work/OrionTreasury/wfi'
photom, = fits.open(os.path.join(wfi_dir, 'Orion_H_A_deep.fits'))
wphot = WCS(photom.header)
# Construct\ the\ synthetic\ slit\ from\ the\ reference\ image:1 ends here

# [[file:alba-orion-west.org::*Find%20the%20world%20coordinates%20of%20each%20pixel%20along%20the%20slit][Find\ the\ world\ coordinates\ of\ each\ pixel\ along\ the\ slit:1]]
def find_slit_coords(db, hdr):
    """Find the coordinates of all the pixels along a spectrograph slit

    Input arguments are a dict-like 'db' of hand-measured values (must
    contain 'saxis', 'islit' and 'shift') and a FITS header 'hdr' from
    the image+slit exposure.

    Returns a dict of 'ds' (slit pixel scale), 'PA' (slit position
    angle), 'RA' (array of RA values in degrees along slit), 'Dec'
    (array of Dec values in degrees along slit)

    """
    jstring = str(db['saxis'])  # which image axis lies along slit
    dRA_arcsec = hdr['CD1_'+jstring]*3600*np.cos(np.radians(hdr['CRVAL2']))
    dDEC_arcsec = hdr['CD2_'+jstring]*3600
    ds = np.hypot(dRA_arcsec, dDEC_arcsec)
    PA = np.degrees(np.arctan2(dRA_arcsec, dDEC_arcsec))

    # Pixel coords of each slit pixel on image (in 0-based convention)
    if jstring == '1':
        # Slit is horizontal in IMAGE coords
        ns = hdr['NAXIS1']
        iarr = np.arange(ns) - float(db['shift'])
        jarr = np.ones(ns)*float(db['islit'])
    elif jstring == '2':
        # Slit is vertical in IMAGE coords
        ns = hdr['NAXIS2']
        iarr = np.ones(ns)*float(db['islit'])
        jarr = np.arange(ns) - float(db['shift'])
    else:
        raise ValueError('Slit axis (saxis) must be 1 or 2')

    # Convert to world coords, using the native frame
    w = WCS(hdr)
    observed_frame = w.wcs.radesys.lower()
    coords = SkyCoord(*w.all_pix2world(iarr, jarr, 0),
                      unit=(u.deg, u.deg), frame=observed_frame)

    # Make sure to return the coords in the ICRS frame
    return {'ds': ds, 'PA': PA,
            'RA': coords.icrs.ra.value,
            'Dec': coords.icrs.dec.value}
# Find\ the\ world\ coordinates\ of\ each\ pixel\ along\ the\ slit:1 ends here

# [[file:alba-orion-west.org::*Make%20some%20useful%20and%20pretty%20plots][Make\ some\ useful\ and\ pretty\ plots:1]]
def make_three_plots(spec, calib, prefix):
    assert spec.shape == calib.shape
    fig, axes = plt.subplots(3, 1)

    vmin, vmax = 0.0, 2*np.median(calib) 

    # First, plot two profiles against each other to check for zero-point offsets
    axes[0].plot(calib, spec, '.')
    axes[0].plot([vmin, vmax], [vmin, vmax], '-')
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_ylim(vmin, vmax)

    # Second, plot each against slit pixel to check spatial offset
    ypix = np.arange(len(calib))
    axes[1].plot(ypix, calib, ypix, spec)
    axes[1].set_xlim(0.0, ypix.max())
    axes[1].set_ylim(vmin, vmax)

    # Third, plot ratio to look for linear trends
    axes[2].plot(ypix, spec/calib)
    axes[2].set_xlim(0.0, ypix.max())
    axes[2].set_ylim(0.0, 1.5)

    fig.set_size_inches(5, 8)
    fig.savefig(prefix+'.png')
# Make\ some\ useful\ and\ pretty\ plots:1 ends here

# [[file:alba-orion-west.org::*Loop%20over%20the%20slit%20positions%20and%20do%20the%20stuff][Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1]]
for row in tab:
    if row['Dataset'] != '2006-02':
        continue
    print(row)
    imslitfile = find_fits_filepath(row, 'image')
    specfile = find_fits_filepath(row, 'fullspec')
    hafile = find_fits_filepath(row, 'ha')
    niifile = find_fits_filepath(row, 'nii')
    imhdu = fits.open(imslitfile)[0]
    spechdu = fits.open(specfile)[0]
    hahdu = fits.open(hafile)[0]
    niihdu = fits.open(niifile)[0]

    # World coordinates along slit
    slit_coords = find_slit_coords(row, imhdu.header)

    # Find synthetic profile from calibration image
    calib_profile = slit_profile(slit_coords['RA'], slit_coords['Dec'],
                                 photom.data, wphot)

    # Find actual profile along slit from spectrum
    wavaxis = 1                 # TODO need to check if this is always true
    ha_profile = hahdu.data.sum(axis=wavaxis)
    nii_profile = niihdu.data.sum(axis=wavaxis)
    spec_profile = (ha_profile+1.333*nii_profile)/row['norm']
    plt_prefix = 'plots/{}-{}-calib'.format(row['Dataset'], row['imid'])
    make_three_plots(spec_profile, calib_profile, plt_prefix)
# Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1 ends here

# [[file:alba-orion-west.org::*Test%20what%20is%20going%20on][Test\ what\ is\ going\ on:1]]
# print(wphot.wcs)
# for row in tab:
#     print([row[x] for x in ('Dataset', 'imid', 'specid', 'Notes')])
# Test\ what\ is\ going\ on:1 ends here
