# [[file:alba-orion-west.org::*Imports][Imports:1]]
import os
import sys
import numpy as np
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
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
        '2007-01': 'Work/SPM2007/Reduced/spec{}-ha-fix.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/spec{}-ha.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spec{}-ha.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spec{}-ha.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}o_sub-ha.fits',
    },
    'nii' : {
        '2006-02': 'Work/SPM2007/Reduced/HH505/slits/SPMnii/spec{}-nii.fits',
        '2007-01b': 'Work/SPM2007/Reduced/HH505/slits/reducciones/spec{}-nii.fits',
        '2007-01': 'Work/SPM2007/Reduced/spec{}-nii-fix.fits',
        '2010-01': 'Dropbox/SPMJAN10/reducciones/spec{}-nii.fits',
        '2013-02': 'Dropbox/SPMFEB13/WesternShocks/spec{}-nii.fits',
        '2013-12': 'Dropbox/papers/LL-Objects/SPMDIC13/spec{}-nii.fits',
        '2015-02': 'Dropbox/SPMFEB15/archivos/spm{}o_sub-nii.fits',
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

# [[nil][Find\ the\ world\ coordinates\ of\ each\ pixel\ along\ the\ slit:1]]
def find_slit_coords(db, hdr, shdr):
    """Find the coordinates of all the pixels along a spectrograph slit

    Input arguments are a dict-like 'db' of hand-measured values (must
    contain 'saxis', 'islit' and 'shift') and a FITS headers 'hdr' from
    the image+slit exposure and 'shdr' from a spectrum exposure

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
        ns = shdr['NAXIS1']
        iarr = np.arange(ns) - float(db['shift'])
        jarr = np.ones(ns)*float(db['islit'])
        try:
            image_binning = hdr['CBIN']
            spec_binning = shdr['CBIN']
        except KeyError:
            image_binning = hdr['CCDXBIN']
            spec_binning = shdr['CCDXBIN']
          
        # correct for difference in binning between the image+slit and the spectrum
        iarr *= spec_binning/image_binning
    elif jstring == '2':
        # Slit is vertical in IMAGE coords
        ns = shdr['NAXIS2']
        iarr = np.ones(ns)*float(db['islit'])
        jarr = np.arange(ns) - float(db['shift'])
        try:
            image_binning = hdr['RBIN']
            spec_binning = shdr['RBIN']
        except KeyError:
            image_binning = hdr['CCDYBIN']
            spec_binning = shdr['CCDYBIN']
          
        jarr *= spec_binning/image_binning
    else:
        raise ValueError('Slit axis (saxis) must be 1 or 2')

    print('iarr =', iarr[::100], 'jarr =', jarr[::100])
    # Also correct the nominal slit plate scale
    ds *= spec_binning/image_binning

    # Convert to world coords, using the native frame
    w = WCS(hdr)
    observed_frame = w.wcs.radesys.lower()
    # Note it is vital to ensure the pix2world transformation returns
    # values in the order (RA, Dec), even if the image+slit may have
    # Dec first
    coords = SkyCoord(*w.all_pix2world(iarr, jarr, 0, ra_dec_order=True),
                      unit=(u.deg, u.deg), frame=observed_frame)
    print('coords =', coords[::100])
    print('Binning along slit: image =', image_binning, 'spectrum =', spec_binning)
    # Make sure to return the coords in the ICRS frame
    return {'ds': ds, 'PA': PA,
            'RA': coords.icrs.ra.value,
            'Dec': coords.icrs.dec.value}
# Find\ the\ world\ coordinates\ of\ each\ pixel\ along\ the\ slit:1 ends here

# [[file:alba-orion-west.org::*Fit%20Chebyshev%20polynomials%20to%20along-slit%20variation][Fit\ Chebyshev\ polynomials\ to\ along-slit\ variation:1]]
def fit_cheb(x, y, npoly=3, mask=None):
    """Fits a Chebyshev poly to y(x) and returns fitted y-values"""
    fitter = fitting.LinearLSQFitter()
    p_init = models.Chebyshev1D(npoly, domain=[x.min(), x.max()])
    if mask is None:
        mask = np.ones_like(x).astype(bool)
    p = fitter(p_init, x[mask], y[mask])
    print(p)
    return p(x)
# Fit\ Chebyshev\ polynomials\ to\ along-slit\ variation:1 ends here

# [[file:alba-orion-west.org::*Make%20some%20useful%20and%20pretty%20plots][Make\ some\ useful\ and\ pretty\ plots:1]]
sns.set_palette('RdPu_d', 3)
def make_three_plots(spec, calib, prefix):
    assert spec.shape == calib.shape
    fig, axes = plt.subplots(3, 1)

    vmin, vmax = 0.0, np.median(calib) + 5*calib.std()

    ypix = np.arange(len(calib))
    ratio = spec/calib
    mask = (ypix > 10.0) & (ypix < ypix.max() - 10.0) \
           & (ratio > np.median(ratio) - 2*ratio.std()) \
           & (ratio < np.median(ratio) + 2*ratio.std()) 
    ratio_fit = fit_cheb(ypix, ratio, mask=mask)

    alpha = 0.8

    # First, plot two profiles against each other to check for zero-point offsets
    axes[0].plot(calib, spec/ratio_fit, '.', alpha=alpha)
    axes[0].plot([vmin, vmax], [vmin, vmax], '-', alpha=alpha)
    axes[0].set_xlim(vmin, vmax)
    axes[0].set_ylim(vmin, vmax)
    axes[0].set_xlabel('Calibration Image')
    axes[0].set_ylabel('Corrected Integrated Spectrum')

    # Second, plot each against slit pixel to check spatial offset
    axes[1].plot(ypix, calib, alpha=alpha, label='Calibration Image')
    axes[1].plot(ypix, spec/ratio_fit, alpha=alpha, lw=1.0, label='Corrected Integrated Spectrum')
    axes[1].plot(ypix, spec, alpha=alpha, lw=0.5, label='Uncorrected Integrated Spectrum')
    axes[1].set_xlim(0.0, ypix.max())
    axes[1].set_ylim(vmin, vmax)
    axes[1].legend(fontsize='xx-small', loc='lower right')
    axes[1].set_xlabel('Slit pixel')
    axes[1].set_ylabel('Profile')

    # Third, plot ratio to look for spatial trends
    axes[2].plot(ypix, ratio, alpha=alpha)
    axes[2].plot(ypix, ratio_fit, alpha=alpha)
    axes[2].set_xlim(0.0, ypix.max())
    axes[2].set_ylim(0.0, 1.5)
    axes[2].set_xlabel('Slit pixel')
    axes[2].set_ylabel('Ratio: Spec / Calib')

    fig.set_size_inches(5, 8)
    fig.tight_layout()
    fig.savefig(prefix+'.png', dpi=300)
# Make\ some\ useful\ and\ pretty\ plots:1 ends here

# [[file:alba-orion-west.org::*Use%20command%20line%20argument%20to%20restrict%20which%20datasets%20are%20processed][Use\ command\ line\ argument\ to\ restrict\ which\ datasets\ are\ processed:1]]
if len(sys.argv) > 1:
    selector_pattern = sys.argv[1]
else:
    selector_pattern = ''
# Use\ command\ line\ argument\ to\ restrict\ which\ datasets\ are\ processed:1 ends here

# [[file:alba-orion-west.org::*Loop%20over%20the%20slit%20positions%20and%20do%20the%20stuff][Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1]]
for row in tab:
    full_id = row['Dataset'] + '-' + row['imid']
    if not full_id.startswith(selector_pattern):
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
    slit_coords = find_slit_coords(row, imhdu.header, hahdu.header)

    # Find synthetic profile from calibration image
    calib_profile = slit_profile(slit_coords['RA'], slit_coords['Dec'],
                                 photom.data, wphot)

    # Find actual profile along slit from spectrum
    wavaxis = row['saxis'] - 1  # This always seems to be true
    ha_profile = (hahdu.data - row['zero']).sum(axis=wavaxis)
    nii_profile = (niihdu.data - row['zero']).sum(axis=wavaxis)
    spec_profile = (ha_profile+1.333*nii_profile)/row['norm']
    plt_prefix = 'plots/{:03d}-{}-calib'.format(row.index, full_id)
    make_three_plots(spec_profile, calib_profile, plt_prefix)
# Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1 ends here

# [[file:alba-orion-west.org::*Test%20what%20is%20going%20on][Test\ what\ is\ going\ on:1]]
# print(wphot.wcs)
# for row in tab:
#     print([row[x] for x in ('Dataset', 'imid', 'specid', 'Notes')])
# Test\ what\ is\ going\ on:1 ends here
