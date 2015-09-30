# [[file:alba-orion-west.org::*Imports][slit-calib-imports]]
import os
import sys
import numpy as np
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from matplotlib import pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
# slit-calib-imports ends here

# [[file:alba-orion-west.org::*Read%20in%20the%20table%20of%20all%20slits][read-slit-table]]
converters = {'imid': [astropy.io.ascii.convert_numpy(np.str)]}
tab = Table.read('all-slits-input.tab',
                 format='ascii.tab', converters=converters)
# read-slit-table ends here

# [[file:alba-orion-west.org::*Fits%20files%20for%20the%20spectra%20and%20image+slit][slit-calib-filenames]]
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
    path = template.format(id_)
    print('~/'+path)
    homedir = os.path.expanduser('~')
    return os.path.join(homedir, path)
# slit-calib-filenames ends here

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

# [[file:alba-orion-west.org::*Package%20up%20the%20slit%20coordinates%20for%20use%20in%20a%20FITS%20header][Package\ up\ the\ slit\ coordinates\ for\ use\ in\ a\ FITS\ header:1]]
def make_slit_wcs(db, slit_coords, spechdu):
    # Input WCS from original spectrum
    wspec = WCS(spechdu.header)
    wspec.fix()

    #
    # First find wavelength scale from the spectrum  
    #

    # For original spectrum, the wavelength and slit axes are 0-based,
    # but in FITS axis order instead of python access order, since
    # that is the way that that the WCS object likes to do it
    ospec_wavaxis = 2 - db['saxis']
    ospec_slitaxis = db['saxis'] - 1

    # The rules are that CDi_j is used if it is present, and only if
    # it is absent should CDELTi be used
    if wspec.wcs.has_cd():
        dwav = wspec.wcs.cd[ospec_wavaxis, ospec_wavaxis]
        # Check that the off-diagonal terms are zero
        assert(wspec.wcs.cd[0, 1] == wspec.wcs.cd[1, 0] == 0.0)
    else:
        dwav = wspec.wcs.cdelt[ospec_wavaxis]
        if wspec.wcs.has_pc():
            # If PCi_j is also present, make sure it is identity matrix
            assert(wspec.wcs.pc == np.eye(2))
    wav0 = wspec.wcs.crval[ospec_wavaxis]
    wavpix0 = wspec.wcs.crpix[ospec_wavaxis]

    #
    # Second, find the displacement scale and ref point from the slit_coords
    #
    # The slit_coords should already be in ICRS frame
    c = SkyCoord(slit_coords['RA'], slit_coords['Dec'], unit=u.deg)
    # Find vector of separations between adjacent pixels
    seps = c[:-1].separation(c[1:])
    # Ditto for the position angles
    PAs = c[:-1].position_angle(c[1:])
    # Check that they are all the same as the first one
    assert(np.allclose(seps/seps[0], 1.0))
    # assert(np.allclose(PAs/PAs[0], 1.0, rtol=1.e-4))
    # Then use the first one as the slit pixel size and PA
    ds, PA, PA_deg = seps[0].deg, PAs.mean().rad, PAs.mean().deg
    # And for the reference values too
    RA0, Dec0 = c[0].ra.deg, c[0].dec.deg

    #
    # Now make a new shiny output WCS, constructed from scratch
    #
    w = WCS(naxis=3)

    # Make use of all the values that we calculated above
    w.wcs.crpix = [wavpix0, 1, 1]
    w.wcs.cdelt = [dwav, ds, ds]
    w.wcs.crval = [wav0, RA0, Dec0]
    # PC order is i_j = [[1_1, 1_2, 1_3], [2_1, 2_2, 2_3], [3_1, 3_2, 3_3]]
    w.wcs.pc = [[1.0, 0.0, 0.0],
                [0.0, np.sin(PA), -np.cos(PA)],
                [0.0, np.cos(PA), np.sin(PA)]]

    #
    # Finally add in auxillary info
    #
    w.wcs.radesys = 'ICRS'
    w.wcs.ctype = ['AWAV', 'RA---TAN', 'DEC--TAN']
    w.wcs.specsys = 'TOPOCENT'
    w.wcs.cunit = [u.Angstrom, u.deg, u.deg]
    w.wcs.name = 'TopoWav'
    w.wcs.cname = ['Observed air wavelength', 'Right Ascension', 'Declination']
    w.wcs.mjdobs = wspec.wcs.mjdobs
    w.wcs.datfix()              # Sets DATE-OBS from MJD-OBS

    # Check the new pixel values
    npix = len(slit_coords['RA'])
    check_coords = pixel_to_skycoord(np.arange(npix), [0]*npix, w, 0)
    # These should be the same as the ICRS coords in slit_coords
    print('New coords:', check_coords[::100])
    print('Displacements in arcsec:', check_coords.separation(c).arcsec[::100])
    # 15 Sep 2015: They seem to be equal to within about 1e-2 arcsec

    return w
# Package\ up\ the\ slit\ coordinates\ for\ use\ in\ a\ FITS\ header:1 ends here

# [[file:alba-orion-west.org::*Package%20up%20the%20slit%20coordinates%20for%20use%20in%20a%20FITS%20header][Package\ up\ the\ slit\ coordinates\ for\ use\ in\ a\ FITS\ header:1]]
def fixup4ds9(w):
    w.wcs.ctype  = ['LINEAR', 'LINEAR', 'LINEAR']
    # w.wcs.cdelt[1:] *= 3600
    # w.wcs.units[1:] = 'arcsec', 'arcsec'
    w.wcs.crval[1], w.wcs.crval[2] = 0.0, 0.0
    w.wcs.name = 'TopoWavDS9'
    return w
# Package\ up\ the\ slit\ coordinates\ for\ use\ in\ a\ FITS\ header:1 ends here

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
def make_three_plots(spec, calib, prefix, niirat=None):
    assert spec.shape == calib.shape
    fig, axes = plt.subplots(3, 1)

    vmin, vmax = 0.0, np.median(calib) + 5*calib.std()

    ypix = np.arange(len(calib))
    ratio = spec/calib
    mask = (ypix > 10.0) & (ypix < ypix.max() - 10.0) \
           & (ratio > np.median(ratio) - 2*ratio.std()) \
           & (ratio < np.median(ratio) + 2*ratio.std()) 
    try:
        ratio_fit = fit_cheb(ypix, ratio, mask=mask)
    except:
        ratio_fit = np.ones_like(ypix)
      
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
    if niirat is not None:
        axes[2].plot(ypix, niirat, 'b')
    axes[2].set_xlim(0.0, ypix.max())
    axes[2].set_ylim(0.0, 1.5)
    axes[2].set_xlabel('Slit pixel')
    axes[2].set_ylabel('Ratio: Spec / Calib')


    fig.set_size_inches(5, 8)
    fig.tight_layout()
    fig.savefig(prefix+'.png', dpi=300)

    return ratio_fit
# Make\ some\ useful\ and\ pretty\ plots:1 ends here

# [[file:alba-orion-west.org::*Use%20command%20line%20argument%20to%20restrict%20which%20datasets%20are%20processed][Use\ command\ line\ argument\ to\ restrict\ which\ datasets\ are\ processed:1]]
if len(sys.argv) > 1:
    selector_pattern = sys.argv[1]
else:
    selector_pattern = ''
# Use\ command\ line\ argument\ to\ restrict\ which\ datasets\ are\ processed:1 ends here

# [[file:alba-orion-west.org::*Remove%20background%20and%20sum%20over%20wavelength%20across%20line][Remove\ background\ and\ sum\ over\ wavelength\ across\ line:1]]
def extract_profile(data, wcs, wavrest, dw=7.0):
    data = remove_bg_and_regularize(data, wcs, wavrest)
    # pixel limits for line extraction
    lineslice = wavs2slice([wavrest-dw/2, wavrest+dw/2], wcs)
    return data[:, lineslice].sum(axis=1)
# Remove\ background\ and\ sum\ over\ wavelength\ across\ line:1 ends here

# [[file:alba-orion-west.org::*Remove%20background%20and%20sum%20over%20wavelength%20across%20line][Remove\ background\ and\ sum\ over\ wavelength\ across\ line:1]]
def wavs2slice(wavs, wcs):
    """Convert a wavelength interval `wavs` (length-2 sequence) to a slice of the relevant axis`"""
    assert len(wavs) == 2
    isT = row['saxis'] == 1
    if isT:
        _, xpixels = wcs.all_world2pix([0, 0], wavs, 0)
    else:
        xpixels, _ = wcs.all_world2pix(wavs, [0, 0], 0)
    print('Wav:', wavs, 'Pixel:', xpixels)
    i1, i2 = np.maximum(0, (xpixels+0.5).astype(int))
    return slice(min(i1, i2), max(i1, i2))

def remove_bg_and_regularize(data, wcs, wavrest, dwbg_in=7.0, dwbg_out=10.0):
    '''
    Transpose data if necessary, and then subtract off the background (blue and red of line)
    '''
    isT = row['saxis'] == 1
    # Make sure array axis order is (position, wavelength)
    if isT:
        data = data.T
    if row['Dataset'] == '2015-02':
        # Don't try this for the newest data, I already removed the BG
        return data
    # pixel limits for blue, red bg extraction
    bslice = wavs2slice([wavrest-dwbg_out/2, wavrest-dwbg_in/2], wcs)
    rslice = wavs2slice([wavrest+dwbg_in/2, wavrest+dwbg_out/2], wcs)
    # extract backgrounds on blue and red sides
    bgblu = data[:, bslice].mean(axis=1)
    bgred = data[:, rslice].mean(axis=1)
    # take weighted average, accounting for cases where the bg region
    # does not fit in the image
    weight_blu = data[:, bslice].size
    weight_red = data[:, rslice].size
    print('Background weights:', weight_blu, weight_red)
    bg = (bgblu*weight_blu + bgred*weight_red)/(weight_blu + weight_red)
    return data - bg[:, None]
# Remove\ background\ and\ sum\ over\ wavelength\ across\ line:1 ends here

# [[nil][Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1]]
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
    ha_profile = extract_profile(hahdu.data, WCS(hahdu.header), 6562.79)
    # Take the nii/ha calibration correction factor  from the table
    nii_profile = row['r(nii)']*extract_profile(niihdu.data, WCS(niihdu.header), 6583.45)
    spec_profile = (ha_profile+1.333*nii_profile)/row['norm']
    plt_prefix = 'plots/{:03d}-{}-calib'.format(row.index, full_id)
    ratio = make_three_plots(spec_profile, calib_profile, plt_prefix, niirat=nii_profile/ha_profile)

    #
    # Save calibrated spectra to files
    #

    for hdu, lineid, restwav  in [[hahdu, 'ha', 6562.79],
                                  [niihdu, 'nii', 6583.45]]:
        print('Saving', lineid, 'calibrated spectrum')
        # Apply basic calibration zero-point and scale
        hdu.data = remove_bg_and_regularize(hdu.data, WCS(hdu.header), restwav)/row['norm']
        # Regularize spectral data so that wavelength is x and pos is y
        # This is now done by the bg subtraction function

        # Apply polynomial correction along slit
        hdu.data /= ratio[:, None]
        # Extend in the third dimension (degenerate axis perp to slit)
        hdu.data = hdu.data[None, :, :]

        # Create the WCS object for the calibrated slit spectra
        wslit = make_slit_wcs(row, slit_coords, hdu)
        # Set the rest wavelength for this line
        wslit.wcs.restwav = (restwav*u.Angstrom).to(u.m).value
        # # Remove WCS keywords that might cause problems
        # for i in 1, 2:
        #     for j in 1, 2:
        #         kwd = 'CD{}_{}'.format(i, j)
        #         if kwd in hdu.header:
        #             hdu.header.remove(kwd) 
        # Then update the header with the new WCS structure as the 'A'
        # alternate transform
        hdu.header.update(wslit.to_header(key='A'))
        # Also save the normalization factor as a per-slit weight to use later
        hdu.header['WEIGHT'] = row['norm']

        # And better not to change the original WCS at all
        # Unless we have transposed the array, which we have to compensate for
        if row['saxis'] == 1:
            for k in ['CRPIX{}', 'CRVAL{}', 'CDELT{}', 'CD{0}_{0}']:
                hdu.header[k.format('1')], hdu.header[k.format('2')] = hdu.header[k.format('2')], hdu.header[k.format('1')] 
        # # And write a bowdlerized version that DS9 can understand as the main WCS
        # hdu.header.update(fixup4ds9(wslit).to_header(key=' '))
        calibfile = 'Calibrated/{}-{}.fits'.format(full_id, lineid)
        hdu.writeto(calibfile, clobber=True)
# Loop\ over\ the\ slit\ positions\ and\ do\ the\ stuff:1 ends here

# [[file:alba-orion-west.org::*Test%20what%20is%20going%20on][Test\ what\ is\ going\ on:1]]
# print(wphot.wcs)
# for row in tab:
#     print([row[x] for x in ('Dataset', 'imid', 'specid', 'Notes')])
# Test\ what\ is\ going\ on:1 ends here
