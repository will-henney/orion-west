# Re-use stuff from slit-calibration.py

# [[file:alba-orion-west.org::*Re-use%20stuff%20from%20slit-calibration.py][Re-use\ stuff\ from\ slit-calibration\.py:1]]
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
from astropy.convolution import convolve_fft, Gaussian1DKernel
converters = {'imid': [astropy.io.ascii.convert_numpy(np.str)]}
tab = Table.read('all-slits-input.tab',
                 format='ascii.tab', converters=converters)
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
# Re-use\ stuff\ from\ slit-calibration\.py:1 ends here

# Convert wavelength to pixel

# [[file:alba-orion-west.org::*Convert%20wavelength%20to%20pixel][Convert\ wavelength\ to\ pixel:1]]
def wav2pix(wav, wcs, nwav, isT):
    if isT:
        _, (xpix,) = wcs.all_world2pix([0], [wav], 0)
    else:
        (xpix,), _ = wcs.all_world2pix([wav], [0], 0)
    print(wcs.wcs.crpix, wcs.wcs.crval, wcs.wcs.get_cdelt(), wcs.wcs.get_pc())
    print('Wav:', wav, 'Pixel:', xpix)
    return max(0, min(nwav, int(xpix+0.5)))
# Convert\ wavelength\ to\ pixel:1 ends here

# Make a sensible WCS (even if wavelength info missing)
# 0.0994382022472


# [[file:alba-orion-west.org::*Make%20a%20sensible%20WCS%20(even%20if%20wavelength%20info%20missing)][Make\ a\ sensible\ WCS\ \(even\ if\ wavelength\ info\ missing\):1]]
wcs_extra = {
    '2007-01': (440, 6583.45, 0.1),
    '2006-02': (446, 6583.45, 0.1),
    ('2006-02', '323'): (442, 6583.45, 0.1),
    ('2006-02', '318'): (442, 6583.45, 0.1),
    ('2006-02', '260'): (491, 6583.45, 0.1),
    '2007-01b': (440, 6583.45, 0.1),
    '2010-01': (440, 6583.45, 0.1),
    '2013-02': (167, 6583.45, -0.056),
    ('2015-02', '0003'): (1015, 6583.45, 0.05775),
    ('2015-02', '0012'): (888, 6583.45, 0.05775),
}
def makeWCS(hdr, dset, imid, jwav):
    w = WCS(hdr)
    dwav = w.wcs.get_cdelt()[jwav]*w.wcs.get_pc()[jwav, jwav]
    if dwav == 1.0:
        # No WCS info from header, so fix it by hand
        extras =  wcs_extra.get((dset, imid)) or wcs_extra.get(dset)
        w.wcs.crpix[jwav], w.wcs.crval[jwav], w.wcs.cdelt[jwav] = wcs_extra[dset] 
    return w
# Make\ a\ sensible\ WCS\ \(even\ if\ wavelength\ info\ missing\):1 ends here

# TODO Extract profile along slit for an isolated line

# [[file:alba-orion-west.org::*Extract%20profile%20along%20slit%20for%20an%20isolated%20line][Extract\ profile\ along\ slit\ for\ an\ isolated\ line:1]]
def extract_profile(hdu, wavrest, dset, imid,
                    dw=4.0, dwbg_in=6.0, dwbg_out=8.0,
                    isT=False, smooth=20):
    jwav = 1 if isT else 0
    w = makeWCS(hdu.header, dset, imid, jwav)
    # Make sure array axis order is (position, wavelength)
    data = hdu.data.T if isT else hdu.data
    nslit, nwav = data.shape
    dwav = w.wcs.get_cdelt()[jwav]*w.wcs.get_pc()[jwav, jwav]
    print(wavrest, dwav, nslit, nwav)
    # pixel limits for line extraction
    i1 = wav2pix(wavrest-dw/2, w, nwav, isT)
    i2 = wav2pix(wavrest+dw/2, w, nwav, isT)
    # pixel limits for blue bg extraction
    iblu1 = wav2pix(wavrest-dwbg_out/2, w, nwav, isT)
    iblu2 = wav2pix(wavrest-dwbg_in/2, w, nwav, isT)
    # pixel limits for red bg extraction
    ired1 = wav2pix(wavrest+dwbg_in/2, w, nwav, isT)
    ired2 = wav2pix(wavrest+dwbg_out/2, w, nwav, isT)
    print(iblu1, iblu2, i1, i2, ired1, ired2)
    # extract backgrounds on blue and red sides
    bgblu = data[:, iblu1:iblu2].mean(axis=1)
    bgred = data[:, ired1:ired2].mean(axis=1)
    # take weighted average, accounting for cases where the bg region
    # does not fit in the image
    weight_blu = data[:, iblu1:iblu2].size
    weight_red = data[:, ired1:ired2].size
    bg = (bgblu*weight_blu + bgred*weight_red)/(weight_blu + weight_red)
    data -= bg[:, None]

    profile = data[:, i1:i2].sum(axis=1)
    if smooth is not None:
        profile = convolve_fft(profile, Gaussian1DKernel(stddev=smooth))
    return profile
# Extract\ profile\ along\ slit\ for\ an\ isolated\ line:1 ends here

# Loop over all the slits and check the ratios

# [[file:alba-orion-west.org::*Loop%20over%20all%20the%20slits%20and%20check%20the%20ratios][Loop\ over\ all\ the\ slits\ and\ check\ the\ ratios:1]]
datasets = set(tab['Dataset'])
sns.set_palette('bright', 17)
ratio_types = 'nii-ha', 'nii-ha-full', 'nii-nii-full'
fig_ax_dict = {(ds, rtype): plt.subplots(1, 1)
               for ds in datasets for rtype in ratio_types}
for row in tab:
    print(row['Dataset'], row['imid'], row['specid'])
    specfile = find_fits_filepath(row, 'fullspec')
    hafile = find_fits_filepath(row, 'ha')
    niifile = find_fits_filepath(row, 'nii')
    spechdu = fits.open(specfile)[0]
    hahdu = fits.open(hafile)[0]
    niihdu = fits.open(niifile)[0]

    isT = row['saxis'] == 1
    dset = row['Dataset']
    imid = row['imid']

    # First use the extracted ha and nii spectra
    ha = extract_profile(hahdu, 6562.79, dset, imid, isT=isT)
    nii = extract_profile(niihdu, 6583.45, dset, imid, isT=isT)
    fig, ax = fig_ax_dict[(dset, 'nii-ha')]
    ax.plot(nii/ha, alpha=0.8, label=str(row['imid']))

    # Then use the full spectrum
    ha = extract_profile(spechdu, 6562.79, dset, imid, isT=isT)
    nii = extract_profile(spechdu, 6583.45, dset, imid, isT=isT)
    niib = extract_profile(spechdu, 6548.05, dset, imid, isT=isT)
    fig, ax = fig_ax_dict[(dset, 'nii-ha-full')]
    ax.plot(nii/ha, alpha=0.8, label=str(row['imid']))
    fig, ax = fig_ax_dict[(dset, 'nii-nii-full')]
    ax.plot(niib/nii, alpha=0.8, label=str(row['imid']))


for (ds, rtype), (fig, ax) in fig_ax_dict.items():
    ax.legend()
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel('Pixel')
    if 'ha' in rtype:
        ax.set_ylabel('6583 / 6563')
    else:
        ax.set_ylabel('6548 / 6583')
    fig.savefig('plots/{}-check-{}.png'.format(rtype, ds), dpi=300)
# Loop\ over\ all\ the\ slits\ and\ check\ the\ ratios:1 ends here
