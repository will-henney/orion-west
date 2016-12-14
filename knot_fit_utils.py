import os
import numpy as np
from scipy.ndimage import generic_filter

from saba import SherpaFitter
from astropy.modeling.models import Gaussian1D, Lorentz1D, Const1D
from astropy.io import fits
from astropy.wcs import WCS
import pyregion

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', 
        style='whitegrid', 
        palette='dark',
        font_scale=1.5,
        color_codes=True)

def get_knots_from_region_file(fn):
    """Return dict of all knots in region file `fn`

    Dict is keyed by region name, with values: j1, j2, u0
    """
    knot_regions = pyregion.open(fn)
    knots = {}
    for r in knot_regions:
        if r.name == 'box' and r.coord_format == 'image':
            k = r.attr[1]['text']
            x0, y0, dx, dy, theta = r.coord_list
            j1 = int(y0 - dy)
            j2 = int(y0 + dy)
            u0 = vel_from_region_text(k)
            knots[k] = j1, j2, u0
    return knots

def vel_from_region_text(text):
    '''Try to parse something like "4299-524 (-70)" to find velocity'''
    # Try and get something like "(-70)"
    maybe_parens = text.split()[-1]
    if maybe_parens.startswith('(') and maybe_parens.endswith(')'):
        vstring = maybe_parens[1:-1]
        try:
            v0 = float(vstring)
        except ValueError:
            v0 = None
    else:
        v0 = None
    return v0

# No upper bound on constant term by default
CORE_CMAX = None

def _init_bgmodel(lorentz_mean=15.0):
    """Initialize model for background: constant plus Lorentzian"""
    lorentz_fixed = {'x_0': True, 'fwhm': True}
    lorentz_bounds = {'amplitude': [0, None]}
    constant_bounds = {'amplitude': [0, CORE_CMAX]}
    bgmodel = (Lorentz1D(0.1, lorentz_mean, 100.0, name='Lorentz',
                         bounds=lorentz_bounds, fixed=lorentz_fixed)
	       + Const1D(1.0e-4, bounds=constant_bounds, name='Constant'))
    return bgmodel


# Don't allow core components to intrude into knot velocity space
CORE_VMIN = -10.0
# Should not be narrower than instrumental profile
CORE_WMIN = 3.0
# And not too wide or they compete with Lorentzian
CORE_WMAX = 25.0

def _init_coremodel():
    """Initialize model for core of profile: sum of 5 Gaussians"""
    bounds = {'amplitude': [0, None],
	      'stddev': [CORE_WMIN, CORE_WMAX],
	      'mean': [CORE_VMIN, None]}
    coremodel = (Gaussian1D(1.0, 5.0, 5.0, bounds=bounds, name='G1') 
                 + Gaussian1D(5.0, 10.0, 5.0, bounds=bounds, name='G2')
                 + Gaussian1D(5.0, 15.0, 5.0, bounds=bounds, name='G3')
                 + Gaussian1D(5.0, 20.0, 5.0, bounds=bounds, name='G4')
                 + Gaussian1D(1.0, 40.0, 5.0, bounds=bounds, name='G5')
    )
    return coremodel


KNOT_VMIN = -120.0
KNOT_VMAX = 0.0
KNOT_WMIN = 3.0
KNOT_WMAX = 30.0

def _init_knotmodel(amp_init=0.01, v_init=-60.0):
    """Initialize model for knot: a single Gaussian"""
    bounds = {'amplitude': [0, None],
	      'stddev': [KNOT_WMIN, KNOT_WMAX],
	      'mean': [KNOT_VMIN, KNOT_VMAX]}
    knotmodel = Gaussian1D(amp_init, v_init, 5.0, bounds=bounds) 
    return knotmodel


# Knot is fitted in region +/- KNOT_WIDTH around the nominal velocity
# The same region is omitted from the core fit
KNOT_WIDTH = 30.0

# Highest value of reduced chi2 that will still allow estimating
# confidence bounds on the fit parameters.  We increase this from the
# default value of 3 since we sometimes have fits that are worse than
# that :(
MAX_RSTAT = 30.0

# Scale for sqrt(N) contribution to the error budget.  Strictly, we
# should go back to the data in electron counts before calibration and
# continuum removal in order to calculate this.  But that is too much
# work, so we just treat it as a free parameter.  Overestimating it is
# harmless.
POISSON_SCALE = 0.02

def fit_knot(hdu, j1, j2, u0):

    NS, NV = hdu.data.shape
    w = WCS(hdu.header)
    vels, _ = w.all_pix2world(np.arange(NV), [0]*NV, 0)
    vels /= 1000.0

    # Ensure we don't go out of bounds
    j1 = max(j1, 0)
    j2 = min(j2, NS)
    print('Slit pixels {}:{} out of {}'.format(j1, j2, NS))

    knotspec = hdu.data[j1:j2, :].sum(axis=0)
    # make sure all pixels are positive, since that helps the fitting/plotting
    knotspec -= knotspec.min()

    # Levenberg-Marquardt for easy jobs
    lmfitter = SherpaFitter(statistic='chi2',
                            optimizer='levmar',
                            estmethod='confidence')
    # Simulated annealing for trickier jobs
    safitter = SherpaFitter(statistic='chi2',
                            optimizer='neldermead',
                            estmethod='covariance')

    # First do the strategy for typical knots (u0 = [-30, -80])

    # Estimate error from the BG: < -120 or > +100
    bgmask = np.abs(vels + 10.0) >= 110.0
    bgerr = np.std(knotspec[bgmask]) * np.ones_like(vels)

    # Fit to the BG with constant plus Lorentz
    try: 
        vmean = np.average(vels, weights=knotspec)
    except ZeroDivisionError:
        vmean = 15.0

    bgmodel = lmfitter(_init_bgmodel(vmean),
		       vels[bgmask], knotspec[bgmask],
		       err=bgerr[bgmask])
    # Now freeze the BG model and add it to the initial core model
    bgmodel['Lorentz'].fixed['amplitude'] = True
    bgmodel['Constant'].fixed['amplitude'] = True

    # Increase the data err in the bright part of the line to mimic Poisson noise
    # Even though we don't know what the normalization is really, we will guess ...
    spec_err = bgerr + POISSON_SCALE*np.sqrt(knotspec)

    # Fit to the line core
    knotmask = np.abs(vels - u0) <= KNOT_WIDTH
    coremodel = safitter(_init_coremodel() + bgmodel,
                         vels[~knotmask], knotspec[~knotmask],
                         err=spec_err[~knotmask])
    core_fit_info = safitter.fit_info

    # Residual should contain just knot
    residspec = knotspec - coremodel(vels)

    # Calculate running std of residual spectrum
    NWIN = 11
    running_mean = generic_filter(residspec, np.mean, size=(NWIN,))
    running_std = generic_filter(residspec, np.std, size=(NWIN,))

    # Increase error estimate for data points where this is larger
    # than spec_err, but only for velocities that are not in knotmask
    residerr = bgerr
    # residerr = spec_err
    mask = (~knotmask) & (running_std > bgerr)
    residerr[mask] = running_std[mask]
    # The reason for this is so that poor modelling of the core is
    # accounted for in the errors.  Otherwise the reduced chi2 of the
    # knot model will be too high

    # Make an extended mask for fitting the knot, omitting the
    # redshifted half of the spectrum since it is irrelevant and we
    # don't want it to affect tha chi2 or the confidance intervals
    bmask = vels < 50.0

    # Fit single Gaussian to knot 
    amplitude_init = residspec[knotmask].max()
    if amplitude_init < 0.0:
        # ... pure desperation here
        amplitude_init = residspec[bmask].max()
    knotmodel = lmfitter(_init_knotmodel(amplitude_init, u0),
                         vels[bmask], residspec[bmask],
                         err=residerr[bmask])

    # Calculate the final residuals, which should be flat
    final_residual = residspec - knotmodel(vels)

    # Look at stddev of the final residuals and use them to rescale
    # the residual errors.  Then re-fit the knot with this better
    # estimate of the errors.  But only if rescaling would reduce the
    # data error estimate.
    residerr_rescale = final_residual[bmask].std() / residerr[bmask].mean()
    if residerr_rescale < 1.0:
        print('Rescaling data errors by', residerr_rescale)
        residerr *= residerr_rescale
        knotmodel = lmfitter(knotmodel,
                             vels[bmask], residspec[bmask],
                             err=residerr[bmask])
    else:
        residerr_rescale = 1.0

    knot_fit_info = lmfitter.fit_info
    lmfitter._fitter.estmethod.config['max_rstat'] = MAX_RSTAT
    if knot_fit_info.rstat < MAX_RSTAT:
        knot_fit_errors = lmfitter.est_errors(sigma=3)
    else:
        knot_fit_errors = None

    return {
        'nominal knot velocity': u0,
        'velocities': vels,
        'full profile': knotspec,
        'error profile': residerr,
        'core fit model': coremodel,
        'core fit profile': coremodel(vels),
        'core fit components': {k: coremodel[k](vels) for k in coremodel.submodel_names},
        'core fit info': core_fit_info,
        'core-subtracted profile': residspec,
        'knot fit model': knotmodel,
        'knot fit profile': knotmodel(vels),
        'knot fit info': knot_fit_info,
        'knot fit errors': knot_fit_errors,
        'error rescale factor': residerr_rescale,
    }

def fit_knot_unified(hdu, j1, j2, u0, lineid='nii'):

    NS, NV = hdu.data.shape
    w = WCS(hdu.header)
    vels, _ = w.all_pix2world(np.arange(NV), [0]*NV, 0)
    vels /= 1000.0

    # Ensure we don't go out of bounds
    j1 = max(j1, 0)
    j2 = min(j2, NS)
    print('Slit pixels {}:{} out of {}'.format(j1, j2, NS))

    knotspec = hdu.data[j1:j2, :].sum(axis=0)
    # make sure all pixels are positive, since that helps the fitting/plotting
    knotspec -= knotspec.min()

    # Levenberg-Marquardt for easy jobs
    lmfitter = SherpaFitter(statistic='chi2',
                            optimizer='levmar',
                            estmethod='confidence')

    # Simulated annealing for trickier jobs
    safitter = SherpaFitter(statistic='chi2',
                            optimizer='neldermead',
                            estmethod='covariance')

    # The idea is that this strategy should work for all knots

    # Estimate error from the BG: < -120 or > +100
    bgmask = np.abs(vels + 10.0) >= 110.0
    bgerr = np.std(knotspec[bgmask]) * np.ones_like(vels)

    # Define core as [-10, 50], or 20 +/- 30
    coremask = np.abs(vels - 20.0) < 30.0

    # Fit to the BG with constant plus Lorentz
    try: 
        vmean = np.average(vels[coremask], weights=knotspec[coremask])
    except ZeroDivisionError:
        vmean = 15.0

    bgmodel = lmfitter(_init_bgmodel(vmean),
		       vels[bgmask], knotspec[bgmask],
		       err=bgerr[bgmask])
    # Now freeze the BG model and add it to the initial core model
    #bgmodel['Lorentz'].fixed['amplitude'] = True
    #bgmodel['Constant'].fixed['amplitude'] = True

    # Increase the data err in the bright part of the line to mimic Poisson noise
    # Even though we don't know what the normalization is really, we will guess ...
    spec_err = bgerr + POISSON_SCALE*np.sqrt(knotspec)


    ## Now for the exciting bit, fit everything at once
    ##
    knotmask = np.abs(vels - u0) <= KNOT_WIDTH
    # For low-velocity knots, we need to exclude positive velocities
    # from the mask, since they will have large residual errors from
    # the core subtraction
    knotmask = knotmask & (vels < 0.0)

    # Start off with the frozen BG model
    fullmodel = bgmodel.copy()
    core_components = list(fullmodel.submodel_names)

    # Add in a model for the core
    DV_INIT = [-15.0, -5.0, 5.0, 10.0, 30.0]
    NCORE = len(DV_INIT)
    BASE_WIDTH = 10.0 if lineid == 'ha' else 5.0
    W_INIT = [BASE_WIDTH]*4 + [1.5*BASE_WIDTH]
    for i in range(NCORE):
        v0 = vmean + DV_INIT[i]
        w0 = W_INIT[i]
        component = 'G{}'.format(i)
        fullmodel += Gaussian1D(
            3.0, v0, w0,
            bounds={'amplitude': [0, None],
                    'mean': [v0 - 10, v0 + 10],
                    'stddev': [w0, 1.5*w0]},
            name=component)
        core_components.append(component)

    # Now, add in components for the knot to extract
    knotmodel_init = Gaussian1D(
        0.01, u0, BASE_WIDTH,
        # Allow +/- 10 km/s leeway around nominal knot velocity
        bounds={'amplitude': [0, None],
                'mean': [u0 - 10, u0 + 10],
                'stddev': [BASE_WIDTH, 25.0]},
        name='Knot')
    fullmodel += knotmodel_init
    knot_components = ['Knot']
    other_components = []

    # Depending on the knot velocity, we may need other components to
    # take up the slack too
    if u0 <= -75.0 or u0 >= -50.0:
        # Add in a generic fast knot
        fullmodel += Gaussian1D(
            0.01, -60.0, BASE_WIDTH,
            bounds={'amplitude': [0, None],
                    'mean': [-70.0, -50.0],
                    'stddev': [BASE_WIDTH, 25.0]},
            name='Fast other')
        other_components.append('Fast other')

    if u0 <= -50.0:
        # Add in a generic slow knot
        fullmodel += Gaussian1D(
            0.01, -30.0, BASE_WIDTH,
            bounds={'amplitude': [0, None],
                    'mean': [-40.0, -10.0],
                    'stddev': [BASE_WIDTH, 25.0]},
            name='Slow other')
        other_components.append('Slow other')

    if u0 >= -75.0:
        # Add in a very fast component
        fullmodel += Gaussian1D(
            0.001, -90.0, BASE_WIDTH,
            bounds={'amplitude': [0, None],
                    'mean': [-110.0, -75.0],
                    'stddev': [BASE_WIDTH, 25.0]},
            name='Ultra-fast other')
        other_components.append('Ultra-fast other')

    if u0 <= 30.0:
        # Add in a red-shifted component just in case
        fullmodel += Gaussian1D(
            0.01, 40.0, BASE_WIDTH,
            bounds={'amplitude': [0, None],
                    'mean': [30.0, 200.0],
                    'stddev': [BASE_WIDTH, 25.0]},
            name='Red other')
        other_components.append('Red other')




    # Moment of truth: fit models to data
    fullmodel = safitter(fullmodel, vels, knotspec, err=spec_err)
    full_fit_info = safitter.fit_info

    # Isolate the core+other model components 
    coremodel = fullmodel[core_components[0]]
    for component in core_components[1:] + other_components:
        coremodel += fullmodel[component]

    # Subtract the core model from the data
    residspec = knotspec - coremodel(vels)

    # Now re-fit the knot model to the residual

    # Calculate running std of residual spectrum
    NWIN = 11
    running_mean = generic_filter(residspec, np.mean, size=(NWIN,))
    running_std = generic_filter(residspec, np.std, size=(NWIN,))

    # Increase error estimate for data points where this is larger
    # than spec_err, but only for velocities that are not in knotmask
    residerr = bgerr
    # residerr = spec_err
    mask = (~knotmask) & (running_std > bgerr)
    residerr[mask] = running_std[mask]
    # The reason for this is so that poor modelling of the core is
    # accounted for in the errors.  Otherwise the reduced chi2 of the
    # knot model will be too high

    # Make an extended mask for fitting the knot, omitting the
    # redshifted half of the spectrum since it is irrelevant and we
    # don't want it to affect tha chi2 or the confidance intervals
    bmask = vels < 50.0

    knotmodel = lmfitter(knotmodel_init,
                         vels[bmask], residspec[bmask],
                         err=residerr[bmask])

    # Calculate the final residuals, which should be flat
    final_residual = residspec - knotmodel(vels)

    # Look at stddev of the final residuals and use them to rescale
    # the residual errors.  Then re-fit the knot with this better
    # estimate of the errors.  But only if rescaling would reduce the
    # data error estimate.
    residerr_rescale = final_residual[bmask].std() / residerr[bmask].mean()
    if residerr_rescale < 1.0:
        print('Rescaling data errors by', residerr_rescale)
        residerr *= residerr_rescale
        knotmodel = lmfitter(knotmodel,
                             vels[bmask], residspec[bmask],
                             err=residerr[bmask])
    else:
        residerr_rescale = 1.0

    knot_fit_info = lmfitter.fit_info
    lmfitter._fitter.estmethod.config['max_rstat'] = MAX_RSTAT
    if knot_fit_info.rstat < MAX_RSTAT:
        knot_fit_errors = lmfitter.est_errors(sigma=3)
    else:
        knot_fit_errors = None

    return {
        'nominal knot velocity': u0,
        'velocities': vels,
        'full profile': knotspec,
        'error profile': residerr,
        'core fit model': coremodel,
        'core fit profile': coremodel(vels),
        'core fit components': {k: coremodel[k](vels) for k in coremodel.submodel_names},
        'core fit info': full_fit_info,
        'core-subtracted profile': residspec,
        'knot fit model': knotmodel,
        'knot fit profile': knotmodel(vels),
        'knot fit info': knot_fit_info,
        'knot fit errors': knot_fit_errors,
        'error rescale factor': residerr_rescale,
        'knot j range': (j1, j2),
    }

def find_fwhm(f, v, frac=0.5):
    """Find literal FWHM of discretely sampled profile f(v) by linear interpolation

    STILL NOT FULLY TESTED

    Based on the Fortran implementation in 
    /Users/will/Work/BobKPNO/src/newlinemod.f90
    """
    ipeak = np.argmax(f)
    fpeak = f[ipeak]
    m = f >= frac*fpeak
    ileft = v.tolist().index(v[m][0])
    iright = v.tolist().index(v[m][-1])
    if ileft <= 0:
        uleft = v[0]
    elif ileft >= len(f):
        uleft = v[-1]
    else:
        uleft = (
            v[ileft] -
            (v[ileft] - v[ileft-1]) * (f[ileft] - frac*fpeak)
            / (f[ileft] - f[ileft-1])
        )
    if iright < 0:
        uright = v[0]
    elif iright >= len(f):
        uright = v[-1]
    else:
        uright = (
            v[iright] +
            (v[iright+1] - v[iright]) * (f[iright] - frac*fpeak)
            / (f[iright] - f[iright-1])
        )
    return uright - uleft


def get_statistics(f, v):
    """Find mean, sigma, flux, fwhm
    """
    flux = np.trapz(f, v)
    try: 
        vbar = np.average(v, weights=f)
        sigma = np.sqrt(np.average(np.square(v - vbar), weights=f))
    except ZeroDivisionError:
        vbar = np.nan
        sigma = np.nan

    # fwhm = find_fwhm(f, v)
    fwhm = sigma * np.sqrt(8.0*np.log(2.0))
    return {'flux': flux, 'mean velocity': vbar, 'sigma': sigma, 'FWHM': fwhm}

import json
from astropy.utils.misc import JsonCustomEncoder

def save_fit_data(kn, save_dir, line_id, slit_id):
    """Save all the fit data for knot and core"""
    knot_id = os.path.basename(save_dir)
    jsonfile = os.path.join(save_dir,
		      '{}-{}-{}.json'.format(line_id, knot_id, slit_id))

    # Start with copy of input data dict
    data = kn.copy()          # should this be a depp copy?

    # Add basic info
    data['knot'] = knot_id
    data['slit'] = slit_id
    data['emission line'] = line_id

    # Add some more summary statistics
    data['core fit moments'] = get_statistics(
          data['core fit profile'], data['velocities'])

    data['knot fit moments'] = get_statistics(
          data['knot fit profile'], data['velocities'])

    data['full profile moments'] = get_statistics(
          data['full profile'], data['velocities'])

    # Take a slightly more generous knot window for calculating residual stats
    m = np.abs(data['velocities']
	       - data['nominal knot velocity']) <= 1.5*KNOT_WIDTH

    data['core-subtracted profile moments'] = get_statistics(
          data['core-subtracted profile'][m], data['velocities'][m])

    # Re-write the confidence levels as per the graphics program
    if data['knot fit errors'] is not None:
        p = {k: (_v, _p if _p else np.nan, _m if _m else np.nan)
             for k, _v, _p, _m in zip(*data['knot fit errors'])}
        p['FWHM'] = [np.sqrt(8.0*np.log(2.0))*_w for _w in p['stddev']]
        p['confidence level'] = '3-sigma'
    else:
        p = {'confidence level': 'MAX CHI-SQUARED EXCEEDED!'}

    p['reduced chi^2'] = data['knot fit info'].rstat
    data['knot fit parameters'] = p

    # Extract the core fit parameters from the best-fit model -
    # don't bother with error estimates
    m = data['core fit model']
    data['core fit parameters'] = {
          mn: dict(zip(m[mn].param_names, m[mn].parameters))
          for mn in m.submodel_names}
    data['core fit parameters']['reduced chi^2'] = data['core fit info'].rstat

    # Remove items that we don't want to save to JSON
    del data['core fit components']
    del data['core fit model']
    del data['core fit profile']
    del data['full profile']
    del data['core-subtracted profile']
    del data['error profile']
    del data['velocities']
    del data['knot fit model']
    del data['knot fit profile']
    del data['knot fit errors']               
    del data['knot fit info']
    del data['core fit info']

    with open(jsonfile, 'w') as f:
          json.dump(data, f, indent=4, cls=JsonCustomEncoder,
	      default=lambda x: repr(x).split('\n'))

SLIT_DIR = 'Calibrated/BGsub'
REGION_DIR = 'Will-Regions-2016-12'
REGION_PREFIX = 'pvboxes-knots'
KNOTS_DIR = 'Knot-Fits-Final'
STRATEGY = 'unified'

def process_slit(fn):
    print('-*^*- '*10)
    print('Processing', fn)

    if fn.startswith(SLIT_DIR):
        fits_path = fn
    else:
        fits_path = os.path.join(SLIT_DIR, fn)
    hdu, = fits.open(fits_path)

    # Rejig the slit name into a slit_id and a line_id
    slit_name, _ = os.path.splitext(os.path.basename(fits_path))
    # e.g., XX1620-2010-01-236-ha-vhel
    _pos, _y, _m, _n, line_id, _ = slit_name.split('-')
    slit_id = '-'.join([_pos, _y, _m, _n])

    region_path = os.path.join(REGION_DIR,
			       '{}-{}.reg'.format(REGION_PREFIX, slit_id))
    try: 
        knots = get_knots_from_region_file(region_path)
    except FileNotFoundError:
        print('No knots in this slit')
        return

    for name, data in knots.items():
        print('Processing knot', name, 'in slit', os.path.basename(fits_path))
        knot_id = name.split()[0]
        save_dir = os.path.join(KNOTS_DIR, knot_id)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if STRATEGY.lower() == 'unified':
            kn = fit_knot_unified(hdu, *data)
        else:
            kn = fit_knot(hdu, *data)

        save_fit_data(kn, save_dir, line_id, slit_id)
        plot_core_fit(kn, save_dir, line_id, slit_id)
        plot_knot_fit(kn, save_dir, line_id, slit_id)

import glob

PATTERNS = ['[XY][XY]*-ha-vhel.fits', '[XY][XY]*-nii-vhel.fits']
def process_all_slits(patterns=PATTERNS):
    slit_list = []
    for pattern in patterns:
        slit_list += glob.glob(os.path.join(SLIT_DIR, pattern))
    for slit in slit_list:
        process_slit(slit)

LINE_LABEL = {'ha': 'Ha 6563',  'nii': '[N II] 6583'}

def plot_core_fit(kn, save_dir, line_id, slit_id):
    fig, ax = plt.subplots(1, 1)
    ax.plot('velocities', 'full profile', '.', data=kn)
    fullfit = kn['core fit profile'] + kn['knot fit profile']
    ax.plot(kn['velocities'], fullfit)
    ax.errorbar('velocities', 'full profile', 
                'error profile', data=kn, fmt=None, alpha=0.4, errorevery=4)
    for k, v in kn['core fit components'].items():
        ax.plot(kn['velocities'], v, '--', alpha=0.6, lw='1', color='k')
    ax.plot(kn['velocities'], kn['knot fit profile'], '--',
            alpha=0.6, lw='2', color='r')

    ax.fill_betweenx([0.0, 100.0], 
                     [kn['nominal knot velocity'] - KNOT_WIDTH]*2,
                     [kn['nominal knot velocity'] + KNOT_WIDTH]*2, 
                     alpha=0.1)

    ax.set(xlim=[-150, 200],
           yscale='log', ylim=[0.001, None],
           xlabel='Heliocentric Velocity',
           ylabel='Line profile',
           title='{:s} - {:s} - {:s}'.format(os.path.basename(save_dir),
                                             slit_id, LINE_LABEL[line_id]),
    )
    fig.set_size_inches(8, 6)
    knot_id = os.path.basename(save_dir)
    plotfile = os.path.join(save_dir,
                            '{}-core-fit-{}-{}.png'.format(
                                line_id, knot_id, slit_id))
    fig.savefig(plotfile, dpi=200)
    # Important to close figure explicitly so as not to leak resources
    plt.close(fig)


def plot_knot_fit(kn, save_dir, line_id, slit_id):
    fig, ax = plt.subplots(1, 1)
    ax.plot('velocities', 'core-subtracted profile', '.', data=kn)
    ax.plot('velocities', 'knot fit profile', data=kn)
    ax.errorbar('velocities', 'core-subtracted profile', 
                'error profile', data=kn, fmt=None, alpha=0.4, errorevery=4)
    ax.axvline(kn['nominal knot velocity'], lw=0.5, ls='--')


    param_errors = kn['knot fit errors']
    if param_errors is not None:
        p = {k: (_v, _p if _p else np.nan, _m if _m else np.nan)
             for k, _v, _p, _m in zip(*param_errors)}

        knotmodel = kn['knot fit model']

        knot_min_a = knotmodel.copy()
        knot_min_a.amplitude.value += p['amplitude'][1]

        knot_max_a = knotmodel.copy()
        knot_max_a.amplitude.value += p['amplitude'][2]

        knot_min_v = knotmodel.copy()
        knot_min_v.mean.value += p['mean'][1]
        if not np.isfinite(knot_min_v.stddev.value):
            knot_min_v.stddev.value = KNOT_VMIN

        knot_max_v = knotmodel.copy()
        knot_max_v.mean.value += p['mean'][2]
        if not np.isfinite(knot_max_v.stddev.value):
            knot_max_v.stddev.value = KNOT_VMAX

        knot_min_w = knotmodel.copy()
        knot_min_w.stddev.value += p['stddev'][1]
        if not np.isfinite(knot_min_w.stddev.value):
            knot_min_w.stddev.value = KNOT_WMIN

        knot_max_w = knotmodel.copy()
        knot_max_w.stddev.value += p['stddev'][2]
        if not np.isfinite(knot_max_w.stddev.value):
            knot_max_w.stddev.value = 1.5*KNOT_WMAX

        vels = kn['velocities']
        alpha = 0.15
        ax.fill_between(vels, knot_min_a(vels), knot_max_a(vels),
                        color='k', alpha=alpha)
        ax.fill_between(vels, knot_min_v(vels), knot_max_v(vels),
                        color='k', alpha=alpha)
        ax.fill_between(vels, knot_min_w(vels), knot_max_w(vels),
                        color='k', alpha=alpha)


        ptext = 'Knot fit parameters' + '\n'
        ptext += '($3\sigma$-confidence interval)' + '\n'
        # Reduced chi2
        ptext += r'$\mathrm{Reduced\ }\chi^2 = '
        ptext += r'{:.2f}$'.format(kn['knot fit info'].rstat) + '\n'
        # Amplitude
        ptext += r'$\mathrm{Amplitude} = '
        ptext += '{:.3f}_{{{:+.3f}}}^{{{:+.3f}}}$'.format(*p['amplitude']) + '\n'
        # Mean
        ptext += r'$\mathrm{Mean\ velocity} = '
        ptext += '{:.1f}_{{{:+.1f}}}^{{{:+.1f}}}$'.format(*p['mean'])
        ptext += r'$\mathrm{\ km\ s^{-1}}$' + '\n'
        # Width
        ptext += r'$\mathrm{FWHM} = '
        fwhm = [np.sqrt(8.0*np.log(2.0))*_ for _ in p['stddev']]
        ptext += '{:.1f}_{{{:+.1f}}}^{{{:+.1f}}}$'.format(*fwhm)
        ptext += r'$\mathrm{\ km\ s^{-1}}$'

        ax.text(0.95, 0.95, ptext.replace('nan', r'\infty'),
                ha='right', va='top', fontsize='small',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.7))

    ax.set(xlim=[-150, 200],
           xlabel='Heliocentric Velocity',
           ylabel='Core-subtracted profile',
           title='{:s} - {:s} - {:s}'.format(os.path.basename(save_dir),
                                             slit_id, LINE_LABEL[line_id]),
    )
    fig.set_size_inches(8, 6)
    knot_id = os.path.basename(save_dir)
    plotfile = os.path.join(save_dir,
                            '{}-knot-fit-{}-{}.png'.format(
                                line_id, knot_id, slit_id))
    fig.savefig(plotfile, dpi=200)
    # Important to close figure explicitly so as not to leak resources
    plt.close(fig)
