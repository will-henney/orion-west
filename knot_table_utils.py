from collections import OrderedDict
import json
import glob
import numpy as np
import pandas as pd

def data_from_json(fn):
    """Returns all data from json file named `fn` in form of dict"""
    with open(fn) as f:
        data = json.load(f)
    return data


# Include 'Red other' component, since this often is quite bright and
# probably isn't a knot at all
CORE_COMPONENTS = ['G0', 'G1', 'G2',
                   'G3', 'G4', 'Red other']
def find_core_moments(data):
    """Estimate moments from the fit parameters for the core

    Returns (mean, sigma)
    """
    vels = []
    fluxes = []
    sigmas = []
    for name, params in data.items():
        if name in CORE_COMPONENTS:
            vels.append(params['mean'])
            fluxes.append(params['amplitude'])
            sigmas.append(params['stddev'])

    vels = np.array(vels)
    fluxes = np.array(fluxes)
    sigmas = np.array(sigmas)

    m = np.isfinite(vels) & np.isfinite(fluxes) & np.isfinite(sigmas)

    vmean = np.average(vels[m], weights=fluxes[m])
    variance = np.average((vels[m]-vmean)**2, weights=fluxes[m])
    variance += np.average(sigmas[m]**2, weights=fluxes[m])

    return vmean, np.sqrt(variance)


def summarise_data(d):
    """Summarise data from json file into form suitable for table

    Returns OrderedDict so we have control over the column order
    """
    out = OrderedDict()
    out['line'] = d['emission line']
    out['knot'] = d['knot']
    out['slit'] = d['slit']
    out['Vnom'] = d['nominal knot velocity']
    # out['Wc'] = d['core fit moments']['FWHM']
    # out['Vc'] = d['core fit moments']['mean velocity']
    u0, sigma = find_core_moments(d['core fit parameters'])
    out['Vc'] = u0
    out['Wc'] = np.sqrt(8*np.log(2.0)) * sigma
    out['Fc'] = d['core fit moments']['flux']
    out['F'] = d['knot fit moments']['flux']
    try:
        out['A'] = d['knot fit parameters']['amplitude'][0]
        out['dA-'] = d['knot fit parameters']['amplitude'][1]
        out['dA+'] = d['knot fit parameters']['amplitude'][2]
        out['V'] = d['knot fit parameters']['mean'][0]
        out['dV-'] = d['knot fit parameters']['mean'][1]
        out['dV+'] = d['knot fit parameters']['mean'][2]
        out['W'] = d['knot fit parameters']['FWHM'][0]
        out['dW-'] = d['knot fit parameters']['FWHM'][1]
        out['dW+'] = d['knot fit parameters']['FWHM'][2]
        out['chi2'] = d['knot fit parameters']['reduced chi^2']
        # Quality of fit: amplitude / amplitude error
        out['Q'] = 2*out['A']/(out['dA+'] - out['dA-'])
        out['chi2c'] = d['core fit parameters']['reduced chi^2']
    except KeyError:
        out['Q'] = 0.0
        for k in ['A', 'dA-', 'dA+',
                  'V', 'dV-', 'dV+',
                  'W', 'dW-', 'dW+',
                  'chi2', 'chi2c']:
            # Fill with NaN only those items that we failed to get
            if not k in out:
                out[k] = np.nan

    return out

JSON_FILE_GLOB = 'Knot-Fits/*/*.json'

def _dictlist_from_json_files(debug=False):
    """Returns list of OrderedDict rows"""
    dictlist = []
    for fn in glob.glob(JSON_FILE_GLOB):
        if debug:
            print('Appending data from', fn)
        dictlist.append(summarise_data(data_from_json(fn)))
    return dictlist


def _dataframe_from_dictlist(data):
    """Return a pandas dataframe"""
    df = pd.DataFrame(data=data, columns=data[0].keys())
    # Use a MultiIndex 
    df = df.set_index(['knot', 'slit', 'line'])
    # And move the line (ha or nii) to the columns
    df = df.unstack()
    return df


def get_dataframe():
    return  _dataframe_from_dictlist(_dictlist_from_json_files())
