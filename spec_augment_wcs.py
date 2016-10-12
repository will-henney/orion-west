# [[file:alba-orion-west.org::*Program%20to%20add%20a%20better%20wcs%20to%20the%20spectra:%20spec_augment_wcs.py][Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:1]]
import sys
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from helio_utils import waves2vels
# Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:1 ends here

# [[file:alba-orion-west.org::*Program%20to%20add%20a%20better%20wcs%20to%20the%20spectra:%20spec_augment_wcs.py][Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:2]]
def get_specmap_wcs():
    pixel_scale = 0.5               # arcsec
    NX, NY = 2048, 2048
    dRA, dDec = -pixel_scale/3600., pixel_scale/3600.
    RA0, Dec0 = 83.69, -5.429
    w = WCS(naxis=2)
    w.wcs.crpix = [0.5*(1 + NX), 0.5*(1 + NY)]
    w.wcs.cdelt = [dRA, dDec]
    w.wcs.crval = [RA0, Dec0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']
    return w
# Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:2 ends here

# [[file:alba-orion-west.org::*Program%20to%20add%20a%20better%20wcs%20to%20the%20spectra:%20spec_augment_wcs.py][Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:3]]
def velocity_world2pix(wcs, vels, iaxis=0):
    """Convert velocities `vels` to pixels by hand using the CDELT, CRPIX
and CRVAL from `wcs`.  The velocity axis in the wcs is given by
`iaxis`

    """
    crval = wcs.wcs.crval[iaxis]
    crpix = wcs.wcs.crpix[iaxis]
    cdelt = wcs.wcs.cdelt[iaxis]
    cunit = wcs.wcs.cunit[iaxis]
    print('CRVAL, CRPIX, CDELT, CUNIT', crval, crpix, cdelt, cunit)
    pixels = crpix + (vels - crval)/cdelt
    # return fractional pixel coordinates on 0-based scale
    return pixels - 1

def fix_up_some_new_wcs(filename, old_new=('.fits', '-vhel.fits')):
    hdu = fits.open(filename)[0]  # Always use first HDU in file
    # Start with the Wav, RA, Dec WCS
    wold = WCS(hdu.header, key='A')

    # This is where we will put the new stuff
    wnew = wold.deepcopy()

    # First do the velocity part
    wav0, dwav = wold.wcs.crval[0], wold.wcs.cdelt[0]
    vel0, vel1 = waves2vels([wav0, wav0 + dwav]*u.m,
                            wold.wcs.restwav*u.m, hdu.header)
    dvel = vel1 - vel0
    wnew.wcs.crval[0] = vel0.to('m/s').value
    wnew.wcs.cdelt[0] = dvel.to('m/s').value
    wnew.wcs.cunit[0] = u.m/u.s
    wnew.wcs.ctype[0] = 'VOPT'
    wnew.wcs.cname[0] = 'Heliocentric velocity'
    wnew.wcs.specsys = 'HELIOCEN'
    wnew.wcs.name = "VHELIO"

    # Now do the spatial part
    wim = get_specmap_wcs()
    RAs, Decs = wold.celestial.all_pix2world([0, 1, 0], [0, 0, 1], 0)
    # print('First two pixels RA and Dec:', RAs, Decs)
    [X0, X1, X2], [Y0, Y1, Y2] = wim.all_world2pix(RAs, Decs, 0)
    # print('First two pixels X and Y:', [X0, X1, X2], [Y0, Y1, Y2])
    wnew.wcs.crval[1:] = [X0, Y0]
    wnew.wcs.cdelt[1:] = [1., 1.]
    wnew.wcs.pc[1:, 1:] = [[X1 - X0, Y1 - Y0], [X2 - X0, Y2 - Y0]]
    wnew.wcs.ctype[1], wnew.wcs.ctype[2] = ['LINEAR']*2
    wnew.wcs.cname[1], wnew.wcs.cname[2] = ['X', 'Y']
    wnew.wcs.cunit[1], wnew.wcs.cunit[2] = [u.dimensionless_unscaled]*2

    # Cut off the velocity range in the data array to [-150..200] km/s
    wnew.fix()                  # Make sure we know it is in SI units
    print(wnew.sub([1]).wcs)
    vwindow = [-150, 200]*u.km/u.s
    print('Velocity window (m/s)', vwindow.to('m/s').value)
    [j1, j2] = velocity_world2pix(wnew, vwindow.to('m/s').value)
    view = slice(None), slice(None), slice(int(j1), int(j2) + 2)
    print('Pixel limits for slice', j1, j2)
    # Apply slice to data and to the WCS
    hdu.data = hdu.data[:, :, j1:j2]
    wnew = wnew.slice(view)
    wold = wold.slice(view)

    # Update header with a new WCS called V
    hdu.header.update(wnew.to_header(key='V'))
    # And re-write 'A' too since we have changed it
    hdu.header.update(wold.to_header(key='A'))

    # Now, sort out the default header

    # New blank wcs with only 2 dimensions
    wdef = WCS(naxis=2)
    # Copy over the velocity part 
    for k in 'crval', 'crpix', 'cdelt', 'cunit', 'ctype', 'cname':
        getattr(wdef.wcs, k)[0] = getattr(wnew.wcs, k)[0]
    wdef.wcs.pc[0, 0] = wnew.wcs.pc[0, 0]

    # Check for orientation
    slit_center = hdu.header['NAXIS2']/2
    if abs(wnew.wcs.pc[1, 1]) > abs(wnew.wcs.pc[1, 2]):
        # largely horizontal slit - use X-axis
        jslit = 1
        _, _, [Ycent] = wnew.all_pix2world([0], [slit_center], [0], 0) 
        wdef.wcs.name = 'YY{:04d}'.format(int(Ycent))
    else:
        # largely vertical slit - use Y-axis
        jslit = 2
        _, [Xcent], _ = wnew.all_pix2world([0], [slit_center], [0], 0) 
        wdef.wcs.name = 'XX{:04d}'.format(int(Xcent))

    # Copy over spatial part (X or Y, depending on orientation)
    for k in 'crval', 'crpix', 'cunit', 'ctype', 'cname':
        getattr(wdef.wcs, k)[1] = getattr(wnew.wcs, k)[jslit]
    # More intuitive to use CDELT instead of PC
    wdef.wcs.cdelt[1] = wnew.wcs.pc[1, jslit]

    # Update header with new default WCS
    hdu.header.update(wdef.to_header(key=' '))
    # Convert from m/s to km/s
    hdu.header['CUNIT1'] = 'km/s'
    hdu.header['CRVAL1'] /= 1000.
    hdu.header['CDELT1'] /= 1000.
    # Remove the pesky CD keywords
    for ij in '1_1', '1_2', '2_2', '1_1':
        if 'CD'+ij in hdu.header:
            hdu.header.remove('CD'+ij)

    # And flatten data array to 2-dimensions
    assert(len(hdu.data.shape) == 3)
    print('Original data array shape:', hdu.data.shape)
    hdu.data, = hdu.data
    print('New data array shape:', hdu.data.shape)

    # Write a new file 
    newfilename = filename.replace(*old_new).replace('BGsub/',
                                                     'BGsub/' + wdef.wcs.name + '-')
    print('Writing', newfilename)
    hdu.writeto(newfilename, clobber=True)
# Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:3 ends here

# [[file:alba-orion-west.org::*Program%20to%20add%20a%20better%20wcs%20to%20the%20spectra:%20spec_augment_wcs.py][Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:4]]
if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except:
        print('Usage:', sys.argv[0], 'FITSFILE')
        sys.exit()

    fix_up_some_new_wcs(fn)
# Program\ to\ add\ a\ better\ wcs\ to\ the\ spectra:\ spec_augment_wcs\.py:4 ends here
