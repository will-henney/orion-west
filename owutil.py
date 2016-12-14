def ra_ow(ra):
    """Convert astropy.coordinates RA to OW96 scheme"""
    h, m, s = ra.hms
    assert(int(h) == 5 and int(m/10) == 3)
    ra_code = "{:04d}".format(int((m - 30)*1000 + 10*(s + 0.05)))
    if ra_code.startswith('5'):
        ra_code = ra_code[1:]
    return ra_code

def dec_ow(dec):
    """Convert astropy.coordinates Dec to OW96 scheme"""
    d, m, s = dec.dms
    assert(int(d) == -5)
    dec_code = "{:04d}".format(int(abs(m)*100 + abs(s) + 0.5))
    if dec_code.startswith('2'):
        dec_code = dec_code[1:]
    return dec_code


def ow_from_coord(c):
    return "{}-{}".format(ra_ow(c.ra), dec_ow(c.dec))


if __name__ == '__main__':
    from astropy import coordinates as coord

    c = coord.get_icrs_coordinates('tet01 ori c')

    print('theta 1 C Ori is', ow_from_coord(c))
