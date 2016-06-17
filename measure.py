'''
Run this on the WWU Compute Cluster, where the DR24 Kepler data is currently stored
'''

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

# the target list from Tessa
targs = 'Mdwarf_kic_ra_dec.txt'
tfiles = pd.read_csv('Mdwarf_kic_ra_dec.txt')

# the list of all 3M Kepler light curve files on the WWU server
fitslist = '/home/davenpj3/data/kepler/all_fits.lis'
fitsfiles = pd.read_table(fitslist, delim_whitespace=True, header=None)

# loop thru every target
for k in range(len(tfiles)):
    # find all the FITS files that match Kep ID
    x1 = fitsfiles[0].str.contains(tfiles['Kepler ID'][k])

    # let us only consider the LLC data
    x2 = fitsfiles[0][x1].str.contains('llc')

    # the list of LLC light curve files that match the k'th target
    kfiles = fitsfiles[0][x1][x2].values

    # now loop over all these files...
    for j in range(len(kfiles)):

        #-- code taken from Appaloosa --
        # read each FITS file in
        hdu = fits.open(kfiles[j])
        data_rec = hdu[1].data

        time = data_rec['TIME']
        flux = data_rec['SAP_FLUX']
        error = data_rec['SAP_FLUX_ERR']
        sap_quality = data_rec['SAP_QUALITY']

        isrl = np.isfinite(flux)
        time = time[isrl]
        sap_quality = sap_quality[isrl]
        flux = flux[isrl]
        error = error[isrl]

        tot_med = np.nanmedian(flux)

        krnl = 10.
        order = 2.

        flux_sm = pd.rolling_median(flux, krnl)
        indx = np.isfinite(flux_sm)  # get rid of NaN's put in by rolling_median.

        fit = np.polyfit(time[indx], flux_sm[indx], order)

        flux_flat = flux - np.polyval(fit, time) + tot_med

        # per the McQuillan (2014) approach, measure 5-95 percentiles
        # do in steps of a couple P_rot, then take median
        prange_j = np.percentile(flux_flat, (5,95))