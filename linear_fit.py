# CODE WRITTTEN BY JALYN KRAUSE FROM MAY 2018 - PRESENT
# CONSTRUCTED FOR CHRISTY'S RESEARCH PROJECT
# METALLICITY INDICATORS DISPLAY BREAKS IN GRADIENT


# PACKAGES NEEDED FOR PROJECT
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from astropy.io import fits
from astropy.stats import bootstrap
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.cosmology import FlatLambdaCDM as flc
from astropy.table import Table, Column
from astropy.modeling import models
from scipy import optimize
from scipy import stats
from scipy.stats import powerlaw
import astropy.table as t
import numpy as np
import math
import warnings
import random
import os
import sys
import scipy.stats.distributions as dist
import lmfit
from lmfit import minimize, Parameters

# Ignores annoying warnings
plt.rcParams.update({'figure.max_open_warning': 0})
np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter("ignore")

def get_data(): 
    fits = 'zooinverse_summary_v2.fit'
    hdu = open_fits(fits)
    # the 39 columns in fits table
    plateifu = hdu[1].data['PLATEIFU']
    mangaid = hdu[1].data['MANGAID']
    ra = hdu[1].data['RA']
    dec = hdu[1].data['DEC']
    z = hdu[1].data['Z']
    fnugriz_absmag = hdu[1].data['FNUGRIZ_ABSMAG'] # array of 7 colors in this order:  FUV, NUV, u, g, r, i, z
    fuv_r = fnugriz_absmag[:,0] - fnugriz_absmag[:,4] # FUV-r color 
    log_mass = hdu[1].data['LOG_MASS']
    subject_id = hdu[1].data['SUBJECT_ID']
    nclass = hdu[1].data['NCLASS']
    bad_re = hdu[1].data['BAD_RE']
    bad_re_err = hdu[1].data['BAD_RE_ERR']
    pa_shift = hdu[1].data['PA_SHIFT']
    pa_shift_error = hdu[1].data['PA_SHIFT_ERR']
    kine_twist = hdu[1].data['KINE_TWIST']
    kine_twist_err = hdu[1].data['KINE_TWIST_ERR']
    disturbed_kine = hdu[1].data['DISTURBED_KINE']
    disturbed_kine_err = hdu[1].data['DISTURBED_KINE_ERR']
    merging = hdu[1].data['MERGING']
    merging_err = hdu[1].data['MERGING_ERR']
    sym_OH = hdu[1].data['SYMMETRIC_OH']
    sym_OH_err = hdu[1].data['SYMMETRIC_OH_ERR']
    distorted_OH = hdu[1].data['DISTORTED_OH']
    distorted_OH_err = hdu[1].data['DISTORTED_OH_ERR']
    chaotic_OH = hdu[1].data['CHAOTIC_OH']
    chaotic_OH_err = hdu[1].data['CHAOTIC_OH_ERR']
    bad_OH = hdu[1].data['BAD_OH']
    bad_OH_err = hdu[1].data['BAD_OH_ERR']
    low_knots = hdu[1].data['LOW_KNOTS']
    low_knots_err = hdu[1].data['LOW_KNOTS_ERR']
    high_knots = hdu[1].data['HIGH_KNOTS']
    high_knots_err = hdu[1].data['HIGH_KNOTS_ERR']
    linear_OHgrad = hdu[1].data['LINEAR_OHGRAD']
    linear_OHgrad_err = hdu[1].data['LINEAR_OHGRAD_ERR']
    slope_change = hdu[1].data['SLOPE_CHANGE']
    slope_change_err = hdu[1].data['SLOPE_CHANE_ERR']
    irr_OHgrad = hdu[1].data['IRREGULAR_OHGRAD']
    irr_OHgrad_err = hdu[1].data['IRREGULAR_OHGRAD_ERR']
    bad_OHgrad = hdu[1].data['BAD_OHGRAD']
    bad_OHgrad_err = hdu[1].data['BAD_OHGRAD_ERR']
    return distorted_OH, slope_change # return data needed

# newest fits file fraction of symmetric O/H map vs. stellar mass
def plot(x, y):
    nbins = 20 # arbitrary number
    bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins) # sets up the edges of the bins on yaxis
    bins_cens = np.zeros(nbins-1) # returns a new array filled with zeros
 
    # creates centers of bins by calculating half way
    for j in range(0, nbins-1):
        bins_cens[j] = ((bins[j] + bins[j+1]) / 2.0) 

    frac = np.zeros(len(bins_cens)) # returns a new array filled with zeros
    c = np.zeros(len(bins_cens)) # returns a new array filled with zeros
    err = np.zeros(len(bins_cens)) # returns a new array filled with zeros
    
    for i in range(0, len(bins) - 1): 
        in_bin = np.where((x > bins[i]) & (x < bins[i+1]))[0] # need the '[0]' because np.where returns a tuple with an array of indices.
        total = len(in_bin) # total number of galaxies in the bin
        # anyone classified galaxy having characteristic
        summed_a = np.sum(y[np.where((x > bins[i]) & (x < bins[i + 1]) & (y > 0))])
        # everyone classified galaxy having characteristic
        summed_e = np.sum(y[np.where((x > bins[i]) & (x < bins[i + 1]) & (y == 1.0 ))])
        data = y[np.where((x > bins[i]) & (x < bins[i + 1]) & (y > 0))]
        boot = bootstrap(data, bootnum = 1000, samples = None, bootfunc = None)
        boot_frac = np.zeros(1000)
        k = 0
        frac[i] =  summed_a / total
        for row in boot:
            c = np.sum(row)
            boot_frac[k] = c / total 
            k += 1
        err[i] = statistics.stdev(boot_frac, xbar = None)  # xbar=none means that the mean is automatically calculated   
                
    # removes any points that are equal to zero
    frac[np.logical_and(frac == 0.0, bins_cens > 0.0)] = np.nan

    # plotted line for fits_v2 file
    plt.plot(bins_cens, frac, lw = 3)
    plt.scatter(bins_cens, frac, lw = 3)
    plt.errorbar(bins_cens, frac, err)
    plt.title('Fraction of Galaxies that have a Slope Change in the O/H Radial Profile vs. Distorted O/H Map Contours')
    plt.ylabel('Fraction')
    plt.xlabel('Distorted O/H Map Contours')
    plt.show()
    #plt.savefig('/Users/jalynkrause/Documents/astro/frac_slopechange_vs_dis_OH_2.png')
    plt.close()

def bootstrap(data, bootnum=1000, samples=None, bootfunc=None): 
    # from http://docs.astropy.org/en/v0.3/_modules/astropy/stats/funcs.html#bootstrap
    if samples is None:
        samples = data.shape[0]
    
    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
        boot = np.empty(resultdims)
    else:
        resultdims = (bootnum,)
        boot = np.empty(resultdims)
        
    for i in range(0, bootnum):
        bootarr = np.random.randint(low = -1, high = data.shape[0], size = samples) # data.shape returns dimension of rows in array
        boot[i] = data[bootarr]
        
    return boot # Returns np.ndarray Bootstrapped data. Each row is a bootstrap resample of the data.    

# function for getting effective radius of a specific galaxy
def get_reff(plateifu):
    hdu = fits.open('/Users/jalynkrause/Documents/astro/drpall-v2_4_3.fits')
    indx = hdu[1].data['PLATEIFU'] == plateifu
    reff = hdu[1].data['NSA_ELPETRO_TH50_R'][indx][0]
    return reff

# function for opening fits file 
def open_fits(fit):
    hdu = fits.open(fit) # version 2 of the fits file
    return hdu 

def get_metdir(feature_str):
    metdir = '/Users/jalynkrause/Documents/astro/slope_changes/metmaps_n2o2/' + feature_str + '/'
    return metdir

def rp_for_fit_2(image, centre=None, distarr=None, mask=None, binwidth=1, radtype='weighted'):
    # code from Adam
    '''
    image = 2D array to calculate RP of.
    centre = centre of image in pixel coordinates. Not needed if distarr is given.
    distarr = 2D array giving distance of each pixel from the centre.
    mask = 2D array, 1 if you want to include given pixels, 0 if not.
    binwidth = width of radial bins in pixels.
    radtype = 'weighted' or 'unweighted'. Weighted will give you the average radius of pixels in the bin. Unweighted will give you the middle of each radial bin.
    '''
    
    distarr = distarr / binwidth
    if centre is None:
        centre = np.array(image.shape, dtype = float) / 2
    if distarr is None:
        y,x = np.indices(image.shape)
        distarr = np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    if mask is None:
        mask = np.ones(image.shape)
        
    rmax = int(np.max(distarr))
    r_edge = np.linspace(0,rmax,rmax+1)
    rp = np.zeros(len(r_edge) -1) * np.nan
    nums = np.zeros(len(r_edge) -1) * np.nan
    sig = np.zeros(len(r_edge) -1) * np.nan
    r = np.zeros(len(r_edge) -1) * np.nan
    
    for i in range(0, len(r)):
        rp[i] = np.nanmean(image[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))])
        nums[i] = len(np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False) & (np.isnan(image)==False))[0])
        sig[i] = np.nanstd((image[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))]))
        if radtype == 'unweighted':
            r[i] = (r_edge[i]+r_edge[i+1]) / 2.0
        elif radtype == 'weighted':
            r[i] = np.nanmean(distarr[np.where((distarr>=r_edge[i]) & (distarr<r_edge[i+1]) & (mask==1.0) & (np.isinf(image)==False))])
    
    r = r * binwidth
    
    return r, rp, nums, sig

    '''
    The 4 values returned are:
    r = the radius vector for the points where you calculate the average metallicitys
    rp = the average value of metallicity at that radius
    nums = the number of points contributing to that average
    sig = the standard deviation of metallicities in that radial bin
    '''
    
def get_map_n2o2(plateifu, cat, re):
    metdir = '/Users/jalynkrause/Documents/astro/slope_changes/metmaps_n2o2/' + cat + '/'
    met_n2o2 = np.load(metdir + plateifu + '_n2o2_met.npy') # Metallicity measurement
    met_err_n2o2 = np.load(metdir + plateifu + '_n2o2_met_err.npy')  # Error array of metallicity map
    issf = np.load(metdir + plateifu + '_issf.npy') # BPT mask
    mask = np.load(metdir + plateifu + '_n2o2_mask.npy') # Indicates if a spaxel has emission due to star formation or not: 1 = good, 0 = bad
    rad = np.load(metdir + plateifu + '_radius.npy') # Distance of each spaxel from the centre in arcsec
    rad_n2o2 = rad/re # Radius[arcsec] / eff Radius[arcsec]
    rad_n2o2_pix = rad*2
    badpix = np.where((mask == 0) | (issf == 0) | (met_err_n2o2 > 0.1))
    met_n2o2[badpix] = np.nan
    badmet = np.where(met_n2o2 < 8.6) # Metallicities below 8.6 are not reliable with this indicator
    met_n2o2[badmet] = np.nan
        
    return met_n2o2, met_err_n2o2, rad_n2o2, rad_n2o2_pix

def get_map_n2s2(plateifu, cat, re):
    metdir = '/Users/jalynkrause/Documents/astro/slope_changes/metmaps_n2s2/' + cat + '/'
    met_n2s2 = np.load(metdir + plateifu + '_n2s2_met.npy') # Metallicity measurement
    met_err_n2s2 = np.load(metdir + plateifu + '_n2s2_met_err.npy')  # Error array of metallicity map
    issf = np.load(metdir + plateifu + '_issf.npy') # BPT mask
    mask = np.load(metdir + plateifu + '_n2s2_mask.npy') # Indicates if a spaxel has emission due to star formation or not: 1 = good, 0 = bad
    rad = np.load(metdir + plateifu + '_radius.npy') # Distance of each spaxel from the centre in arcsec
    rad_n2s2 = rad/re # Radius[arcsec] / eff Radius[arcsec]
    rad_n2s2_pix = rad*2
    badpix = np.where((mask == 0) | (issf == 0) | (met_err_n2s2 > 0.1))
    met_n2s2[badpix] = np.nan
        
    return met_n2s2, met_err_n2s2, rad_n2s2, rad_n2s2_pix

# function to plot map comparing both metallicity indicators
def plot_map(met_n2o2, met_err_n2o2, rad_n2o2, rp_n2o2, met_n2s2, met_err_n2s2, rad_n2s2,vrp_n2s2, plateifu, re, cat):
    #plt.imshow(met, origin = 'lower')
    plt.figure()
    plt.xlabel('R/$R_e$')
    plt.ylabel('Metallicity')
    plt.title(plateifu)
    plt.errorbar(rad_n2o2.ravel(),met_n2o2.ravel(),yerr=met_err_n2o2.ravel(),fmt='o',ecolor='lightcoral',mfc='maroon',mec='maroon',ms=3,label='N2O2 Metallicity Indicator')
    plt.errorbar(rad_n2s2.ravel(),met_n2s2.ravel(),yerr=met_err_n2s2.ravel(),fmt='o',ecolor='steelblue',mfc='navy',mec='navy',ms=3,label='N2S2 Metallicity Indicator')
    #plt.axvline(x=2.5/(2*re), color='darkgrey', zorder=20)
    plt.legend(loc = 'lower left') # try changing loc to 'best' instead of 'lower left' for positioning?
    
    # To plot midpoints for n2s2
    met_n2s2, met_err_n2s2, rad_n2s2, rad_n2s2_pix = get_map_n2s2(plateifu, cat, re)
    r_n2s2, rp_n2s2, nums_n2s2, sig_n2s2 = rp_for_fit_2(met_n2s2, distarr=rad_n2s2_pix, radtype='weighted')
    r_n2s2 = r_n2s2/(2*re)
    plt.plot(r_n2s2, rp_n2s2,c='k',lw=1, zorder=10)
    
    # To plot midpoints for n2o2
    met_n2o2, met_err_n2o2, rad_n2o2, rad_n2o2_pix = get_map_n2o2(plateifu, cat, re)
    r_n2o2, rp_n2o2, nums_n2o2, sig_n2o2 = rp_for_fit_2(met_n2o2, distarr=rad_n2o2_pix, radtype='weighted') 
    r_n2o2 = r_n2o2/(2*re)
    plt.plot(r_n2o2, rp_n2o2,c='k',lw=1, zorder=10)
    
    #plt.plot(x, y, "o")
    plt.show()
    plt.close() 
    #plt.savefig('/Users/jalynkrause/Documents/astro/slope_changes/bars_plots/' + plateifu + '.png')
    return

# Function to get redshift for each galaxy
def get_redshift(plateifu, drpall):
    hdu = fits.open(drpall)
    indx = hdu[1].data['PLATEIFU'] == plateifu
    return hdu[1].data['NSA_Z'][indx][0]

# Function to get the b/a ratio (inclination) for each galaxy
def get_ba(plateifu, drpall):
    hdu = fits.open(drpall)
    indx = hdu[1].data['PLATEIFU'] == plateifu
    return hdu[1].data['nsa_elpetro_ba'][indx][0]

# Functions to extract the data and convert it to a stellar mass surface density in the units, log(Sigma_{*} / M_{sun} kpc^{2})
def get_massmap(plateifu, p3ddir, drpall):
    cosmo = flc(H0 = 70,Om0 = 0.3) # Chosen cosmological parameters

    hdu = fits.open(p3ddir+'manga-'+plateifu+'.Pipe3D.cube.fits.gz')    
    mass = hdu[1].data[19,:,:] # Stellar Mass density per pixel, dust-corrected, log10(M*/spaxel)
    mass_err = hdu[1].data[20,:,:]
    #convert from mass/spaxel to mass/kpc^2 with an inclination correction
    z = get_redshift(plateifu, drpall)
    ba = get_ba(plateifu, drpall)
    sig_cor = 2 * np.log10((0.5/cosmo.arcsec_per_kpc_proper(z).value))
    mass = mass - sig_cor + np.log10(ba)
    mass_err = mass_err - sig_cor + np.log10(ba)

    return mass, mass_err

 # Function for getting a galaxy's radius in arcseconds
def get_radius(plate, ifu):
    rad = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/radius.npy') # Elliptical radius in units of arcseconds
    return rad

def get_met_n2o2(plate, ifu):
    n2o2_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met.npy') # KD04 [NII]/[OII] metallicity (12+log(O/H)). Not reliable for 12+log(O/H)<8.6
    n2o2_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met_err.npy') # Error on the N2O2 metallcity. This is derived by a Monte Carlo method.
    n2o2_mask = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_mask.npy') # 1 if ([NII]6583/[NII]6583_err > 3 ) & ([OII]3727/[OII]3727_err > 3) & ([OII]3729/[OII]3729_err > 3 ) & (ha/ha_err>3) & (hb/hb_err > 3), 0 otherwise.
    issf = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/issf.npy') # 1 if a spaxel is considered to be in the star-forming part of the [OIII]/Hb, [NII/Ha] BPT diagram (passes both Kauffmann and Kewley criteria), 0 if not
    ha_ew = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/ha_ew.npy')
    
    n2o2_bad = (n2o2_mask == 0) | (issf == 0) | (n2o2_met_err > 0.1) | (ha_ew < 6)
    n2o2_met[n2o2_bad] = np.nan
    n2o2_met_err[n2o2_bad] = np.nan
    n2o2_met[n2o2_met < 8.6] = np.nan # Metallicities below 8.6 are not reliable with this indicator
    
    return n2o2_met

def six_pannel(plateifu, plate, ifu, mass, mass_err, reff):
    plt.figure(figsize = (20,9), facecolor = 'white')
 
    rad = get_radius(plate, ifu) # Elliptical radius in units of arcseconds

    # SFR surface density in units of M_sun/yr/kpc^2. It assumes a flat lambdaCDM cosmology with h=0.7, Omega_lambda=0.7, Omega_matter=0.3. SFR surface density is corrected for inclination using the Petrossian b/a values, so they are probably not accurate for highly inclined systems. 
    sfrd = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/sfr_map.npy')
    
    sfrd_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/sfr_map_err.npy') # Error on the SFRD
    # Elliptical radius in units of arcseconds
        
    issf = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/issf.npy') # 1 if a spaxel is considered to be in the star-forming part of the [OIII]/Hb, [NII/Ha] BPT diagram (passes both Kauffmann and Kewley criteria), 0 if not
    ha_ew = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/ha_ew.npy')
    
    n2s2_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_met.npy') # Dopita N2S2 Halpha metallcity
    
    n2s2_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_met_err.npy') # Error on Dopita 2016 metallicity, calculated by standard error propagation
    n2s2_mask = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_mask.npy') # Mask for N2S2 metallcity. 1 if S/N of all lines used >3
    
    n2o2_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met.npy') # KD04 [NII]/[OII] metallicity (12+log(O/H)). Not reliable for 12+log(O/H)<8.6
    
    n2o2_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met_err.npy') # Error on the N2O2 metallcity. This is derived by a Monte Carlo method.
    
    n2o2_mask = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_mask.npy') # 1 if ([NII]6583/[NII]6583_err > 3 ) & ([OII]3727/[OII]3727_err > 3) & ([OII]3729/[OII]3729_err > 3 ) & (ha/ha_err>3) & (hb/hb_err > 3), 0 otherwise.
    
    pp04_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/pp04_o3n2_met.npy')
    
    pp04_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/pp04_o3n2_met_err.npy')
    
    n2o2_bad = (n2o2_mask == 0) | (issf == 0) | (n2o2_met_err > 0.1) | (ha_ew < 6)
    n2o2_met[n2o2_bad] = np.nan
    n2o2_met_err[n2o2_bad] = np.nan
    n2o2_met[n2o2_met < 8.6] = np.nan # Metallicities below 8.6 are not reliable with this indicator

    n2s2_bad = (n2s2_mask == 0) | (issf == 0) | (n2s2_met_err > 0.1) | (ha_ew < 6)
    n2s2_met_err[n2s2_bad] = np.nan
    n2s2_met[n2s2_bad] = np.nan
    
    pp04_bad = (issf == 0) | (pp04_met_err > 0.1) | (ha_ew < 6)
    pp04_met_err[pp04_bad] = np.nan
    pp04_met[pp04_bad] = np.nan
    
    bad_mass = mass < 3
    mass[bad_mass] = np.nan
    
    sfrd_bad = (issf == 0)
    sfrd[sfrd_bad] = np.nan
    
    rad = rad/reff # Radial measurements now in units of Re, was in arcseconds
    rmax = int(np.max(rad))
    r_edge = np.linspace(0,rmax,len(rad))
    r = np.zeros(len(r_edge) -1) * np.nan # Radius
    s = np.zeros(len(r_edge) -1) * np.nan # SFRD
    m = np.zeros(len(r_edge) -1) * np.nan # Stellar mass surface density
    rp_n2s2 = np.zeros(len(r_edge) -1) * np.nan # rp = the average value of metallicity at that radius
    rp_n2o2 = np.zeros(len(r_edge) -1) * np.nan
    rp_pp04 = np.zeros(len(r_edge)-1) * np.nan
    
    # Midpoints
    for i in range(0, len(rp_n2s2)-1):
        rp_n2s2[i] = np.nanmean(n2s2_met[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
        r[i] = np.nanmean(rad[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
        s[i] = np.nanmean(sfrd[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
        m[i] = np.nanmean(mass[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
        
    for j in range(0, len(rp_n2o2)-1):
        rp_n2o2[j] = np.nanmean(n2o2_met[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])
        r[j] = np.nanmean(rad[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])
        s[j] = np.nanmean(sfrd[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])
        m[j] = np.nanmean(mass[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])
        
    for k in range(0, len(rp_pp04)-1):
        rp_pp04[k] = np.nanmean(pp04_met[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])
        r[k] = np.nanmean(rad[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])
        s[k] = np.nanmean(sfrd[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])
        m[k] = np.nanmean(mass[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])    
     
        
    ################################################################################################################
    ############# Creates 6 pannel plots comparing N2O2 and N2S2 indicators side by side ###########################
    ################################################################################################################

    
    # plot for the ratio of O/H for N2S2 metallicity indicator vs. Radius
    plt.subplot(3,2,1)
    plt.scatter(rad, n2s2_met, s=15, color = 'purple')
    plt.plot(r, rp_n2s2, color = 'k') # plots midpoints
    plt.axvline(x = 0.5*reff, color = 'plum') # plots vertical line at .5 effective radii
    plt.axvline(x = 2*reff, color = 'plum') # plots vertical line at 2 effective radii
    plt.xticks(fontsize=14) # increases size of numbers on tick marks
    plt.yticks(fontsize=14) # increases size of numbers on tick marks
    plt.xlabel('Arcseconds', fontsize = 'xx-large')
    #plt.ylabel('[NII]/[SII] (12+log(O/H))', fontsize = 'xx-large')
    plt.ylabel('Oxygen Abundance (12+log(O/H)', fontsize = 'xx-large') # label for undergrad symposium example
    plt.title('Radial Oxygen Abundance', fontsize = 'xx-large') # label for undergrad symposium example

    # plot for the ratio of O/H for the N2O2 metallicity indicator vs. Radius
    plt.subplot(3,2,2)
    plt.scatter(rad, n2o2_met, s=15, color = 'r')
    plt.axvline(x = 0.5*reff, color = 'm')
    plt.axvline(x = 2*reff, color = 'm')
    plt.plot(r, rp_n2o2, color = 'k') # plots midpoints
    plt.xticks(fontsize=14) # increases size of numbers on tick marks
    plt.yticks(fontsize=14) # increases size of numbers on tick marks
    plt.xlabel('Arcseconds', fontsize = 'xx-large')
    plt.ylabel('[NII]/[OII] (12+log(O/H))', fontsize = 'xx-large')
    
    # plot of Stellar Mass Surface Density vs. Radius
    plt.subplot(3,2,3)
    plt.scatter(rad, mass, s=4, color = 'r')
    plt.plot(r, m, color='k') # plots midpoints
    plt.xlabel('Radius (arcseconds)')
    plt.ylabel('$\Sigma_{M_{*}} (log(\Sigma_* / M_\odot kpc^2$))')
    
    # plot for SFR density vs. Radius
    plt.subplot(3,2,4)
    plt.scatter(rad, np.log10(sfrd), s=4, color = 'r')
    plt.plot(r, np.log10(s), color = 'k') # plots midpoints
    plt.xlabel('Radius (arcseconds)',)
    plt.ylabel('$\Sigma_{SFR} (log(M_\odot/yr/kpc^2)$)') 
    
    # plots the ratio of O/H for the N2S2 metallicity indicator vs. Stellar Mass Surface Density
    plt.subplot(3,2,5)
    plt.scatter(mass, n2s2_met, s=4, color = 'r')
    plt.plot(m, rp_n2s2, color='k') # plots midpoints
    plt.xlabel('$\Sigma_{M_{*}} (log(\Sigma_* / M_\odot kpc^2)$')
    plt.ylabel('[NII]/[SII] (12+log(O/H))')
    
    # plots the ratio of O/H for the N2S2 metallicity indicator vs. Stellar Mass Surface Density
    plt.subplot(3,2,6)
    plt.scatter(mass, n2o2_met, s=4, color = 'r')
    plt.plot(m, rp_n2o2, color='k') # plots midpoints
    plt.xlabel('$\Sigma_{M_{*}} (log(\Sigma_* / M_\odot kpc^2)$')
    plt.ylabel('[NII]/[OII] (12+log(O/H))')
    
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.97, top=0.93, wspace=0.17, hspace=0)
    plt.suptitle(plateifu)
    plt.suptitle('Radial Metallicity', fontsize = 'xx-large')
    
    #plt.savefig('/Users/jalynkrause/Documents/astro/AAS/' + plateifu + '.png')
    #plt.savefig('/Users/jalynkrause/Documents/astro/symposium_2019/nonlinear_gradient_example.png')
    
    plt.show()
    plt.close()

def reject_invalid(variables, bad_flag = None):
    '''
    FROM ADAM: This function takes in a list of variables stored in numpy arrays and returns these arrays where there are no nans.
    variables=[variable1,variable2,variable3...]
    bad_flag=a value that denotes bad data e.g. -999
    '''

    if type(variables) != list:
        print("please input a list of numpy arrays")
        return
    good = np.ones(variables[0].shape)
    for var in variables:
        if type(var[0])!=str and type(var[0])!= np.str_:
            bad = np.where((np.isnan(var)==True) | (np.isinf(var)==True) | (var==bad_flag))
            good[bad] = 0
    var_out = []
    for var in variables:
        var_out.append(var[good == 1])
    return var_out

# Plots the radial profile of O/H (logOH+12) 
def oh_map(plateifu):
    hdulist = fits.open('/Users/jalynkrause/Documents/goodmaps/manga-' + plateifu + '-MAPS-HYB10-GAU-MILESHC.fits.gz')
    
    fluxes = hdulist['EMLINE_GFLUX'].data
    errs= (hdulist['EMLINE_GFLUX_IVAR'].data)**-0.5
    Ha = hdulist['EMLINE_GFLUX'].data[18,...]
    H_alpha = fluxes[18,:,:]
    Ha = H_alpha
    Ha_err = errs[18,:,:]
    NII = fluxes[19,:,:]
    n2_err = errs[19,:,:]
    s21 = fluxes[20,:,:]
    s22 = fluxes[21,:,:]
    s21_err = errs[20,:,:]
    s22_err = errs[21,:,:]
    
    logOH12, logOH12error = n2s2_dopita16_w_errs(H_alpha, NII, s21, s22, Ha_err, n2_err, s21_err, s22_err)
    
    # Our maximum error is decided to be the 95th percentile of the logOH12 errors. We authomatically set it to 0.1
    max_err = np.nanpercentile(logOH12error.ravel(), 95)
    max_err = 0.1
    
    minimum = np.nanpercentile(logOH12, 5)
    maximum = np.nanpercentile(logOH12, 95)
    median = np.nanpercentile(logOH12, 50)
    
    shape = (logOH12.shape[1])
    shapemap = [-.25*shape, .25*shape, -.25*shape, .25*shape]
    
    logOH12[np.isinf(logOH12)==True]=np.nan
    badpix = (logOH12error > max_err) | ((Ha/Ha_err) < 3) | ((s22/s22_err) < 3) | ((s21/s21_err) < 3) | ((NII/n2_err) < 3)
    logOH12[badpix] = np.nan
    
    plt.imshow(logOH12, cmap = "viridis", extent = shapemap, vmin = minimum, vmax = maximum, zorder = 1)
    plt.title("Metallicity Map")
    plt.gca().invert_yaxis() 
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    cb = plt.colorbar(shrink = .7)
    cb.set_label('12+log(O/H)', rotation = 270, labelpad = 25)
    plt.show()
    plt.close()
    
 #  From Celeste github senior/annotated_six_plots_creator.py
def n2s2_dopita16_w_errs(ha,n22,s21,s22,ha_err,n22_err,s21_err,s22_err):
    '''
    N2S2 metallicity diagnostic from Dopita et al. (2016)
    includes a calculation of the errors
    '''
    arb = n22/(s21+s22)
    arb_2 = n22/ha
    y = np.log10(arb) + 0.264 * np.log10(arb_2)
    s2 = s21 + s22
    s2_err = np.sqrt(s21_err**2 + s22_err**2)
    met_err = (1.0/np.log(10)) * np.sqrt((1+0.264**2)*(n22_err/n22)**2 + (s2_err/s2)**2 + (ha_err/ha)**2)
    met = 8.77 + y
    return met, met_err

# Method to get plateifu for given galaxy 
def get_plateifu():
    plateifu = '7958-3702'
    return plateifu

# Used for contraining parameters in gradient plots
def model_residual(params, x, data):
    model = 12 - 0.1 * x
    residuals = data - model
    return residuals

 # Main method for comparing metalicity indicators on same plot
def main_1():
    #fig = plt.figure(figsize = (20,9), facecolor = 'white')
    #distorted_OH, slope_change = get_data()
    #plot(distorted_OH, slope_change)
    
    '''
    # Arrays containing plateifu as string of galaxy that has certain feature
    
        bars_arr = np.array(['7958-3702', '8318-12703', '8330-12703', '8553-9102', '8931-12702'])
    
        low_surface_brightness_arr = np.array(['7495-3701', '8134-12705', '8155-6104', '8320-6104', '8442-12703', '8459-1902', '8459-6102', '8550-6103',    '8727-3701', '8941-12705', '9033-1901', '9033-9102', '9487-3704'])
    
        bad_inclination_arr = np.array(['8145-12703', '8440-1902', '8712-12702', '9194-12701'])
    
        small_ifu_arr = np.array(['7992-3702', '8241-6101', '8252-9102', '8255-9102', '8258-3704', '8450-12701', '8453-6102', '8549-6103', '8551-12702',        '8604-9102', '8615-12701', '8716-12703', '8716-12705', '8934-3701', '8936-12704', '8942-3704', '9088-6101', '9487-12703'])
    '''
    
    plateifu = get_plateifu()
    re = get_reff(plateifu) # Calls method to get effective radius of given galaxy
    cat = 'bars' # Category we are looking at
    met_n2o2, met_err_n2o2, rad_n2o2, rad_n2o2_pix = get_map_n2o2(plateifu, cat, re)
    met_n2s2, met_err_n2s2, rad_n2s2, rad_n2s2_pix = get_map_n2s2(plateifu, cat, re)
    
    r_n2o2, rp_n2o2, nums_n2o2, sig_n2o2 = rp_for_fit_2(met_n2o2, distarr=rad_n2o2_pix, radtype='weighted') 
    r_n2s2, rp_n2s2, nums_n2s2, sig_n2s2 = rp_for_fit_2(met_n2s2, distarr=rad_n2s2_pix, radtype='weighted')
    
    plot_map(met_n2o2, met_err_n2o2, rad_n2o2, rp_n2o2, met_n2s2, met_err_n2s2, rad_n2s2, rp_n2s2, plateifu, re, cat)
    
# Main method for generating 6-panel plots and add on to fit abnormal met gradients
def main_2():
  
    # Remaining 46 galaxies
    '''
    7495-3701, 7495-12704, 7957-9101, 7958-3702, 7990-3703, 8077-6104, 8137-12703, 8144-6101, 8145-6104, 8150-6101
    8243-9101, 8252-9102, 8257-6101, 8259-6101, 8329-12701, 8338-6102, 8338-12701, 8455-6101, 8458-3702, 8459-12705
    8485-6101, 8548-3704, 8549-6103, 8592-9101, 8592-12701, 8595-12703, 8712-12702, 8718-12703, 8931-12701, 8931-12702
    8944-12703, 8952-6104, 8979-12701, 8984-12702, 8990-9102, 8991-12701, 8997-9102, 9029-6102, 9042-9102, 9194-12701
    9487-12703, 9490-12702, 9502-12702, 9512-12704, 9868-9102, 10001-9101
   
    plateifu = get_plateifu()
    plate = '7958' 
    ifu = '3702'
   
    hdu = fits.open('/Users/jalynkrause/Documents/astro/SFRD_and_SFRT/' + plateifu + '_SFRD.fits')
    p3ddir = '/Users/jalynkrause/Documents/astro/pipe3d_maps/' 
    drpall = '/Users/jalynkrause/Documents/astro/drpall-v2_4_3.fits'
  
    mass, mass_err = get_massmap(plateifu, p3ddir, drpall)
    reff = get_reff(plateifu) # gets the effective radius for a specific galaxy

    plateifu = hdu[1].data['plateifu'][0]
    sfrd = hdu[1].data['sfrd']
    sfrt = hdu[1].data['sfrt']
    
    six_pann = six_pannel(plateifu, plate, ifu, mass, mass_err, reff)
    '''
    
    ##############################################################################################################
    ######################       --ADD ON-- FITS ABNORMAL O/H GRADIENTS          #################################
    ##############################################################################################################
    
    t = Table() # Creates table to fill values later 
    # Holds PlateIFU
    name = []
    mass_arr = []
    sfrd_arr = []
    rad_arr = []

    # Holds calculated values of slope for fitted line
    slope_n2o2 = []
    slope_n2s2 = []
    slope_pp04 = []

    # Holds ratio of fitted segments
    r01_n2o2_arr = []
    r10_n2o2_arr = []
    r12_n2o2_arr = []
    r21_n2o2_arr = []

    r01_n2s2_arr = []
    r10_n2s2_arr = []
    r12_n2s2_arr = []
    r21_n2s2_arr = []

    r01_pp04_arr = []
    r10_pp04_arr = []
    r12_pp04_arr = []
    r21_pp04_arr = []
    
    # Holds derivative arrays for poly fit function
    der_n2o2_arr = []
    der_n2s2_arr = []
    der_pp04_arr = []

    # Holds calculated 30% range for categorized slope change
    rang_n2o2_arr = [] 
    rang_n2s2_arr = [] 
    rang_pp04_arr = [] 

    # Holds calculated standard deviation of points from fitted lines
    # Measures the amount of variability, or dispersion, for a subject set of data from the mean
    fit_stdev_n2o2_arr = []
    fit_stdev_n2s2_arr = []
    fit_stdev_pp04_arr = []
    
    # Holds calculatedstandard error of the mean points from the fitted lines
    # Measures how far the sample mean of the data is likely to be from the true population mean
    fit_stderr_n2o2_arr = [] 
    fit_stderr_n2s2_arr = [] 
    fit_stderr_pp04_arr = [] 
    
    # Empty array to fill with 0 or 1 if there is a slope change for that galaxy
    slopechange_n2o2 = []
    slopechange_n2s2 = [] 
    slopechange_pp04 = [] 
    
    gal_list = np.genfromtxt('/Users/jalynkrause/Documents/astro/Good_Galaxies_SPX_3_N2S2.txt',usecols=(0),skip_header=1,dtype='str',delimiter=',')
  
    for plateifu in gal_list:
        plate, ifu = plateifu.split('-') # splits plateifu numbers into two variables with the numbers before and after '-'
        # Indent try loop after this to iterate through all galaxies
        '''
        # Bad galaxy test
        plateifu = '8139-3702'
        plate = '8139'
        ifu = '3702'

        # Good galaxy tes
        plateifu = '8137-12703'
        plate = '8137'
        ifu= '12703'
        '''

        try:
            hdu = fits.open('/Users/jalynkrause/Documents/astro/SFRD_and_SFRT/' + plateifu + '_SFRD.fits')
            p3ddir = '/Users/jalynkrause/Documents/astro/pipe3d_maps_all/' 
            drpall = '/Users/jalynkrause/Documents/astro/drpall-v2_4_3.fits'
            mass, mass_err = get_massmap(plateifu, p3ddir, drpall)
            reff = get_reff(plateifu) # gets the effective radius for a specific galaxy
            plateifu = hdu[1].data['plateifu'][0]
            sfrd = hdu[1].data['sfrd']
            sfrt = hdu[1].data['sfrt']

            rad = get_radius(plate, ifu) # Elliptical radius in units of arcseconds
            rad = rad/reff # Radial measurements now in units of Re, was in arcseconds

            # SFR surface density in units of M_sun/yr/kpc^2. It assumes a flat lambdaCDM cosmology with h=0.7, Omega_lambda=0.7, Omega_matter=0.3. SFR surface density is corrected for inclination using the Petrossian b/a values, so they are probably not accurate for highly inclined systems. 
            sfrd = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/sfr_map.npy')

            sfrd_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/sfr_map_err.npy') # Error on the SFRD
            # Elliptical radius in units of arcseconds

            issf = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/issf.npy') # 1 if a spaxel is considered to be in the star-forming part of the [OIII]/Hb, [NII/Ha] BPT diagram (passes both Kauffmann and Kewley criteria), 0 if not
            ha_ew = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/ha_ew.npy')

            n2s2_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_met.npy') # Dopita N2S2 Halpha metallcity

            n2s2_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_met_err.npy') # Error on Dopita 2016 metallicity, calculated by standard error propagation
            n2s2_mask = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2s2_mask.npy') # Mask for N2S2 metallcity. 1 if S/N of all lines used >3

            n2o2_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met.npy') # KD04 [NII]/[OII] metallicity (12+log(O/H)). Not reliable for 12+log(O/H)<8.6

            n2o2_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_met_err.npy') # Error on the N2O2 metallcity. This is derived by a Monte Carlo method.

            n2o2_mask = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/n2o2_mask.npy') # 1 if ([NII]6583/[NII]6583_err > 3 ) & ([OII]3727/[OII]3727_err > 3) & ([OII]3729/[OII]3729_err > 3 ) & (ha/ha_err>3) & (hb/hb_err > 3), 0 otherwise.

            pp04_met = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/pp04_o3n2_met.npy')

            pp04_met_err = np.load('/Users/jalynkrause/Documents/astro/metallicities/' + plate + '/' + ifu + '/pp04_o3n2_met_err.npy')

            # Masking bad data 
            n2o2_bad = (n2o2_mask == 0) | (issf == 0) | (n2o2_met_err > 0.1) | (ha_ew < 6)
            n2o2_met[n2o2_bad] = np.nan
            n2o2_met_err[n2o2_bad] = np.nan
            n2o2_met[n2o2_met < 8.6] = np.nan # Metallicities below 8.6 are not reliable with this indicator
            n2o2_good = np.where(np.isnan(n2o2_met) == False)[0]

            n2s2_bad = (n2s2_mask == 0) | (issf == 0) | (n2s2_met_err > 0.1) | (ha_ew < 6)
            n2s2_met_err[n2s2_bad] =np.nan
            n2s2_met[n2s2_bad] = np.nan
            n2s2_good = np.where(np.isnan(n2s2_met) == False)[0]

            pp04_bad = (issf == 0) | (pp04_met_err > 0.1) | (ha_ew < 6)
            pp04_met_err[pp04_bad] =np.nan
            pp04_met[pp04_bad] = np.nan
            pp04_good = np.where(np.isnan(pp04_met) == False)[0]

            bad_mass = mass < 3
            mass[bad_mass] = np.nan

            sfrd_bad = (issf == 0)
            sfrd[sfrd_bad] = np.nan

            # Requires a galaxy to have an effective radius greater than 1.5, and more than 100 metallicity data points
            if (np.max(rad) >= 1.5) & ((len(n2o2_good) >= 100) & (len(n2s2_good) >= 100) & (len(pp04_good) >= 100)):

                rmax = int(np.max(rad))
                r_edge = np.linspace(0,rmax,len(rad))
                r = np.zeros(len(r_edge) -1) * np.nan # radius
                s = np.zeros(len(r_edge) -1) * np.nan # SFRD
                m = np.zeros(len(r_edge) -1) * np.nan # stellar mass surface density
                rp_n2s2 = np.zeros(len(r_edge) -1) * np.nan # rp = the average value of metallicity at that radius
                rp_n2o2 = np.zeros(len(r_edge) -1) * np.nan
                rp_pp04 = np.zeros(len(r_edge)-1) * np.nan

                # midpoints
                for i in range(0, len(rp_n2s2)-1):
                    rp_n2s2[i] = np.nanmean(n2s2_met[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
                    r[i] = np.nanmean(rad[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
                    s[i] = np.nanmean(sfrd[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
                    m[i] = np.nanmean(mass[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])

                for j in range(0, len(rp_n2o2)-1):
                    rp_n2o2[j] = np.nanmean(n2o2_met[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])

                for k in range(0, len(rp_pp04)-1):
                    rp_pp04[k] = np.nanmean(pp04_met[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])   

                    
                ####################################################################################################################  
                ######### The section of the function below fits 4 evenly spaced points between .5-2 Re     ########################
                ######### to the average metallicity measurements at that interval.                         ########################
                ####################################################################################################################  
                '''

                x = r # in units of radius / effective radius

                y_n2o2 = np.zeros(len(x)) * np.nan # y-values for each indicator
                y_n2s2 = np.zeros(len(x)) * np.nan
                y_pp04 = np.zeros(len(x)) * np.nan

                for i in range(0, len(y_n2o2)):
                    y_n2o2[i] = np.nanmean(n2o2_met[np.where((rad>=r_edge[i]) & (rad<r_edge[i+1]))])
                for j in range(0, len(y_n2s2)):    
                    y_n2s2[j] = np.nanmean(n2s2_met[np.where((rad>=r_edge[j]) & (rad<r_edge[j+1]))])
                for k in range(0, len(y_pp04)):     
                    y_pp04[k] = np.nanmean(pp04_met[np.where((rad>=r_edge[k]) & (rad<r_edge[k+1]))])

                x, y_n2o2, y_n2s2, y_pp04 = reject_invalid([x.ravel(),y_n2o2.ravel(),y_n2s2.ravel(),y_pp04.ravel()]) # returns these arrays where there are no nans

                low = 0.5 # lower bound on xvals
                high = 2.0 # upper bounds on xvals

                xvals = np.linspace(low, high, 4) # 4 points creates 3 lines to evaluate slope between 0.5 and 2 Re
            
                #### N2O2 ##################################################################
                #  Performs least squares polynomial fit on 3nd degree poly
                poly_n2o2 = np.polyfit(x, y_n2o2, 3) # Returns polynomial coefficients, highest power first
                p_eqn_n2o2 = np.poly1d(poly_n2o2) # Construct the polynomial into an equation ax^3+bx^2+cx+d
                xmodel = np.linspace(low, high, len(y_n2o2))
                ymodel_n2o2 = np.polyval(poly_n2o2, xmodel) # Evaluates polynomial at specific values
                der_n2o2 = np.polyder(poly_n2o2, 1) # Returns poly coeff. of first order deriv. of function
                roots_n2o2 = np.roots(der_n2o2) # Critical points of function
                der_ymodel_n2o2 = np.polyval(der_n2o2, xmodel) # Evaluates polynomial at specific values
                realroots_n2o2 = roots_n2o2[np.isreal(roots_n2o2)] # Roots of function
                
                # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
                yinterp_n2o2 = np.interp(xmodel, x, y_n2o2) # Returns the one-dimensional piecewise linear interpolant to a function
                
                m2_n2o2 = np.diff(yinterp_n2o2)/np.diff(xmodel) # Calculates slope of yinterp and xvals
                slope_diff_n2o2 = np.diff(m2_n2o2) # Calculates difference in slopes
                fit_stdev_n2o2 = np.std(y_n2o2, axis=0) # Calcs standard deviation of metallicity values along fitted line
                fit_stderr_n2o2 = stats.sem(y_n2o2, axis=0) # Calcs median error of met. values along fitted ling

                # Calculates ratios of fitted slopes
                r01_n2o2 = np.absolute(np.divide(m2_n2o2[0], m2_n2o2[1])) # inverse of r01 (r01 = ratio index 0 / index 1)
                r10_n2o2 = np.absolute(np.divide(m2_n2o2[1], m2_n2o2[0])) # inverse of r10 (to appply to various slope trends)
                r12_n2o2 = np.absolute(np.divide(m2_n2o2[1], m2_n2o2[2]))
                r21_n2o2 = np.absolute(np.divide(m2_n2o2[2], m2_n2o2[1]))

                rang_n2o2 = m2_n2o2 * .30 # range of classification of slope ratio (30%)
                ch_add_n2o2 = m2_n2o2 + rang_n2o2 # adds calculated 30% range to slope measurement to create bounds
                ch_sub_n2o2 = m2_n2o2 - rang_n2o2 # only calculates for last 2 segments if [:1] is placed at in to index array 

                # Will print out statements if galaxy's oxygen abundance gradient has a slope ratio greater than 30%
                if ((.3 <= r01_n2o2 < 1.) | (.3 <= r10_n2o2 < 1.) | (.3 <= r12_n2o2 < 1.) | (.3 <= r21_n2o2 < 1.)):
                    slopechange_n2o2.append([1]) # 1 for true (true there is a change in slope)
                    print('Slope Change in N2O2; ', plateifu)

                else:
                    slopechange_n2o2.append([0]) # 0 for false
                    print('No Significant Change in Slope in N2O2; ', plateifu)
                ############################################################################
                ############################################################################

                #### N2S2 ###################################################################
                #  Performs least squares polynomial fit on 3nd degree poly
                poly_n2s2 = np.polyfit(x, y_n2s2, 3) # Returns polynomial coefficients, highest power first
                p_eqn_n2s2 = np.poly1d(poly_n2s2) # Construct the polynomial into an equation ax^3+bx^2+cx+d
                ymodel_n2s2 = np.polyval(poly_n2s2, xmodel) # Evaluates polynomial at specific values
                der_n2s2 = np.polyder(poly_n2s2, 1) # Returns poly coeff. of first order deriv. of function
                roots_n2s2 = np.roots(der_n2s2) # Critical points of function
                der_ymodel_n2s2 = np.polyval(der_n2s2, xmodel)
                realroots_n2s2 = roots_n2s2[np.isreal(roots_n2s2)] # Roots of function
                
                # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
                yinterp_n2s2 = np.interp(xmodel, x, y_n2s2) # Returns the one-dimensional piecewise linear interpolant to a function
                
                m2_n2s2 = np.diff(yinterp_n2s2)/np.diff(xmodel) # Calculates slope of yinterp and xvals
                slope_diff_n2s2 = np.diff(m2_n2s2) # Calculates difference in slopes
                fit_stdev_n2s2 = np.std(y_n2s2, axis=0) # Calcs standard deviation of metallicity values along fitted line
                fit_stderr_n2s2 = stats.sem(y_n2s2, axis=0) # Calcs median error of met. values along fitted ling

                # Calculates ratios of fitted slopes
                r01_n2s2 = np.absolute(np.divide(m2_n2s2[0], m2_n2s2[1])) # inverse of r01 (r01 = ratio index 0 / index 1)
                r10_n2s2 = np.absolute(np.divide(m2_n2s2[1], m2_n2s2[0])) # inverse of r10 (to appply to various slope trends)
                r12_n2s2 = np.absolute(np.divide(m2_n2s2[1], m2_n2s2[2]))
                r21_n2s2 = np.absolute(np.divide(m2_n2s2[2], m2_n2s2[1]))

                rang_n2s2 = m2_n2s2*.30 # range of classification of slope ratio (30%)
                ch_add_n2s2 = m2_n2s2+rang_n2s2 # adds calculated 30% range to slope measurement to create bounds
                ch_sub_n2s2 = m2_n2s2-rang_n2s2 # only calculates for last 2 segments if [:1] is placed at in to index array

                # Will print out statements if galaxy's oxygen abundance gradient has a slope ratio greater than 30%
                if ((.3 <= r01_n2s2 < 1.) | (.3 <= r10_n2s2 < 1.) | (.3 <= r12_n2s2 < 1.) | (.3 <= r21_n2s2 < 1.)):
                    print('Slope Change in N2S2; ', plateifu)
                    slopechange_n2s2.append([1]) # 1 for true (true there is a change in slope)

                else:
                    print('No Significant Change in Slope in N2S2; ', plateifu)
                    slopechange_n2s2.append([0]) # 0 for false
                ##############################################################################
                ##############################################################################

                #### PPO4 ###################################################################
                #  Performs least squares polynomial fit on 3nd degree poly
                poly_pp04 = np.polyfit(x, y_pp04, 3) # Returns polynomial coefficients, highest power first [a,b,c,d]
                p_eqn_pp04 = np.poly1d(poly_pp04) # Construct the polynomial into an equation ax^3+bx^2+cx+d
                ymodel_pp04 = np.polyval(poly_pp04, xmodel) # Evaluates polynomial at specific values
                
                der_pp04 = np.polyder(poly_pp04, 1) # Returns poly coeff. of first order deriv. of function
                pder_eqn_pp04 = np.poly1d(der_pp04)  # Construct the polynomial into an equation ax^2+bx+c
                roots_pp04 = np.roots(der_pp04) # Critical points of function
                der_ymodel_pp04 = np.polyval(der_pp04, xmodel) # Evaluates polynomial at specific values
                realroots_pp04 = roots_pp04[np.isreal(roots_pp04) & (roots_pp04 >= 0.5) & (roots_pp04 <= 2.0)] # Roots of function, radii where max slope change
                print(plateifu, ':', realroots_pp04)
                
                # Second Deriv. [NOT NEEDED]
                secder_pp04 = np.polyder(poly_pp04, 2) # Returns poly coeff. of second order deriv. of function
                psecder_eqn_pp04 = np.poly1d(secder_pp04) # in equn. form ax+b
                secder_ymodel_pp04 = np.polyval(secder_pp04, xmodel) # Evaluates polynomial at specific values
            
                [NOT NEEDED]
                print(poly_pp04)
                print(p_eqn_pp04)
                print(ymodel_pp04)
                print(der_pp04)
                print(pder_eqn_pp04)
                print(roots_pp04)
                print(der_ymodel_pp04)
                print(realroots_pp04)
              
    
                if realroots_pp04.size > 0:
                   
                    if realroots_pp04.size == 1:    
                        # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
                        yinterp_pp04 = np.interp([0.5, realroots_pp04[0], 2], xmodel, ymodel_pp04) # Returns the one-dimensional piecewise linear interpolant to a function

                    if realroots_pp04.size == 2:
                        # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
                        yinterp_pp04 = np.interp([0.5, realroots_pp04.min(), realroots_pp04.max(), 2], xmodel, ymodel_pp04) # Returns the one-dimensional piecewise linear interpolant to a function

                    m2_pp04 = np.diff(ymodel_pp04)/np.diff(xmodel) # Calculates slope of yinterp and xvals
                    slope_diff_pp04 = np.diff(m2_pp04) # Calculates difference in slopes
                    fit_stdev_pp04 = np.std(y_pp04, axis=0) # Calcs standard deviation of metallicity values along fitted line
                    fit_stderr_pp04 = stats.sem(y_pp04, axis=0) # Calcs median error of met. values along fitted ling

                    # Calculates ratios of fitted slopes
                    r01_pp04 = np.absolute(np.divide(m2_pp04[0], m2_pp04[1])) # inverse of r01 (r01 = ratio index 0 / index 1)
                    r10_pp04 = np.absolute(np.divide(m2_pp04[1], m2_pp04[0])) # inverse of r10 (to appply to various slope trends)
                    r12_pp04 = np.absolute(np.divide(m2_pp04[1], m2_pp04[2]))
                    r21_pp04 = np.absolute(np.divide(m2_pp04[2], m2_pp04[1]))

                    rang_pp04 = m2_pp04*.30 # range of classification of slope ratio (30%)
                    ch_add_pp04 = m2_pp04+rang_pp04 # adds calculated 30% range to slope measurement to create bounds
                    ch_sub_pp04 = m2_pp04-rang_pp04 # only calculates for last 2 segments if [:1] is placed at in to index array

                    # Will print out statements if galaxy's oxygen abundance gradient has a slope ratio greater than 30%
                    if ((.3 <= r01_pp04 < 1.) | (.3 <= r10_pp04 < 1.) | (.3 <= r12_pp04 < 1.) | (.3 <= r21_pp04 < 1.)):
                        print('Slope Change in PP04; ', plateifu)
                        slopechange_pp04.append([1]) # 1 for true (true there is a change in slope)

                    else:
                        print('No Significant Change in Slope in PP04; ', plateifu)
                        slopechange_pp04.append([0]) # 0 for false           
                else:
                    print('No Significant Change in Slope in PP04; ', plateifu)
                    slopechange_pp04.append([0]) # 0 for false 
                 
                '''
                ##############################################################################
                ############### ^ END OF THIS SECTION ^ ##########################################
                ##############################################################################
                
                
                #################  CONSTRAINING MODEL USING LMFIT FROM ADAM ##################
                
                # 'vary = False' fixes the parameter.
                # Setting min = 1 and max = 2 will allow that parameter to vary between 1 and 2.
                
                params = Parameters()
                params.add('Z0', value=0.003, min=0.0001, max=0.03)
                params.add('alpha', value=0.727, vary=True, min=0.1, max=2)
                params.add('lam0', value=0.27, vary=True, min=0, max=0.5)
                params.add('v0', value=545.0, vary=True, min=1, max=900)
                params.add('y', value=0.014, vary=False)
                params.add('R', value=0.4, vary=False)
                params.add('epsdrdt', value=-0.25, vary=False)
                
               

                # Perform the minimization. Default is leastsq.
                # 'out' is an object that stores all of the fitted model parameters and other stuff. So to see what Z0 is: out.params['Z0']
                #out = minimize(gas_regulator_model_residual, params, args=[data_a])

                ###############################################################################
                
                
                # PLOTS FIGURE
                fig = plt.figure(figsize = (11,8), facecolor = 'white') # figsize = (width, height) in inches
                fig.subplots(2,1,sharex='col')
                fig.suptitle(plateifu, fontsize = 'xx-large')
                
                '''
                # N2O2
                plt.subplot(3,1,1)
                plt.errorbar(rad.ravel(), n2o2_met.ravel(),yerr=n2o2_met_err.ravel(),fmt='o',ecolor='aqua', mfc='navy',mec='navy',ms=1, label='N2O2', zorder=1) # plots metallicity points with error bars
                plt.plot(xmodel, yinterp_n2o2, '-*', c='red', lw=2, zorder=2, ms=3, label='Interp.') # plots interp points
                #plt.scatter(roots_n2o2, roots_n2o2(ymodel_n2o2)) # Plots critical points
                #plt.plot(xmodel, ymodel_n2o2, c='blue', zorder=3, label='Poly Fit') # plots polyfit function
                plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
                plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
                plt.xlabel('R/$R_e$')
                plt.ylabel('12+log(O/H)')
                plt.legend(loc='best')
                
                #N2S2
                plt.subplot(3,1,2)
                plt.errorbar(rad.ravel(), n2s2_met.ravel(),yerr=n2s2_met_err.ravel(),fmt='o',ecolor='lime', mfc='green',mec='green',ms=1, label='N2S2', zorder=1) # plots metallicity points with error bars
                plt.plot(xmodel, yinterp_n2s2, '-*', c='red', lw=2, zorder=2, ms=3, label='Interp.') # plots points found for calculating the slope change
                #plt.plot(xmodel, ymodel_n2s2, c='blue', zorder=3, label='Poly Fit')
                plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
                plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
                plt.xlabel('R/$R_e$')
                plt.ylabel('12+log(O/H)')
                plt.legend(loc='best')
                '''
                #PP04
                plt.subplot(2,1,1)
                plt.errorbar(rad.ravel(), pp04_met.ravel(),yerr=pp04_met_err.ravel(),fmt='o',ecolor='plum', mfc='indigo',mec='indigo',ms=2, label='PP04', zorder=1) # plots metallicity points with error bars
                
                #if realroots_pp04.size == 0:
                #plt.plot([0.5, x[np.max(np.diff(der_ymodel_pp04))], 2], [ymodel_pp04[0.5], ymodel[np.max(np.diff(der_ymodel_pp04))], ymodel_pp04[2.0]])
                
                if realroots_pp04.size == 1:
                    plt.plot([0.5, realroots_pp04[0], 2], yinterp_pp04, '-*', c='red', lw=3, zorder=10, ms=10, label='Interp.') # plots points found for calculating the slope change
                
                if realroots_pp04.size == 2:
                    plt.plot([0.5, realroots_pp04.min(), realroots_pp04.max(), 2], yinterp_pp04, '-*', c='red', lw=3, zorder=10, ms=10, label='Interp.') # plots points found for calculating the slope change
                
                plt.plot(xmodel, ymodel_pp04, c='green', lw=3, zorder=3, label='Poly Fit')
                plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
                plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
                plt.xlabel('R/$R_e$')
                plt.ylabel('12+log(O/H)')
                plt.legend(loc='best')
                #plt.title(plateifu)
                
                #PP04 1st Deriv
                plt.subplot(2,1,2)
                plt.plot(xmodel, der_ymodel_pp04, label = realroots_pp04)
                
                if realroots_pp04.size == 1:
                    plt.scatter([0.5, realroots_pp04[0], 2], np.polyval(der_pp04, [0.5, realroots_pp04[0], 2]))
                    
                if realroots_pp04.size == 2:
                    plt.scatter([0.5, realroots_pp04[0], realroots_pp04[1], 2], np.polyval(der_pp04, [0.5, realroots_pp04[0], realroots_pp04[1], 2]))    
                    
                plt.xlabel('R/$R_e$')
                plt.ylabel('Derivative of Poly')
                plt.legend(loc='best')
            
                plt.savefig('/Users/jalynkrause/Documents/astro/grad_ratio_1/plots0920/' + plateifu + '.png')
                #plt.show()
                plt.close('all')
                
                ##############################################################################
                ############### Writes data to fits file #####################################
                ##############################################################################
                
                name.append([plateifu]) # numpy.append() puts next plateIFU at END of array

                ### CHECK THESE ARE CALCULATED RIGHT ###
                mass_arr.append([np.nansum(mass)])
                rad_arr.append([np.max(rad)])
                sfrd_arr.append([np.nansum(sfrd)])

                # Will add array of values to end of current array for given plate ifu. If no values, adds nans in place.  
                # N2O2
                slope_n2o2.append(m2_n2o2)
                rang_n2o2_arr.append(rang_n2o2)

                r01_n2o2_arr.append([r01_n2o2])
                r10_n2o2_arr.append([r10_n2o2])
                r12_n2o2_arr.append([r12_n2o2])
                r21_n2o2_arr.append([r21_n2o2])

                der_n2o2_arr.append([der_n2o2])
                
                fit_stdev_n2o2_arr.append([fit_stdev_n2o2])
                fit_stderr_n2o2_arr.append([fit_stderr_n2o2])
   
                # N2S2
                slope_n2s2.append(m2_n2s2)
                rang_n2s2_arr.append(rang_n2s2)

                r01_n2s2_arr.append([r01_n2s2])
                r10_n2s2_arr.append([r10_n2s2])
                r12_n2s2_arr.append([r12_n2s2])
                r21_n2s2_arr.append([r21_n2s2])

                der_n2s2_arr.append([der_n2s2])
                
                fit_stdev_n2s2_arr.append([fit_stdev_n2s2])
                fit_stderr_n2s2_arr.append([fit_stderr_n2s2])
    
                # PP04
                slope_pp04.append(m2_pp04)
                rang_pp04_arr.append(rang_pp04)

                r01_pp04_arr.append([r01_pp04])
                r10_pp04_arr.append([r10_pp04])
                r12_pp04_arr.append([r12_pp04])
                r21_pp04_arr.append([r21_pp04])

                der_pp04_arr.append([der_pp04])
                
                fit_stdev_pp04_arr.append([fit_stdev_pp04])
                fit_stderr_pp04_arr.append([fit_stderr_pp04])
              
            # Requires a galaxy to have an R_eff > 1.5, and more than 100 metallicity data points, if doesn't follows loop:                                       
            else: 
                print('FLAGGED! Insufficient Data; Galaxy PlateIFU: ' + plateifu)

                name.append([plateifu]) # numpy.append() puts next plateIFU at END of array

                mass_arr.append([-999])
                rad_arr.append([-999])
                sfrd_arr.append([-999])

                # N2O2
                slope_n2o2.append([-999, -999, -999])
                slopechange_n2o2.append([-999])

                rang_n2o2_arr.append([-999, -999,-999])

                r01_n2o2_arr.append([-999])
                r10_n2o2_arr.append([-999])
                r12_n2o2_arr.append([-999])
                r21_n2o2_arr.append([-999])

                der_n2o2_arr.append([-999, -999, -999])
                
                fit_stdev_n2o2_arr.append([-999])
                fit_stderr_n2o2_arr.append([-999])

                # N2S2
                slope_n2s2.append([-999, -999, -999])
                slopechange_n2s2.append([-999])

                rang_n2s2_arr.append([-999, -999, -999])

                r01_n2s2_arr.append([-999])
                r10_n2s2_arr.append([-999])
                r12_n2s2_arr.append([-999])
                r21_n2s2_arr.append([-999])

                der_n2s2_arr.append([-999, -999, -999])
                
                fit_stdev_n2s2_arr.append([-999])
                fit_stderr_n2s2_arr.append([-999])

                # PP04
                slope_pp04.append([-999, -999, -999])
                slopechange_pp04.append([-999])

                rang_pp04_arr.append([-999, -999, -999])

                r01_pp04_arr.append([-999])
                r10_pp04_arr.append([-999])
                r12_pp04_arr.append([-999])
                r21_pp04_arr.append([-999])

                der_pp04_arr.append([-999, -999, -999])
                
                fit_stdev_pp04_arr.append([-999])
                fit_stderr_pp04_arr.append([-999])
        
        except (OSError, IOError):
            print('File Not Found Exception; Galaxy PlateIFU: ' + plateifu)
            print('Error on Line {}'.format(sys.exc_info()[-1].tb_lineno))
    
    try:
        # Defines the columns of table to write all data to
        t['PLATEIFU']=Column(np.vstack(name), description = 'MaNGA PlateIFU')

        t['MASS']=Column(np.vstack(mass_arr), description = 'Total (*)Mass Surface Density')
        t['RAD']=Column(np.vstack(rad_arr), description = 'Max Radii [R/R_e]')
        t['SFRD']=Column(np.vstack(sfrd_arr), description = 'Total SFR Surface Density')

        # N2O2
        t['SLOPE_CH_N2O2']=Column(np.vstack(slopechange_n2o2), description = 'N2O2: 1=Slope Change, 0=No Slope Change')
        t['SLOPE_VAL_N2O2']=Column(np.vstack(slope_n2o2), description = 'N2O2: Calculated Slope Values (line 0, 1, 2)')
        t['30RANGE_N202']=Column(np.vstack(rang_n2o2_arr), description = 'N2O2: 30% of O/H val')

        t['R01_N2O2']=Column(np.vstack(r01_n2o2_arr), description = 'N2O2: Ratio of line 0/line 1')
        t['R10_N2O2']=Column(np.vstack(r10_n2o2_arr), description = 'N2O2: Ratio of line 1/line 0')
        t['R12_N2O2']=Column(np.vstack(r12_n2o2_arr), description = 'N2O2: Ratio of line 1/line 2')
        t['R21_N2O2']=Column(np.vstack(r21_n2o2_arr), description = 'N2O2: Ratio of line 2/line 1')

        t['STDEV_N2O2']=Column(np.vstack(fit_stdev_n2o2_arr), description = 'N2O2: Std. dev. of points from fitted line')
        t['STDERR_N2O2']=Column(np.vstack(fit_stderr_n2o2_arr), description = 'N2O2: Std. err. of points from fitted line')
        
        t['POLYFIT_SlOPES_N2O2']=Column(np.vstack(der_n2o2_arr), description = 'N2O2: Slope vals. using polyfit func.')

        # N2S2
        t['SLOPE_CH_N2S2']=Column(np.vstack(slopechange_n2s2), description = 'N2S2: 1=Slope Change, 0=No Slope Change')
        t['SLOPE_VAL_N2S2']=Column(np.vstack(slope_n2s2), description = 'N2S2: Calculated Slope Values (line 0, 1, 2)')
        t['30RANGE_N2S2']=Column(np.vstack(rang_n2s2_arr), description = 'N2S2: 30% of O/H val')

        t['R01_N2S2']=Column(np.vstack(r01_n2s2_arr), description = 'N2S2: Ratio of line 0/line 1')
        t['R10_N2S2']=Column(np.vstack(r10_n2s2_arr), description = 'N2S2: Ratio of line 1/line 0')
        t['R12_N2S2']=Column(np.vstack(r12_n2s2_arr), description = 'N2S2: Ratio of line 1/line 2')
        t['R21_N2S2']=Column(np.vstack(r21_n2s2_arr), description = 'N2S2: Ratio of line 2/line 1')

        t['STDEV_N2S2']=Column(np.vstack(fit_stdev_n2s2_arr), description = 'N2S2: Std. dev. of points from fitted line')
        t['STDERR_N2S2']=Column(np.vstack(fit_stderr_n2s2_arr), description = 'N2S2: Std. err. of points from fitted line')
        
        t['POLYFIT_SlOPES_N2S2']=Column(np.vstack(der_n2s2_arr), description = 'N2S2: Slope vals. using polyfit func.')

        # PP04
        t['SLOPE_CH_PP04']=Column(np.vstack(slopechange_pp04), description = 'PP04: 1=Slope Change, 0=No Slope Change')
        t['SLOPE_VAL_PP04']=Column(np.vstack(slope_pp04), description = 'PP04: Calculated Slope Values (line 0, 1, 2)')
        t['30RANGE_PP04']=Column(np.vstack(rang_pp04_arr), description = 'PP04: 30% of O/H val')

        t['R01_PP04']=Column(np.vstack(r01_pp04_arr), description = 'PP04: Ratio of line 0/line 1')
        t['R10_PP04']=Column(np.vstack(r10_pp04_arr), description = 'PP04: Ratio of line 1/line 0')
        t['R12_PP04']=Column(np.vstack(r12_pp04_arr), description = 'PP04: Ratio of line 1/line 2')
        t['R21_PP04']=Column(np.vstack(r21_pp04_arr), description = 'PP04: Ratio of line 2/line 1')

        t['STDEV_PP04']=Column(np.vstack(fit_stdev_pp04_arr), description = 'PP04: Std. dev. of points from fitted line')
        t['STDERR_PP04']=Column(np.vstack(fit_stderr_pp04_arr), description = 'PP04: Std. err. of points from fitted line')
        
        t['POLYFIT_SlOPES_PP04']=Column(np.vstack(der_pp04_arr), description = 'PP04: Slope vals. using polyfit func.')

        t.write('/Users/jalynkrause/Documents/astro/grad_ratio_1/output_file.fits', overwrite = True) 

    except (ValueError):
        print('ValueError Raised in Table')
        print('Error on Line {}'.format(sys.exc_info()[-1].tb_lineno))
        plateifu += plateifu

    except (TypeError):
        print('Type Error Raised; Galaxy PlateIFU: ' + plateifu)
        print('Error on Line {}'.format(sys.exc_info()[-1].tb_lineno))
    
 # Main method for generating radial O/H metallicity color map of galaxy for specific ifu                         
def main_3():
    plateifu = get_plateifu()
    plt = oh_map(plateifu)
 
 # Main method for generating fake data to fit a broken power law to radial O/H metallicity profiles
def main_4():
    
    low = 0.5
    high = 2.0
    
    x = np.linspace(0.25, 2.25, 500) # Fake x data with evenly spaced numbers over a specified interval of 0.25 to 2.25 (Re)
    data = 12. - 0.1 * x  # Fake linear metallicity
    res = np.random.normal(0,0.05,500) 
    #data = y+res
    
    '''
    poly = np.polyfit(x, y, 2) # Returns polynomial coefficients, highest power first [a,b,c]
    p_eqn = np.poly1d(poly) # Construct the polynomial into an equation ax^2+bx+c
    xmodel = np.linspace(low, high, len(y))
    ymodel = np.polyval(poly, xmodel) # Evaluates polynomial at specific values

    der = np.polyder(poly, 1) # Returns poly coeff. of first order deriv. of function
    pder_eqn = np.poly1d(der)  # Construct the polynomial into an equation ax+b
    roots = np.roots(der) # Critical points of function
    der_ymodel = np.polyval(der, xmodel) # Evaluates polynomial at specific values
    realroots = roots[np.isreal(roots) & (roots >= 0.5) & (roots <= 2.0)] # Roots of function, radii where max slope change
    print(realroots)
    
     if realroots.size > 0:

        if realroots.size == 1:    
            # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
            yinterp = np.interp([0.5, realroots[0], 2], xmodel, ymodel) # Returns the one-dimensional piecewise linear interpolant to a function

        if realroots.size == 2:
            # Interpolation is a method of constructing new data points within the range of a discrete set of known data points
            yinterp = np.interp([0.5, realroots.min(), realroots.max(), 2], xmodel, ymodel) # Returns the one-dimensional piecewise linear interpolant to a function

    # PLOTS FIGURE
    fig = plt.figure(figsize = (11,8), facecolor = 'white') # figsize = (width, height) in inches
    fig.subplots(2,1,sharex='col')  

    plt.subplot(2,1,1)

    if realroots.size == 1:
        plt.plot([0.5, realroots[0], 2], yinterp, '-*', c='red', lw=3, zorder=10, ms=10, label='Interp.') # plots points found for calculating the slope change

    if realroots.size == 2:
        plt.plot([0.5, realroots.min(), realroots.max(), 2], yinterp, '-*', c='red', lw=3, zorder=10, ms=10, label='Interp.') # plots points found for calculating the slope change

    res = np.random.normal(0, 0.05*x, 500) # Residual now gets larger with radius (0.05 dex), 1000 random points
    plt.scatter(x, y+res, s=3)
    plt.plot(x, y, c='navy')   
    plt.plot(xmodel, ymodel, c='green', lw=3, zorder=3, label='Poly Fit')
    plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
    plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
    plt.title("(Fake Data)")
    plt.xlabel('R/$R_e$')
    plt.ylabel('12+log(O/H)')
    plt.legend(loc='best')

    # 1st Deriv
    plt.subplot(2,1,2)
    plt.plot(xmodel, der_ymodel, label = realroots)

    if realroots.size == 1:
        plt.scatter([0.5, realroots[0], 2], np.polyval(der, [0.5, realroots[0], 2]))

    if realroots.size == 2:
        plt.scatter([0.5, realroots[0], realroots[1], 2], np.polyval(der, [0.5, realroots[0], realroots[1], 2]))    

    plt.xlabel('R/$R_e$')
    plt.ylabel('Derivative of Poly')
    plt.legend(loc='best')
    '''

    #################  
       plt.ylabel('Derivative of Poly')
    plt.legend(loc='best')
    '''

    #################  CONSTRAINING MODEL USING LMFIT FROM ADAM ##################

    # 'vary = False' fixes the parameter.
    # Setting min = 1 and max = 2 will allow that parameter to vary between 1 and 2.

    params = Parameters()
    params.add('x', value=0.25)

    # Perform the minimization. Default is leastsq.
    # 'out' is an object that stores all of the fitted model parameters and other stuff
    # So to see what Z0 is: out.params['Z0']
    out = minimize(model_residual, params, args=[x, data])
    
    # calculate final result
    final = data + out.residual
    

    # PLOTS FIGURE
    fig = plt.figure(figsize = (11,8), facecolor = 'white') # figsize = (width, height) in inches
    
    plt.scatter(x, data+res, s=3)
    plt.plot(x, final, 'r')
    plt.title("(Fake Data)")
    #plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
    #plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
    plt.xlabel('Simulated R/$R_e$')
    plt.ylabel('Simulated Metallicity (y = 12 - 0.1 * x )')

    ###############################################################################

    
    #plt.savefig('/Users/jalynkrause/Documents/astro/grad_ratio_1/plots0920/' + plateifu + '.png')
    plt.show()
    plt.close('all')

#################################################################################################################################    
#################################################################################################################################  
    
#main_1()
#main_2()
#main_3()
main_4()

def test():
    #hdu = fits.open('/Users/jalynkrause/Documents/astro/grad_ratio_1/output_file.fits')
    #data = hdu[1].data
    #print(data[0])
    #print(hdu.info())
    drpall = fits.open('drpall-v2_4_3.fits')
    tbdata = drpall[1].data

    # Print column names
    print(tbdata.columns.names)
    
#test()    
    
################################################################################################################################# 
################################################################################################################################# 
