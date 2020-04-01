# CODE WRITTEN UP UNTIL 4/1/2020 TO FIT ARBITRARY DATA WITH BROKEN GRADIENT USING LMFIT
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
from lmfit import minimize, Minimizer, Parameters, fit_report

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

# Used for contraining parameters in gradient plots using LMFIT
def model_residual(params, data):

    x, y = data
    xvar = params['xvar']
    yvar = params['yvar']
    grad = params['gradient']
    inter = params['intercept']
    min_break = params['min_break']
    model = inter - grad * x
    residuals = y- model

    return residuals
 
 # Main method for generating fake data to fit a broken power law to radial O/H metallicity profiles
def main_4():
    
    low = 0.5
    high = 2.0
    
    xdata = np.linspace(0.25, 3.0, 500) # Fake x data with evenly spaced numbers over a specified interval of 0.25 to 2.25 (Re)
    y = 12. - 1.3 * np.log(xdata) 
    # Fake linear metallicity
    res = np.random.normal(0, 0.30, 500) #  Draw random samples from a normal (Gaussian) distribution.
    ydata = y + res 
    

    #################  CONSTRAINING MODEL USING LMFIT FROM ADAM ##################

    # value=initial value; 
    #... 'vary = Whether the Parameter is varied during a fit (default is True)
    #...setting min = 1 and max = 2 will allow that parameter to vary between the min 1 and the max 2;
    #... expr=Mathematical expression used to constrain the value during the fit.

    data = (xdata, ydata)
    params = Parameters()
    
    # params.add('Z0',value=0.003,min=0.0001,max=0.03)
    # params.add('alpha',value=0.727,vary=True,min=0.1,max=2)
    # params.add('lam0',value=0.27,vary=True,min=0,max=0.5)
    
    params.add('xvar', value=0., vary=True, min=0.5, max=2.0) 
    params.add('yvar', value=12., vary=True, min=12., max=16.)
    params.add('gradient', value=0.1, vary=True, min=-0.5, max=2)
    params.add('intercept', value=12., vary=True, min=10., max=14.)
    params.add('min_break', value=50, vary=False)

    # Perform the minimization. Default is leastsq.
    # 'out' is an object that stores all of the fitted model parameters and other stuff
    # So to see what Z0 is: out.params['Z0']
    out = minimize(model_residual, params, args=[data])
    
    
    print(fit_report(out))
    print(5*'#')
    print( out.params['gradient'])    
    print( out.params['intercept'])    
    # calculate final result
    #final = ydata + out.residual
    
    
    ##################################################################################
    
    # SECTION THAT FINDING MIDPOINTS
    
    xlin = xdata[(xdata>=0.5) & (xdata <=2.0)]
    ylin = ydata[(xdata>=0.5) & (xdata <=2.0)]

    x_edge = np.linspace(0.5,2,10)
    
    ypt = np.zeros(len(x_edge)) * np.nan # y value to be plotted
    xpt = np.zeros(len(x_edge)) * np.nan # x val to be plotted
    
    # Midpoints
    for i in range(0, len(xpt)-1):
        ypt[i] = np.nanmean(ylin[np.where((xlin>=x_edge[i]) & (xlin<=x_edge[i+1]))])
        xpt[i] = np.nanmean(xlin[np.where((xlin>=x_edge[i]) & (xlin<=x_edge[i+1]))])
        
        
    # SECTION THAT PLOTS FIGURE
    
    fig = plt.figure(figsize = (11,8), facecolor = 'white') # figsize = (width, height) in inches
  
    #plt.plot(xdata, final, s=3)
    plt.scatter(xdata, ydata, color='k')
    plt.plot(xpt, ypt, 'r', ) # plots midpoints
    #plt.plot(xdata, final)
    plt.title("(Fake Data)")
    plt.axvline(x = 0.5, color='deeppink', lw=1, zorder=10) # plots vertical line at .5 effective radii
    plt.axvline(x = 2, color='deeppink', lw=1, zorder=10) # plots vertical line at 2 effective radii
    plt.xlabel('Simulated R/$R_e$')
    plt.ylabel('Simulated Metallicity')

    ###############################################################################

    #plt.savefig('/Users/jalynkrause/Documents/astro/grad_ratio_1/plots0920/' + plateifu + '.png')
    plt.show()
    plt.close('all')

#################################################################################################################################    
#################################################################################################################################  
    
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
