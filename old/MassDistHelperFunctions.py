######################################
## Imports
import numpy as np
import h5py as h5
import os

from astropy.table import Table
import astropy.units as u
from astropy import constants as const

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import ticker, cm

from scipy import stats

# Chosen cosmology 
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value

#########################################
# Chirp mass
#########################################
def Mchirp(m1, m2):
    chirp_mass = np.divide(np.power(np.multiply(m1, m2), 3./5.), np.power(np.add(m1, m2), 1./5.))
    return chirp_mass 


#########################################
# Read data
#########################################
def read_data(loc = '', verbose=False):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                  --> [string] Location of data
            rate_key             --> [string] group key name of COMPAS HDF5 data that contains your merger rate
            read_SFRD            --> [bool] If you want to also read in sfr data
            verbose              --> [bool] If you want to print statements while reading in 

        Returns:
            DCO                        --> [astropy table] contains all your double compact object
            DCO_mask                   --> [array of bool] reduces your DCO table to your systems of interest (determined in CI)
            rate_mask                  --> [array of bool] reduces intrinsic_rate_density to systems (flavour) of interest
            redshifts                  --> [array of floats] list of redshifts where you calculated the merger rate
            Average_SF_mass_needed     --> [float]    Msun SF needed to produce the binaries in this simulation
            intrinsic_rate_density     --> [2D array] merger rate in N/Gpc^3/yr
            intrinsic_rate_density_z0  --> [2D array] merger rate in N/Gpc^3/yr at finest/lowest redshift bin calculated

    """
    print('Reading ',loc)
    ################################################
    ## Open hdf5 file
    File        = h5.File(loc ,'r')
    if verbose: print(File.keys(), File[rate_key].keys())
    
    # Older simulations use this naming
    dcokey,  syskey, CEcount, dcomask = 'DoubleCompactObjects', 'SystemParameters', 'CE_Event_Count', 'DCOmask' 
    if dcokey in File.keys():
        if verbose: print('using file with key', dcokey)
    # Newer simulations use this
    else:
        if verbose: print('using file with key', dcokey)
        dcokey,  syskey, CEcount, dcomask = 'BSE_Double_Compact_Objects', 'BSE_System_Parameters', 'CE_Event_Counter', 'DCOmask'
 
    DCO = Table()

    DCO['SEED']                  = File[dcokey]['SEED'][()] 
    DCO[CEcount]                 = File[dcokey][CEcount][()] 
    DCO['Mass(1)']               = File[dcokey]['Mass(1)'][()]
    DCO['Mass(2)']               = File[dcokey]['Mass(2)'][()]
    #DCO['M_tot']                 = DCO['Mass(1)'] + DCO['Mass(2)']
    DCO['M_moreMassive']         = np.maximum(File[dcokey]['Mass(1)'][()], File[dcokey]['Mass(2)'][()])
    #DCO['M_lessMassive']         = np.minimum(File[dcokey]['Mass(1)'][()], File[dcokey]['Mass(2)'][()])
    #DCO['Mchirp']                = Mchirp(DCO['M_moreMassive'], DCO['M_lessMassive'])
    DCO['Stellar_Type(1)']       = File[dcokey]['Stellar_Type(1)'][()]
    DCO['Stellar_Type(2)']       = File[dcokey]['Stellar_Type(2)'][()]
    DCO['Optimistic_CE']         = File[dcokey]['Optimistic_CE'][()]
    DCO['Immediate_RLOF>CE']     = File[dcokey]['Immediate_RLOF>CE'][()]

    SYS_DCO_seeds_bool           = np.in1d(File[syskey]['SEED'][()], DCO['SEED']) #Bool to point SYS to DCO
    # This needs to be done in two steps, otherwise the Snakemake workflow gets stuck for unknown reasons
    SYS_ZAMSType1 = File[syskey]['Stellar_Type@ZAMS(1)'][()]
    SYS_ZAMSType2 = File[syskey]['Stellar_Type@ZAMS(2)'][()]
    DCO['Stellar_Type@ZAMS(1)']  = SYS_ZAMSType1[SYS_DCO_seeds_bool] #File[syskey]['Stellar_Type@ZAMS(1)'][SYS_DCO_seeds_bool]
    DCO['Stellar_Type@ZAMS(2)']  = SYS_ZAMSType2[SYS_DCO_seeds_bool] #File[syskey]['Stellar_Type@ZAMS(2)'][SYS_DCO_seeds_bool]
    
    File.close()
    
    print('Done with reading DCO data for this file :)')
    return DCO




#########################################
# Read data
#########################################
def read_rate_data(loc = '', rate_key = 'Rates_mu00.025_muz-0.05_alpha-1.77_sigma01.125_sigmaz0.05_zBinned', verbose = True):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                  --> [string] Location of data
            rate_key             --> [string] group key name of COMPAS HDF5 data that contains your merger rate

        Returns:
            DCO_mask                   --> [array of bool] reduces your DCO table to your systems of interest (BBH by default)
            redshifts                  --> [array of floats] list of redshifts where you calculated the merger rate
            Average_SF_mass_needed     --> [float]    Msun SF needed to produce the binaries in this simulation
            intrinsic_rate_density     --> [2D array] merger rate in N/Gpc^3/yr
            intrinsic_rate_density_z0  --> [2D array] merger rate in N/Gpc^3/yr at finest/lowest redshift bin calculated

    """
    ################################################
    ## Read merger rate related data
    if verbose: print('Reading ',loc)
    ################################################
    ## Open hdf5 file
    File        = h5.File(loc ,'r')
    redshifts                 = File['redshifts'][()]
    
    # Different per rate key:
    DCO_mask                  = File[rate_key][dcomask][()] # Mask from DCO to merging systems  
    #(contains filter for RLOF>CE and optimistic CE)
    if verbose: print('sum(DCO_mask)', sum(DCO_mask))
    intrinsic_rate_density    = File[rate_key]['merger_rate'][()]
    intrinsic_rate_density_z0 = File[rate_key]['merger_rate_z0'][()] #Rate density at z=0 for the smallest z bin
    
    print('np.shape(intrinsic_rate_density)',np.shape(intrinsic_rate_density) )
    
    File.close()
    
    return DCO_mask, redshifts, intrinsic_rate_density, intrinsic_rate_density_z0 





#########################################
# Bin rate density over crude z-bin
#########################################
def get_crude_rate_density(intrinsic_rate_density, fine_redshifts, crude_redshifts):
    """
        A function to take the 'volume averaged' intrinsic rate density for large (crude) redshift bins. 
        This takes into account the change in volume for different redshift shells

        !! This function assumes an integrer number of fine redshifts fit in a crude redshiftbin !!
        !! We also assume the fine redshift bins and crude redshift bins are spaced equally in redshift !!
        
        Args:
            intrinsic_rate_density    --> [2D float array] Intrinsic merger rate density for each binary at each redshift in 1/yr/Gpc^3
            fine_redshifts            --> [list of floats] Edges of redshift bins at which the rates where evaluated
            crude_redshifts           --> [list of floats] Merger rate for each binary at each redshift in 1/yr/Gpc^3

        Returns:
            crude_rate_density       --> [2D float array] Intrinsic merger rate density for each binary at new crude redshiftbins in 1/yr/Gpc^3

    """
    # Calculate the volume of the fine redshift bins
    fine_volumes       = cosmo.comoving_volume(fine_redshifts).to(u.Gpc**3).value
    fine_shell_volumes = np.diff(fine_volumes) #same len in z dimension as weight

    
    # Multiply intrinsic rate density by volume of the redshift shells, to get the number of merging BBHs in each z-bin
    N_BBH_in_z_bin         = (intrinsic_rate_density[:,:] * fine_shell_volumes[:])
    
    # !! the following asusmes your redshift bins are equally spaced in both cases!!
    # get the binsize of 
    fine_binsize, crude_binsize    = np.diff(fine_redshifts), np.diff(crude_redshifts) 
    if np.logical_and(np.all(np.round(fine_binsize,8) == fine_binsize[0]),  np.all(np.round(crude_binsize,8) == crude_binsize[0]) ):
        fine_binsize    = fine_binsize[0]
        crude_binsize   = crude_binsize[0] 
    else:
        print('Your fine redshifts or crude redshifts are not equally spaced!,',
              'fine_binsize:', fine_binsize, 'crude_binsize', crude_binsize)
        return -1

    # !! also check that your crude redshift bin is made up of an integer number of fine z-bins !!
    i_per_crude_bin = crude_binsize/fine_binsize 
    print('i_per_crude_bin', i_per_crude_bin)
    if (i_per_crude_bin).is_integer():
        i_per_crude_bin = int(i_per_crude_bin)
    else: 
        print('your crude redshift bin is NOT made up of an integer number of fine z-bins!: i_per_crude_bin,', i_per_crude_bin)
        return -1
    
    
    # add every i_per_crude_bin-th element together, to get the number of merging BBHs in each crude redshift bin
    N_BBH_in_crudez_bin    = np.add.reduceat(N_BBH_in_z_bin, np.arange(0, len(N_BBH_in_z_bin[0,:]), int(i_per_crude_bin) ), axis = 1)
    
    
    # Convert crude redshift bins to volumnes and ensure all volumes are in Gpc^3
    crude_volumes       = cosmo.comoving_volume(crude_redshifts).to(u.Gpc**3).value
    crude_shell_volumes = np.diff(crude_volumes)# split volumes into shells 
    
    
    # Finally tunr rate back into an average (crude) rate density, by dividing by the new z-volumes
    # In case your crude redshifts don't go all the way to z_first_SF, just use N_BBH_in_crudez_bin up to len(crude_shell_volumes)
    crude_rate_density     = N_BBH_in_crudez_bin[:, :len(crude_shell_volumes)]/crude_shell_volumes
    
    return crude_rate_density

def broken_power_law(m1s,lmbda1,lmbda2,mBreak,mMin,mMax,dmMin):
    
    """
    From Callister&Farr 2024 - https://github.com/tcallister/autoregressive-bbh-inference/blob/main/figures/make_figure_03.ipynb

    Helper function defining an unnormalized broken power law
    
    Parameters
    ----------
    m1s : Set of primary masses
    lmbda1 : Power-law index at low masses
    lmbda2 : Power-law index at high masses
    mBreak : Break point at which power-law slope changes
    mMin : Mass below which we will smoothly taper to zero
    mMax : Maximum mass above which our distribution vanishes
    dmMin : Scale over which low-mass tapering is performed
    """
    
    # Define broken power law
    bpl = np.where(m1s<mBreak,(m1s/mBreak)**lmbda1,(m1s/mBreak)**lmbda2)
    
    # Apply smooth tapering below mMin
    tapering = np.exp(-(m1s-mMin)**2/(2.*dmMin**2))
    bpl = np.where(m1s<mMin,bpl*tapering,bpl)
    
    # Set masses above mMax to zero
    bpl[m1s>mMax] = 0
    
    return bpl

def CallisterFarr_model(m1,f_peak_1,f_peak_2,mu1,sig1,mu2,sig2,lmbda1,lmbda2,mBreak,mMin,mMax,dmMin):
    
    """

    Model from Callister&Farr 2024 - https://github.com/tcallister/autoregressive-bbh-inference/blob/main/figures/make_figure_03.ipynb

    Full expression for our parametric model that we will fit to our AR mass results,
    comprising a broken power law and two Gaussian peaks
    
    Parameters
    ----------
    m1 : Mass values at which to evaluate
    f_peak_1 : Mixture fraction of events occurring in the low-mass peak
    f_peak_2 : Mixture fraction of events occurring in the high-mass peak
    mu1 : Location of low mass peak
    sig1 : Standard deviation of low mass peak
    mu2: Location of high mass peak
    sig1 : Standard deviation of high mass peak
    lmbda1 : Power-law slope below the break
    lmbda2 : Power-law slope above the break
    mBreak : Break point at which power-law slope changes
    mMin : Mass below which the power-law component is smoothly tapered to zero
    mMax : Maximum mass above which our distribution vanishes
    dmMin : Scale over which low-mass tapering is performed
    """
    
    # Define integral over tapered broken power law
    m1_grid = np.linspace(2,100,1000)
    bpl_grid = broken_power_law(m1_grid,lmbda1,lmbda2,mBreak,mMin,mMax,dmMin)
    bpl_norm = np.trapz(bpl_grid,m1_grid)
    
    # Normalized broken (and tapered) power-law
    bpl = broken_power_law(m1,lmbda1,lmbda2,mBreak,mMin,mMax,dmMin)/bpl_norm
    
    # Probability densities corresponding to each peak
    peak_1 = np.exp(-(m1-mu1)**2/(2.*sig1**2))/np.sqrt(2.*np.pi*sig1**2)
    peak_2 = np.exp(-(m1-mu2)**2/(2.*sig2**2))/np.sqrt(2.*np.pi*sig2**2)
    
    # Construct and return full distribution
    p_m1 = f_peak_1*peak_1 + f_peak_2*peak_2 + (1.-f_peak_1-f_peak_2)*bpl
    return p_m1

