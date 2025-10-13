import h5py as h5
import numpy as np
import astropy.units as u
import scipy.interpolate as interpolate
from astropy.cosmology import Planck15 as cosmology

def readTNGdata(filename):
    ##########################################
    # Simulated SFRD data (from TNG)
    ##########################################

    filename = str(filename)

    if 'TNG50' in filename:
        rbox=35
    elif 'TNG100' in filename:
        rbox=75
    elif 'TNG300' in filename:
        rbox = 205
    else:
        raise Exception("Make sure that you are reading in TNG data, and that the TNG version is contained in the filename (e.g. TNG50)")

    ## Read in the simulation data
    with h5.File(filename, "r") as f:
        MetalBins     = f["MetalBins"][:]
        Lookbacktimes = f["Lookbacktimes"][:]
        BoxSfr        = f["Sfr"][:]
        Redshifts     = f["Redshifts"][:]
        Metals        = f["Metals"][:]

    Sim_center_Zbin  = (MetalBins[:-1] + MetalBins[1:])/2.

    # Convert SFR from sfr/box to sfr Mpc-3
    littleh  = 0.6774
    Rbox     = rbox/littleh
    Sim_SFRD = BoxSfr / Rbox**3 *u.Mpc**-3
    Sim_SFRD = Sim_SFRD.value

    step_fit_logZ  = np.diff(np.log(MetalBins))[0]    

    # The model comes in SFRD/DeltaZ, make sure your data does as well!! 
    Sim_SFRD       = Sim_SFRD/step_fit_logZ
    
    return Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals

def interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, Metaldist, minZ_popSynth=1e-6, redshiftlimandstep=[0, 10.1, 0.05], nmetals=500):

    # Adjust what metallicities to include 
    tofit_Sim_metals = Sim_center_Zbin[np.where(Sim_center_Zbin > minZ_popSynth)[0]]   

    # Reverse the time axis of the SFRD and lookback time for the fit
    tofit_Sim_SFRD      = Sim_SFRD[:,np.where(Sim_center_Zbin > minZ_popSynth)[0]][::-1]
    tofit_Sim_Metaldist = Metaldist[:,np.where(Sim_center_Zbin > minZ_popSynth)[0]][::-1]
    tofit_Sim_lookbackt = Lookbacktimes[::-1]

    # Interpolate the simulation data
    f_interp = interpolate.interp2d(tofit_Sim_lookbackt, tofit_Sim_metals, tofit_Sim_SFRD.T, kind='cubic')
    f_interp_metals = interpolate.interp2d(tofit_Sim_lookbackt, tofit_Sim_metals, tofit_Sim_Metaldist.T, kind='cubic')

    # Retrieve values at higher res regular intervals
    Redshift_new         = np.arange(redshiftlimandstep[0], redshiftlimandstep[1], redshiftlimandstep[2])
    Lookbacktimes_new    = [cosmology.lookback_time(z).value for z in Redshift_new]

    # Get new metallicity bins
    log_tofit_Sim_metals = np.log10(tofit_Sim_metals) 
    metals_new           = np.logspace(min(log_tofit_Sim_metals), max(log_tofit_Sim_metals), nmetals)
    step_fit_logZ_new = np.diff(np.log(metals_new))[0]

    # Interpolate SFRD and metallicity distribution
    SFRDnew = f_interp(Lookbacktimes_new,metals_new)
    SFRDnew[SFRDnew < 0] = 0
    Metaldist_new = f_interp_metals(Lookbacktimes_new, metals_new) #same bins as SFRDnew
    
    return SFRDnew, Redshift_new, Lookbacktimes_new, metals_new, Metaldist_new, step_fit_logZ_new