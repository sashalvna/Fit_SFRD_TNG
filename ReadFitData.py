"""
Simple script to load fit data for SFRD fit
"""
import h5py as h5
import numpy as np
import astropy.units as u

# from astropy.cosmology import WMAP9, z_at_value
from astropy.cosmology import Planck15  as cosmo# Planck 2015
from astropy.cosmology import z_at_value


def load_TNG100(loc = './'):
    ##########################################
    # Simulated SFRD data (from TNG)
    ##########################################
    ## Read in the pure simulation data
    with h5.File(loc, "r") as f:
        MetalBins     = f["MetalBins"][:]
        Lookbacktimes = f["Lookbacktimes"][:]
        BoxSfr        = f["Sfr"][:]
        Redshifts     = f["Redshifts"][:]

    Sim_center_Zbin  = (MetalBins[:-1] + MetalBins[1:])/2.

    # Convert SFR from sfr/box to sfr Mpc-3
    littleh  = 0.6774
    Rbox     = 75/littleh
    Sim_SFRD = BoxSfr / Rbox**3 *u.Mpc**-3
    Sim_SFRD = Sim_SFRD.value

    ## The model comes in SFRD/DeltaZ, make sure your data does as well!! 
    step_fit_logZ  = np.diff(np.log(MetalBins))[0]    
    Sim_SFRD       = Sim_SFRD/step_fit_logZ
    
    return Sim_SFRD, Lookbacktimes, Sim_center_Zbin, step_fit_logZ, Redshifts



def load_martyna_data(data_dir = './', file_name = '204'+ 'f14SB'+'Boco'+ '_FMR270'+ '_FOH_z_dM.dat', 
                       zmin=0., zmax = 10.):
    ##########################################
    # Observational (Chruslinska data)
    ##########################################
    def solar_metallicity_scales():
        Asplund09=[0.0134,8.69]
        AndersGrevesse89=[0.017,8.83]
        GrevesseSauval98=[0.0201,8.93]
        Villante14=[0.019,8.85]
        scale_ref=np.array(['Asplund09','AndersGrevesse89','GrevesseSauval98','Villante14'])
        Z_FOH_solar=np.array([Asplund09,AndersGrevesse89,GrevesseSauval98,Villante14])
        return scale_ref, Z_FOH_solar


    ##########################
    def FOH2ZZ(foh,solar_Z_scale='AndersGrevesse89'):
        '''convert from 12+log[O/H] to ZZ'''
        scale_ref, Z_FOH_solar=solar_metallicity_scales()
        idx=np.where(scale_ref==solar_Z_scale)[0][0]
        Zsun,FOHsun = Z_FOH_solar[idx]    
        logZ = np.log10(Zsun) + foh - FOHsun
        ZZ=10**logZ
        return ZZ

    #(array) oxygen to hydrogen abundance ratio ( FOH == 12 + log(O/H) )
    # as used in the calculations - do not change
    FOH_min, FOH_max = 5.3, 9.7
    FOH_arr = np.linspace( FOH_min,FOH_max, 200)
    dFOH=FOH_arr[1]-FOH_arr[0]

    #read time, redshift and timestep as used in the calculations
    #starts at the highest redshift (z=z_start=10) and goes to z=0
    time, redshift_global, delt = np.loadtxt(data_dir +'Time_redshift_deltaT.dat',unpack=True) 
    #reading mass per unit (comoving) volume formed in each z (row) - FOH (column) bin
    data=np.loadtxt(data_dir + file_name )
    image_data=np.array( [data[ii]/(1e6*delt[ii]) for ii in range(len(delt))] )#fill the array with SFRD(FOH,z)
    redshift=redshift_global

    image_data/=dFOH

    # np.log10(Z_array/ZsunTNG), lowZ_image[min_diff_i,:] ,
    # lowZ_image, redshift, delt 
    Sim_center_Zbin  = FOH2ZZ(FOH_arr,solar_Z_scale='AndersGrevesse89')
    Lookbacktimes    = [cosmo.lookback_time(z).value for z in redshift]
    Sim_SFRD         = image_data
    
    return Sim_SFRD, Lookbacktimes, Sim_center_Zbin, dFOH


# print('MetalBins', np.log10(MetalBins))
