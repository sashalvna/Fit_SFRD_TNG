import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from scipy import interpolate
from scipy.optimize import minimize

from astropy.cosmology import Planck15  as cosmo # Planck 2015 since that's what TNG uses

############################
# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
import paths

def readTNGdata(loc = './', rbox=75, SFR=False, metals=True):
    ##########################################
    # Simulated SFRD data (from TNG)
    ##########################################
    ## Read in the pure simulation data
    with h5.File(loc, "r") as f:
        MetalBins     = f["MetalBins"][:]
        Lookbacktimes = f["Lookbacktimes"][:]
        BoxSfr        = f["Sfr"][:]
        Redshifts     = f["Redshifts"][:]
        if metals==True:
            Metals        = f["Metals"][:]

    Sim_center_Zbin  = (MetalBins[:-1] + MetalBins[1:])/2.

    # Convert SFR from sfr/box to sfr Mpc-3
    littleh  = 0.6774
    Rbox     = rbox/littleh
    Sim_SFRD = BoxSfr / Rbox**3 *u.Mpc**-3
    Sim_SFRD = Sim_SFRD.value

    step_fit_logZ  = np.diff(np.log(MetalBins))[0]    

    if SFR == False:
        ## The model comes in SFRD/DeltaZ, make sure your data does as well!! 
        Sim_SFRD       = Sim_SFRD/step_fit_logZ
    
    if metals==True:
        return Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals, MetalBins
    else:
        return Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ


def interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, ver, minZ_popSynth=1e-6, saveplot=False, redshiftlimandstep=[0, 14.1, 0.05]):

    # Adjust what metallicities to include 
    tofit_Sim_metals = Sim_center_Zbin[np.where(Sim_center_Zbin > minZ_popSynth)[0]]   

    # Reverse the time axis of the SFRD and lookback time for the fit
    tofit_Sim_SFRD      = Sim_SFRD[:,np.where(Sim_center_Zbin > minZ_popSynth)[0]][::-1]
    tofit_Sim_lookbackt = Lookbacktimes[::-1] 

    # Interpolate the simulation data
    f_interp = interpolate.interp2d(tofit_Sim_lookbackt, tofit_Sim_metals, tofit_Sim_SFRD.T, kind='cubic')

    # Retrieve values at higher res regular intervals
    redshift_new         = np.arange(redshiftlimandstep[0], redshiftlimandstep[1], redshiftlimandstep[2])
    Lookbacktimes_new    = [cosmo.lookback_time(z).value for z in redshift_new]
    redshifts_Sim = Redshifts

    #log_tofit_Sim_metals = np.log10(tofit_Sim_metals)
    log_tofit_Sim_metals = np.log10(tofit_Sim_metals)
    print("min max metals", min(tofit_Sim_metals), max(tofit_Sim_metals))
    print("log min max metals", min(log_tofit_Sim_metals), max(log_tofit_Sim_metals))
    metals_new           = np.logspace(min(log_tofit_Sim_metals), max(log_tofit_Sim_metals), 500) #base=np.e

    SFRDnew = f_interp(Lookbacktimes_new,metals_new)
    SFRDnew[SFRDnew < 0] = 0
    step_fit_logZ_new = np.diff(np.log(metals_new))[0]

    if saveplot==True:
        #check what the SFR and interpolated SFRD (with metallicities) look like 
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax[0].plot(redshifts_Sim, tofit_Sim_SFRD)
        ax[0].set_xlim(0,2)
        ax[0].set_xlabel("Redshift")
        ax[0].set_ylabel("SFRD(z)")

        ax[1].plot(redshift_new, SFRDnew.T)
        ax[1].set_title('Interpolated')
        ax[1].set_xlabel("Redshift")
        ax[1].set_ylabel("SFRD(z)");

        if ver>1:
            fig.savefig('figures/TNG%s_%s_dataSFRD.png'%(tng, ver), bbox_inches='tight')
        else:
            fig.savefig('figures/TNG%s_dataSFRD.png'%tng, bbox_inches='tight')

    return(SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new)


def calc_chi_square(fit_metals, Redshifts = [],  simulation_SFRD = [],
                    mu_0  = 0.026, muz  =-0.09, sigma0  = 1.9, sigmaz  = 1.9, alpha =-3.3,
                    sf_a =0.01 , sf_b=2.6, sf_c=3.2 , sf_d=6.2):
    """
    Calculate the squared residual of your simulation_SFRD vs your analytical model.
    Args:
        fit_metals        --> [float]          metals used for fit
        Redshifts         --> [float]          redshifts used to fit
        simulation_SFRD   --> [float]          cosmological simulation data to fit to
        
        # metallicity distribution parameters (skew-log-normal)
        mu_0              --> [float]          location (mean in normal) at redshift 0
        muz               --> [float]          redshift evolution of the location
        sigma0            --> [float]          Scale at redshift 0 (variance in normal)
        sigmaz            --> [float]          redshift evolution of Scale (variance in normal)
        alpha             --> [float]          shape (skewness, alpha = 0 retrieves normal dist)
        # overall SFR parameters
        sf_a              --> [float]          SFR(z) parameter (shape of Madau & Dickenson 2014)
        sf_b              --> [float]          SFR(z) parameter (shape of Madau & Dickenson 2014)
        sf_c              --> [float]          SFR(z) parameter (shape of Madau & Dickenson 2014)
        sf_d              --> [float]          SFR(z) parameter (shape of Madau & Dickenson 2014)

    Returns:
        tot_chi_square    --> [float ] 

    """ 
    #####################################
    # Get the SFR (Shape of Madau & Fragos 2014)
    sfr = Z_SFRD.Madau_Dickinson2014(Redshifts, a=sf_a, b=sf_b, c=sf_c, d=sf_d).value # Msun year-1 Mpc-3 
    # Get dPdZ (skew-log-normal distribution)
    dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                    Z_SFRD.skew_metallicity_distribution(Redshifts,mu_z = muz , mu_0 = mu_0 ,
                                                  omega_0= sigma0 , omega_z=sigmaz , alpha = alpha , 
                                                  metals=fit_metals)
    
    ######################################
    data           = simulation_SFRD                # Model comes in dP/dlogZ, so should your sim-data !
    model          = sfr[:,np.newaxis] * dPdlogZ 

    # The square of the residuals
    res_squared = ((data - model )**2)
    
    # Divide over the sum of the model at each redshift,
    # reduces contribution from high sfr redshifts & increases the weight where sfr is low
    chi_square = res_squared/np.sum(model, axis = 1)[:,np.newaxis]
    
    # Return sum Chi_squared and the max squared residual
    return np.sum(chi_square), data, model  


def test_chi(x0 = [-0.09, 0.026, 1.9, 0.1, -3.3, 0.01, 2.6, 3.2, 6.2] ):
    chi_square, data, model = calc_chi_square(metals_new, Redshifts = redshift_new, simulation_SFRD = SFRDnew.T, 
                                       muz  =x0[0], mu_0  =x0[1],sigma0  =x0[2], sigmaz =x0[3], alpha  =x0[4],
                                       sf_a =x0[5], sf_b=x0[6], sf_c=x0[7], sf_d=x0[8])
    return chi_square


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    tng= 100
    ver = 2
    Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasTNG%s-%s.hdf5"%(tng,ver))
    fit_filename = 'test_best_fit_parameters_TNG%s-%s_TEST.txt'%(tng,ver)
    if tng==50:
        rbox=35
    elif tng==100:
        rbox=75
    elif tng==300:
        rbox=205

    if tng==300:
        minimize_method = 'nelder-mead' #nelder-mead, BFGS
    elif ver>1:
        minimize_method = 'nelder-mead' #nelder-mead, BFGS
    else:
        minimize_method = 'BFGS'

    #Read the TNG data and interpolate it
    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals, MetalBins = readTNGdata(loc = Cosmol_sim_location, rbox=rbox)
    SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, saveplot=True, tng=tng, ver=ver, redshiftlimandstep=[0, 14.1, 0.05])
    
    #Fit the model to the data
    x0     = np.array([-0.15, 0.026, 1.1, 0.1, -3.3, 0.01, 2.6, 3.2, 6.2]) #best guess
    #        # mu_z        # mu_0     # omega_0 #omega_z  #alpha       #sf_a       #sf_b       #sf_c       #sf_d
    bounds = ((-1., 0), (0.001, 0.1), (0.01, 5), (0, 1.), (-10, 0), (None,None),(None,None),(None,None),(None,None))

    # FIT
    if minimize_method=='BFGS':
        res = minimize(test_chi, x0= x0, method = 'BFGS',options = {'gtol': 0.05})
    
    else:
        res = minimize(test_chi, x0= x0, method=minimize_method, options = {'maxiter' : 2000})


    print(res.success, res.message, 'N iterations: ', res.nit)
    muz_best, mu0_best, sigma0_best, sigmaz_best, alpha_best = res.x[0], res.x[1], res.x[2], res.x[3],res.x[4]
    sf_a_best, sf_b_best, sf_c_best, sf_d_best               = res.x[5], res.x[6], res.x[7], res.x[8] 

    print('\nBEST FITTING PARAMETERS:')
    print('mu0 =%s, muz =%s, sigma_0 =%s, sigma_z =%s, alpha=%s'% (mu0_best, muz_best, sigma0_best, sigmaz_best, alpha_best) )
    print('sf_a =%s, sf_b =%s, sf_c =%s, sf_d =%s'% (sf_a_best, sf_b_best, sf_c_best, sf_d_best) )

    chi_square, data, model = calc_chi_square(metals_new, Redshifts = redshift_new, simulation_SFRD = SFRDnew.T, 
                                   muz =muz_best, mu_0 =mu0_best,sigma0 =sigma0_best, sigmaz=sigmaz_best,alpha=alpha_best,
                                   sf_a =sf_a_best, sf_b=sf_b_best, sf_c=sf_c_best, sf_d=sf_d_best)
    print('chi_square',chi_square, 'max_res_squared', np.amax( (data-model)**2) )

    if res.success:
        np.savetxt(paths.data / fit_filename,
               np.c_[mu0_best, muz_best, sigma0_best, sigmaz_best, alpha_best,sf_a_best, sf_b_best, sf_c_best, sf_d_best],
               header = "mu0, muz, omega0, omegaz, alpha0,sf_a, sf_b, sf_c, sf_d", delimiter=',', fmt="%s")