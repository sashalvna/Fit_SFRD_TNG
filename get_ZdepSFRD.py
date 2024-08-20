import numpy as np
import astropy.units as u
from scipy.stats import norm as NormDist

########################################################
## Chrip mass
########################################################
def Mchirp(m1, m2):
    chirp_mass = np.divide(np.power(np.multiply(m1, m2), 3./5.), np.power(np.add(m1, m2), 1./5.))
    return chirp_mass    


########################################################
## # SFR(z) Madau & Dickinson 2014 shape
########################################################
def Madau_Dickinson2014(z, a=0.015, b=2.77, c=2.9, d=5.6):
    """
    Args:
        z             --> [list of floats] List of redshifts at which to calculate things
        a,b,c,d       --> [floats] values to determine the shape of our SFR
    
    Calculates the star-formation rate density as a function of redshift
    Based on the functional form from Madau & Dickinson 2014
    default 'Neijssel et al 2019': a=0.01, b=2.77, c=2.9,  d=4.7
    Madau & Dickinson 2014: a=0.015, b=2.7, c=2.9,  d=5.6
    Madau & Fragos 2017: a=0.01, b=2.6, c=3.2,  d=6.2

    Returns:
        SFR(z) in Msun/yr/Mpc^3
    """
    dm_dtdMpc = a * (1 + z)**b/( 1 + ( (1+z)/c )**d ) *u.Msun *u.yr**-1 *u.Mpc**-3
    return dm_dtdMpc # Msun year-1 Mpc-3 


    
########################################################
##  The mettalicity distribution dP/dZ(z)
########################################################
def skew_metallicity_distribution(redshifts, metals = [], min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03),
                                  mu_0=0.025, mu_z=-0.048, omega_0=1.125, omega_z=0.048, alpha =-1.767,
                                  min_logZ=-12.0, max_logZ=0.0, step_logZ =0.01):
                                 
    """
    Calculate the distribution of metallicities at different redshifts using a log skew normal distribution
    the log-normal distribution is a special case of this log skew normal distribution distribution, and is retrieved by setting 
    the skewness to zero (alpha = 0). 
    Based on the method in Neijssel+19. Default values of mu_0=0.035, mu_z=-0.23, omega_0=0.39, omega_z=0.0, alpha =0.0, 
    retrieve the dP/dZ distribution used in Neijssel+19

    NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution!

    Args:
        max_redshift       --> [float]          max redshift for calculation
        redshift_step      --> [float]          step used in redshift calculation
        min_logZ_COMPAS    --> [float]          Minimum logZ value that COMPAS samples
        max_logZ_COMPAS    --> [float]          Maximum logZ value that COMPAS samples
        
        mu_0    =  0.035    --> [float]           location (mean in normal) at redshift 0
        mu_z    = -0.25    --> [float]           redshift scaling/evolution of the location
        omega_0 = 0.39     --> [float]          Scale (variance in normal) at redshift 0
        omega_z = 0.00     --> [float]          redshift scaling of the scale (variance in normal)
        alpha   = 0.00    --> [float]          shape (skewness, alpha = 0 retrieves normal dist)

        min_logZ           --> [float]          Minimum logZ at which to calculate dPdlogZ (influences normalization)
        max_logZ           --> [float]          Maximum logZ at which to calculate dPdlogZ (influences normalization)
        step_logZ          --> [float]          Size of logZ steps to take in finding a Z range

    Returns:
        dPdlogZ            --> [2D float array] Probability of getting a particular logZ at a certain redshift
        metallicities      --> [list of floats] Metallicities at which dPdlogZ is evaluated
        p_draw_metallicity --> float            Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)
    """ 
    ##################################
    # Log-Linear redshift dependence of omega, eq. 9 van Son 2023
    omega = omega_0* 10**(omega_z*redshifts)
    
    ##################################
    # Follow Langer & Norman 2006 in assuming that mean metallicities evolve in z
    # eq. 7 
    mean_metallicities = mu_0 * 10**(mu_z * redshifts) 
    
    ## The first moment of our log-skew-normal (i.e. E[dP/dZ] is:
    # eq. 8 here: https://www-sciencedirect-com.ezp-prod1.hul.harvard.edu/science/article/pii/S0021850219300400
    beta = alpha/(np.sqrt(1 + (alpha)**2))
    PHI  = NormDist.cdf(beta * omega)  
    PHI[PHI == 0] = 1e-310                                                    # Avoid division by zero
    # eq. 8 in van Son 2022
    xi_metallicities = np.log(mean_metallicities/(2 * PHI)) - (omega**2)/2.   # Now we re-write the expected value to retrieve xi


    ##################################
    if len(metals) == 0:
        # create a range of metallicities (thex-values, or raandom variables)
        log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
        metallicities     = np.exp(log_metallicities)
    else: 
        #use a pre-determined array of metals
        metallicities     = metals
        log_metallicities = np.log(metallicities)
        step_logZ         = np.diff(log_metallicities)[0]
    
    
    ##################################
    # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
    ### eq.2) in van son +22 dP/dlogZ = 2/omega * phi((lnZ- xi)/ omega) * PHI(alpha (lnZ - xi)/omega)
    dPdlogZ = 2./(omega[:,np.newaxis]) \
    * NormDist.pdf((log_metallicities -  xi_metallicities[:,np.newaxis])/omega[:,np.newaxis]) \
    * NormDist.cdf( alpha * (log_metallicities -  xi_metallicities[:,np.newaxis]) /omega[:,np.newaxis] )
    #(see also eq.  1 here: https://www.emis.de/journals/RCE/V36/v36n1a03.pdf)

    ##################################
    # normalise the distribution over all metallicities
    norm    = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ /norm[:,np.newaxis]

    ##################################
    # assume a flat in log distribution in metallicity to find probability of drawing Z in COMPAS
    p_draw_metallicity = 1 / (max_logZ_COMPAS - min_logZ_COMPAS)
    
    return dPdlogZ, metallicities, step_logZ, p_draw_metallicity



