import numpy as np
import h5py  as h5
import os
import time
import matplotlib.pyplot as plt
#from astropy.cosmology import Planck15 as cosmology
import scipy
from scipy.interpolate import interp1d
from scipy.stats import norm as NormDist
import ClassCOMPAS
import selection_effects
import warnings
import astropy.units as u
import argparse
import importlib
from read_TNGdata import readTNGdata, interpolate_TNGdata

def calculate_redshift_related_params(max_redshift=10.0, max_redshift_detection=1.0, redshift_step=0.001, z_first_SF = 10.0):
    """ 
        Given limits on the redshift, create an array of redshifts, times, distances and volumes

        Args:
            max_redshift           --> [float]          Maximum redshift to use for calculations
            max_redshift_detection --> [float]          Maximum redshift to calculate detection rates (must be <= max_redshift)
            redshift_step          --> [float]          size of step to take in redshift
            z_first_SF             --> [float]          redshift of first star formation

        Returns:
            redshifts              --> [list of floats] List of redshifts between limits supplied
            n_redshifts_detection  --> [int]            Number of redshifts in list that should be used to calculate detection rates
            times                  --> [list of floats] Equivalent of redshifts but converted to age of Universe
            distances              --> [list of floats] Equivalent of redshifts but converted to luminosity distances
            shell_volumes          --> [list of floats] Equivalent of redshifts but converted to shell volumes
    """
    # create a list of redshifts and record lengths
    redshifts = np.arange(0, max_redshift + redshift_step, redshift_step)
    n_redshifts_detection = int(max_redshift_detection / redshift_step)

    # convert redshifts to times and ensure all times are in Myr
    times = cosmology.age(redshifts).to(u.Myr).value

    # and time of first Sf
    time_first_SF = cosmology.age(z_first_SF).to(u.Myr).value

    # convert redshifts to distances and ensure all distances are in Mpc (also avoid D=0 because division by 0)
    distances = cosmology.luminosity_distance(redshifts).to(u.Mpc).value
    distances[0] = 0.001

    # convert redshifts to volumnes and ensure all volumes are in Gpc^3
    volumes = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value

    # split volumes into shells and duplicate last shell to keep same length
    shell_volumes = np.diff(volumes)
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])

    return redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes


def find_sfr(redshifts, a = 0.01, b =2.77, c = 2.90, d = 4.70):
    """
        Calculate the star forming mass per unit volume per year following
        Neijssel+19 Eq. 6, using functional form of Madau & Dickinson 2014

        Args:
            redshifts --> [list of floats] List of redshifts at which to evaluate the sfr

        Returns:
            sfr       --> [list of floats] Star forming mass per unit volume per year for each redshift
    """
    # get value in mass per year per cubic Mpc and convert to per cubic Gpc then return
    sfr = a * ((1+redshifts)**b) / (1 + ((1+redshifts)/c)**d) * u.Msun / u.yr / u.Mpc**3
    return sfr.to(u.Msun / u.yr / u.Gpc**3).value


def find_metallicity_distribution(redshifts, min_logZ_COMPAS, max_logZ_COMPAS,
                                  mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0,
                                  min_logZ  =-12.0, max_logZ  =0.0, step_logZ = 0.01):
                                 
    """
    Calculate the distribution of metallicities at different redshifts using a log skew normal distribution
    the log-normal distribution is a special case of this log skew normal distribution distribution, and is retrieved by setting 
    the skewness to zero (alpha = 0). 
    Based on the method in Neijssel+19. Default values of mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0, 
    retrieve the dP/dZ distribution used in Neijssel+19

    NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution!

    Args:
        max_redshift       --> [float]          max redshift for calculation
        redshift_step      --> [float]          step used in redshift calculation
        min_logZ_COMPAS    --> [float]          Minimum logZ value that COMPAS samples
        max_logZ_COMPAS    --> [float]          Maximum logZ value that COMPAS samples
        
        mu0    =  0.035    --> [float]           location (mean in normal) at redshift 0
        muz    = -0.25    --> [float]           redshift scaling/evolution of the location
        sigma_0 = 0.39     --> [float]          Scale (variance in normal) at redshift 0
        sigma_z = 0.00     --> [float]          redshift scaling of the scale (variance in normal)
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
    # Log-Linear redshift dependence of sigma
    sigma = sigma_0* 10**(sigma_z*redshifts)
    
    ##################################
    # Follow Langer & Norman 2007? in assuming that mean metallicities evolve in z as:
    mean_metallicities = mu0 * 10**(muz * redshifts) 
        
    # Now we re-write the expected value of ou log-skew-normal to retrieve mu
    beta = alpha/(np.sqrt(1 + (alpha)**2))
    PHI  = NormDist.cdf(beta * sigma) 
    mu_metallicities = np.log(mean_metallicities/(2.*PHI)) - (sigma**2)/2.   

    ##################################
    # create a range of metallicities (thex-values, or random variables)
    log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
    metallicities = np.exp(log_metallicities)


    ##################################
    # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
    dPdlogZ = 2./(sigma[:,np.newaxis]) * NormDist.pdf((log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis]) * NormDist.cdf(alpha * (log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis] )

    ##################################
    # normalise the distribution over al metallicities
    norm = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ /norm[:,np.newaxis]

    ##################################
    # assume a flat in log distribution in metallicity to find probability of drawing Z in COMPAS
    p_draw_metallicity = 1 / (max_logZ_COMPAS - min_logZ_COMPAS)
    
    return dPdlogZ, metallicities, p_draw_metallicity

def calc_dPdlogZ(COMPAS_metallicity, metallicities, Metal_dist):

    metallicity_ind = np.digitize(COMPAS_metallicity, metallicities)
    dPdlogZs =  Metal_dist[:, metallicity_ind]

    #metallicity_ind = np.digitize(COMPAS_metallicity, metallicities)
    #metal_dist_form =  Metal_dist[metallicity_ind, :]
    #dPdlogZs = metal_dist_form/np.trapz(metal_dist_form)

    #for z_ind, z in enumerate(redshifts):

        #find closest value in the metallicities in the data to the binary metallicity
        #metallicity_ind = np.digitize()
        
        #np.argmin(np.abs(log_data_metallicity - log_binary_metallicity)) #index
        #metallicity_form = 10**log_data_metallicity[metallicity_ind] #value of closest metallicity

        #find the probability of getting that metallicity using the metallicity distribution at this redshift
        #metal_dist_form = metallicity_dists[:, z_ind]
        #norm_metal_dist_form = metal_dist_form/np.trapz(metal_dist_form)

        #if ind==0 and z_ind==0:
        #    print('Metallicity form: ', metallicity_form, metallicity_ind, norm_metal_dist_form[metallicity_ind])

        #dPdlogZs.append(norm_metal_dist_form[metallicity_ind])
        #metals_form.append(metallicity_form)

    return dPdlogZs


def find_formation_and_merger_rates(n_binaries, redshifts, times, time_first_SF, metallicities, p_draw_metallicity,
                                    COMPAS_metallicites, COMPAS_delay_times, COMPAS_weights=None, n_formed=None, dPdlogZ=None,
                                    Metal_distributions=[]):
    """
        Find both the formation and merger rates for each binary at each redshift

        Args:
            n_binaries          --> [int]            Number of DCO binaries in the arrays
            redshifts           --> [list of floats] Redshifts at which to evaluate the rates
            times               --> [list of floats] Equivalent of the redshifts in terms of age of the Universe
            n_formed            --> [float]          Binary formation rate (number of binaries formed per year per cubic Gpc) represented by each simulated COMPAS binary
            dPdlogZ             --> [2D float array] Probability of getting a particular logZ at a certain redshift
            metallicities       --> [list of floats] Metallicities at which dPdlogZ is evaluated
            p_draw_metallicity  --> [float]          Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)
            COMPAS_metallicites --> [list of floats] Metallicity of each binary in COMPAS data
            COMPAS_delay_times  --> [list of floats] Delay time of each binary in COMPAS data
            COMPAS_weights      --> [list of floats] Adaptive sampling weights for each binary in COMPAS data (defaults to all 1s for unweighted samples)

        Returns:
            formation_rate      --> [2D float array] Formation rate for each binary at each redshift
            merger_rate         --> [2D float array] Merger rate for each binary at each redshift
    """
    
    # check if weights were provided, if not use uniform weights
    if COMPAS_weights is None:
        COMPAS_weights = np.ones(n_binaries)

    # initalise rates to zero
    n_redshifts = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    formation_rate = np.zeros(shape=(n_binaries, n_redshifts))
    merger_rate = np.zeros(shape=(n_binaries, n_redshifts))

    # interpolate times and redshifts for conversion
    times_to_redshifts = interp1d(times, redshifts)

    # make note of the first time at which star formation occured
    age_first_sfr = time_first_SF

    if len(Metal_distributions) > 0:
        normalized_metal_dists = []
        for z_ind, z in enumerate(redshifts):
            metal_dist_form = Metal_distributions[:, z_ind] * metallicities
            norm_metal_dist_form = metal_dist_form/np.linalg.norm(metal_dist_form)
            normalized_metal_dists.append(norm_metal_dist_form)
        normalized_metal_dists = np.array(normalized_metal_dists)

    # go through each binary in the COMPAS data
    for i in range(n_binaries):
        if len(Metal_distributions) > 0:
            formation_rate[i, :] = n_formed * calc_dPdlogZ(COMPAS_metallicites[i], metallicities, normalized_metal_dists) / p_draw_metallicity * COMPAS_weights[i]

            #if i==0:
                #print("The binary metallicity is", COMPAS_metallicites[i])

                #fig, ax = plt.subplots(figsize = (12,8))
                #plt.plot(redshifts, formation_rate[i])
                #plt.xlim(0, 10)
                #plt.ylabel("Formation rate", fontsize=20)
                #plt.xlabel("Redshift", fontsize=20)
                #plt.title('Z = %s'%COMPAS_metallicites[i])
                #plt.savefig('/home/alevina1/figures/formationrate_data.png')
                #plt.clf()

                #fig, ax = plt.subplots(figsize = (12,8))
                #plt.plot(redshifts, calc_dPdlogZ(COMPAS_metallicites[i], metallicities, normalized_metal_dists))
                #plt.xlim(0, 10)
                #plt.ylabel("dP/dlogZ", fontsize=20)
                #plt.xlabel("Redshift", fontsize=20)
                #plt.title('Z = %s'%COMPAS_metallicites[i])
                #plt.savefig('/home/alevina1/figures/dPdlogZ_data.png')
                #plt.clf()

                #print("min max Z", min(metallicities), max(metallicities))
                #print("min max COMPAS Z", min(COMPAS_metallicites), max(COMPAS_metallicites))
                #print("np digitize", np.digitize(COMPAS_metallicites[i], metallicities))
                #print("metallicities", metallicities[np.digitize(COMPAS_metallicites[i], metallicities)], len(metallicities))
                #print("len dPdlogZ", (calc_dPdlogZ(COMPAS_metallicites[i], metallicities, normalized_metal_dists)).shape)
                #print("dPdlogZ", calc_dPdlogZ(COMPAS_metallicites[i], metallicities, normalized_metal_dists))

            #with open("COMPASbinaryindex.txt", 'w') as f:
            #    f.write("%s"%i)

        else:
            # calculate formation rate (see Neijssel+19 Section 4) - note this uses dPdlogZ for *closest* metallicity
            formation_rate[i, :] = n_formed * dPdlogZ[:, np.digitize(COMPAS_metallicites[i], metallicities)] / p_draw_metallicity * COMPAS_weights[i]

            #if i==0:
                #print("The binary metallicity is", COMPAS_metallicites[i])
                #print("min max Z", min(metallicities), max(metallicities))
                #print("min max COMPAS Z", min(COMPAS_metallicites), max(COMPAS_metallicites))
                #print("np digitize", np.digitize(COMPAS_metallicites[i], metallicities))
                #print("metallicities", metallicities[np.digitize(COMPAS_metallicites[i], metallicities)], metallicities, len(metallicities))
                #print("len dPdlogZ", (dPdlogZ[:, np.digitize(COMPAS_metallicites[i], metallicities)]).shape)
                #print("dPdlogZ", (dPdlogZ[:, np.digitize(COMPAS_metallicites[i], metallicities)]))

                #fig, ax = plt.subplots(figsize = (12,8))
                #plt.plot(redshifts, formation_rate[i])
                #plt.xlim(0, 10)
                #plt.ylabel("Formation rate", fontsize=20)
                #plt.xlabel("Redshift", fontsize=20)
                #plt.title('Z = %s'%COMPAS_metallicites[i])
                #plt.savefig('/home/alevina1/figures/formationrate_model.png')
                #plt.clf()

                #fig, ax = plt.subplots(figsize = (12,8))
                #plt.plot(redshifts, dPdlogZ[:, np.digitize(COMPAS_metallicites[i], metallicities)])
                #plt.xlim(0, 10)
                #plt.ylabel("dP/dlogZ", fontsize=20)
                #plt.xlabel("Redshift", fontsize=20)
                #plt.title('Z = %s'%COMPAS_metallicites[i])
                #plt.savefig('/home/alevina1/figures/dPdlogZ_model.png')
                #plt.clf()

            #with open("COMPASbinaryindex_model.txt", 'w') as f:
                #f.write("%s"%i)
        
        # calculate the time at which the binary formed if it merges at this redshift
        time_of_formation = times - COMPAS_delay_times[i]

        # we have only calculated formation rate up to z=max(redshifts), so we need to only find merger rates for formation times at z<max(redshifts)
        # first locate the index above which the binary would have formed before z=max(redshifts)
        first_too_early_index = np.digitize(age_first_sfr, time_of_formation)

        # include the whole array if digitize returns end of array and subtract one so we don't include the time past the limit
        first_too_early_index = first_too_early_index + 1 if first_too_early_index == n_redshifts else first_too_early_index      

        # as long as that doesn't preclude the whole range
        if first_too_early_index > 0:
            # work out the redshift at the time of formation
            z_of_formation = times_to_redshifts(time_of_formation[:first_too_early_index - 1])

            # calculate which index in the redshift array these redshifts correspond to
            z_of_formation_index = np.ceil(z_of_formation / redshift_step).astype(int)

            # set the merger rate at z (with z<10) to the formation rate at z_form
            merger_rate[i, :first_too_early_index - 1] = formation_rate[i, z_of_formation_index]

    return formation_rate, merger_rate

def compute_snr_and_detection_grids(sensitivity="O1", snr_threshold=8.0, Mc_max=300.0, Mc_step=0.1,
                                    eta_max=0.25, eta_step=0.01, snr_max=1000.0, snr_step=0.1):
    """
        Compute a grid of SNRs and detection probabilities for a range of masses and SNRs

        These grids are computed to allow for interpolating the values of the snr and detection probability. This function
        combined with find_detection_probability() could be replaced by something like
            for i in range(n_binaries):
                detection_probability = selection_effects.detection_probability(COMPAS.mass1[i],COMPAS.mass2[i],
                                            redshifts, distances, GWdetector_snr_threshold, GWdetector_sensitivity)
        if runtime was not important.

        Args:
            sensitivity                    --> [string]         Which detector sensitivity to use: one of ["design", "O1", "O3"]
            snr_threshold                  --> [float]          What SNR threshold required for a detection
            Mc_max                         --> [float]          Maximum chirp mass in grid
            Mc_step                        --> [float]          Step in chirp mass to use in grid
            eta_max                        --> [float]          Maximum symmetric mass ratio in grid
            eta_step                       --> [float]          Step in symmetric mass ratio to use in grid
            snr_max                        --> [float]          Maximum snr in grid
            snr_step                       --> [float]          Step in snr to use in grid

        Returns:
            snr_grid_at_1Mpc               --> [2D float array] The snr of a binary with masses (Mc, eta) at a distance of 1 Mpc
            detection_probability_from_snr --> [list of floats] A list of detection probabilities for different SNRs
    """
    # get interpolator given sensitivity
    interpolator = selection_effects.SNRinterpolator(sensitivity)

    # create chirp mass and eta arrays
    Mc_array = np.arange(Mc_step, Mc_max + Mc_step, Mc_step)
    eta_array = np.arange(eta_step, eta_max + eta_step, eta_step)

    # convert to total, primary and secondary mass arrays
    Mt_array = Mc_array / eta_array[:,np.newaxis]**0.6
    M1_array = Mt_array * 0.5 * (1. + np.sqrt(1. - 4 * eta_array[:,np.newaxis]))
    M2_array = Mt_array - M1_array

    # interpolate to get snr values if binary was at 1Mpc
    snr_grid_at_1Mpc = interpolator(M1_array, M2_array)

    # precompute a grid of detection probabilities as a function of snr
    snr_array = np.arange(snr_step, snr_max + snr_step, snr_step)
    detection_probability_from_snr = selection_effects.detection_probability_from_snr(snr_array, snr_threshold)

    return snr_grid_at_1Mpc, detection_probability_from_snr

def find_detection_probability(Mc, eta, redshifts, distances, n_redshifts_detection, n_binaries, snr_grid_at_1Mpc, detection_probability_from_snr,
                                Mc_step=0.1, eta_step=0.01, snr_step=0.1):
    """
        Compute the detection probability given a grid of SNRs and detection probabilities with masses

        Args:
            Mc                             --> [list of floats] Chirp mass of binaries in COMPAS
            eta                            --> [list of floats] Symmetric mass ratios of binaries in COMPAS
            redshifts                      --> [list of floats] List of redshifts
            distances                      --> [list of floats] List of distances corresponding to redshifts
            n_redshifts_detection          --> [int]            Index (in redshifts) to which we evaluate detection probability
            n_binaries                     --> [int]            Number of merging binaries in the COMPAS file
            snr_grid_at_1Mpc               --> [2D float array] The snr of a binary with masses (Mc, eta) at a distance of 1 Mpc
            detection_probability_from_snr --> [list of floats] A list of detection probabilities for different SNRs
            Mc_step                        --> [float]          Step in chirp mass to use in grid
            eta_step                       --> [float]          Step in symmetric mass ratio to use in grid
            snr_step                       --> [float]          Step in snr to use in grid
    """
    # by default, set detection probability to one
    detection_probability = np.ones(shape=(n_binaries, n_redshifts_detection))

    # for each binary in the COMPAS file
    for i in range(n_binaries):
        # shift frames for the chirp mass
        Mc_shifted = Mc[i] * (1 + redshifts[:n_redshifts_detection])

        # work out the closest index to the given values of eta and Mc
        eta_index = np.round(eta[i] / eta_step).astype(int) - 1
        Mc_index = np.round(Mc_shifted / Mc_step).astype(int) - 1

        # lookup values for the snr (but make sure you don't go over the top of the array)
        snrs = np.ones(n_redshifts_detection) * 0.00001
        Mc_below_max = Mc_index < snr_grid_at_1Mpc.shape[1]
        snrs[Mc_below_max] = snr_grid_at_1Mpc[eta_index, Mc_index[Mc_below_max]]

        # convert these snr values to the correct distances
        snrs = snrs / distances[:n_redshifts_detection]

        # lookup values for the detection probability (but make sure you don't go over the top of the array)
        detection_list_index = np.round(snrs / snr_step).astype(int) - 1
        snr_below_max = detection_list_index < len(detection_probability_from_snr)
        snr_below_min = detection_list_index < 0

        # remember we set probability = 1 by default? Because if we don't set it here, we have snr > max snr
        # which is 1000 by default, meaning very detectable
        detection_probability[i, snr_below_max] = detection_probability_from_snr[detection_list_index[snr_below_max]]
        #on the other hand, if SNR is too low, the detection probability is effectively zero
        detection_probability[i, snr_below_min] = 0

    return detection_probability

def find_detection_rate(path, filename="COMPAS_Output.h5", dco_type="BBH", weight_column=None,
                        merges_hubble_time=True, pessimistic_CEE=True, no_RLOF_after_CEE=True,
                        max_redshift=10.0, max_redshift_detection=1.0, redshift_step=0.001, z_first_SF = 10,
                        m1_min=5 * u.Msun, m1_max=150 * u.Msun, m2_min=0.1 * u.Msun, fbin=0.7,
                        aSF = 0.01, bSF = 2.77, cSF = 2.90, dSF = 4.70,
                        mu0=0.035, muz=-0.23, sigma0=0.39,sigmaz=0., alpha=0.0, 
                        min_logZ=-12.0, max_logZ=0.0, step_logZ=0.01,
                        sensitivity="O1", snr_threshold=8, 
                        Mc_max=300.0, Mc_step=0.1, eta_max=0.25, eta_step=0.01,
                        snr_max=1000.0, snr_step=0.1, data_fname=None):
    """
        The main function of this file. Finds the detection rate, formation rate and merger rate for each
        binary in a COMPAS file at a series of redshifts defined by intput. Also returns relevant COMPAS
        data.

        NOTE: This code assumes that assumes that metallicities in COMPAS are drawn from a flat in log distribution

        Args:
            ===================================================
            == Arguments for finding and masking COMPAS file ==
            ===================================================
            path                   --> [string] Path to the COMPAS file that contains the output
            filename               --> [string] Name of the COMPAS file
            dco_type               --> [string] Which DCO type to calculate rates for: one of ["all", "BBH", "BHNS", "BNS"]
            weight_column          --> [string] Name of column in "DoubleCompactObjects" file that contains adaptive sampling weights
                                                    (Leave this as None if you have unweighted samples)
            merges_in_hubble_time  --> [bool]   whether to mask binaries that don't merge in a Hubble time
            no_RLOF_after_CEE      --> [bool]   whether to mask binaries that have immediate RLOF after a CCE
            pessimistic_CEE        --> [bool]   whether to mask binaries that go through Optimistic CE scenario

            ===========================================
            == Arguments for creating redshift array ==
            ===========================================
            max_redshift           --> [float]  Maximum redshift to use in array
            max_redshift_detection --> [float]  Maximum redshift to calculate detection rates (must be <= max_redshift)
            redshift_step          --> [float]  Size of step to take in redshift

            ====================================================================
            == Arguments for determining star forming mass per sampled binary ==
            ====================================================================
            m1_min                 --> [float]  Minimum primary mass sampled by COMPAS
            m1_max                 --> [float]  Maximum primary mass sampled by COMPAS
            m2_min                 --> [float]  Minimum secondary mass sampled by COMPAS
            fbin                   --> [float]  Binary fraction used by COMPAS

            =======================================================================
            == Arguments for creating metallicity distribution and probabilities ==
            =======================================================================
            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)
            min_logZ               --> [float]  Minimum logZ at which to calculate dPdlogZ
            max_logZ               --> [float]  Maximum logZ at which to calculate dPdlogZ
            step_logZ              --> [float]  Size of logZ steps to take in finding a Z range

            =======================================================
            == Arguments for determining detection probabilities ==
            =======================================================
            sensitivity            --> [string] Which detector sensitivity to use: one of ["design", "O1", "O3"]
            snr_threshold          --> [float]  What SNR threshold required for a detection
            Mc_max                 --> [float]  Maximum chirp mass in grid
            Mc_step                --> [float]  Step in chirp mass to use in grid
            eta_max                --> [float]  Maximum symmetric mass ratio in grid
            eta_step               --> [float]  Step in symmetric mass ratio to use in grid
            snr_max                --> [float]  Maximum snr in grid
            snr_step               --> [float]  Step in snr to use in grid

        Returns:
            detection_rate         --> [2D float array] Detection rate for each binary at each redshift in 1/yr
            formation_rate         --> [2D float array] Formation rate for each binary at each redshift in 1/yr/Gpc^3
            merger_rate            --> [2D float array] Merger rate for each binary at each redshift in 1/yr/Gpc^3
            redshifts              --> [list of floats] List of redshifts
            COMPAS                 --> [Object]         Relevant COMPAS data in COMPASData Class
    """
    print("Running find_detection_rate with the following parameters: path {}, filename {}, dco_type{}, weight_column {}".format(path, filename, dco_type, weight_column)+\
            "\nmerges_hubble_time {}, pessimistic_CEE {}, no_RLOF_after_CE {},".format(merges_hubble_time, pessimistic_CEE, no_RLOF_after_CEE)+\
            "\nmax_redshift {}, max_redshift_detection {}, redshift_step {}, z_first_SF {},".format(max_redshift, max_redshift_detection, redshift_step, z_first_SF)+\
            "\nm1_min {}, m1_max {}, m2_min {}, fbin {},".format(m1_min, m1_max, m2_min, fbin)+\
            "\naSF {}, bSF {}, cSF {}, dSF {}, mu0 {}, muz {}, sigma0 {}, sigmaz {}, alpha {}, min_logZ {}, max_logZ {}, step_logZ {}".format(aSF, bSF, cSF, dSF, mu0, muz, sigma0,sigmaz, alpha, min_logZ, max_logZ, step_logZ)+\
            "\nsensitivity {}, snr_threshold {}, Mc_max {}, Mc_step {}, eta_max {}, eta_step {}, snr_max {}, snr_step {}".format(sensitivity, snr_threshold, Mc_max, Mc_step, eta_max, eta_step,snr_max, snr_step))

    # assert that input will not produce errors
    assert max_redshift_detection <= max_redshift, "Maximum detection redshift cannot be below maximum redshift"
    assert m1_min <= m1_max, "Minimum sampled primary mass cannot be above maximum sampled primary mass"
    assert np.logical_and(fbin >= 0.0, fbin <= 1.0), "Binary fraction must be between 0 and 1"
    assert Mc_step < Mc_max, "Chirp mass step size must be less than maximum chirp mass"
    assert eta_step < eta_max, "Symmetric mass ratio step size must be less than maximum symmetric mass ratio"
    assert snr_step < snr_max, "SNR step size must be less than maximum SNR"
    
    nonnegative_args = [(max_redshift, "max_redshift"), (max_redshift_detection, "max_redshift_detection"), (m1_min.value, "m1_min"), (m1_max.value, "m1_max"),
                        (m2_min.value, "m2_min"), (mu0, "mu0"), (sigma0, "sigma0"),  
                        (step_logZ, "step_logZ"), (snr_threshold, "snr_threshold"), (Mc_max, "Mc_max"),
                        (Mc_step, "Mc_step"), (eta_max, "eta_max"), (eta_step, "eta_step"), (snr_max, "snr_max"), (snr_step, "snr_step")]


    for arg, arg_str in nonnegative_args:
        assert arg >= 0.0, "{} must be nonnegative".format(arg_str)

    # warn if input is not advisable
    if redshift_step > max_redshift_detection:
        warnings.warn("Redshift step is greater than maximum detection redshift", stacklevel=2)
    if Mc_step > 1.0:
        warnings.warn("Chirp mass step is greater than 1.0, large step sizes can produce unpredictable results", stacklevel=2)
    if eta_step > 0.1:
        warnings.warn("Symmetric mass ratio step is greater than 0.1, large step sizes can produce unpredictable results", stacklevel=2)
    if snr_step > 1.0:
        warnings.warn("SNR step is greater than 1.0, large step sizes can produce unpredictable results", stacklevel=2)
    
    # start by getting the necessary data from the COMPAS file
    COMPAS = ClassCOMPAS.COMPASData(path, fileName=filename, Mlower=m1_min, Mupper=m1_max, m2_min=m2_min, binaryFraction=fbin, suppress_reminder=True)
    COMPAS.setCOMPASDCOmask(types=dco_type, withinHubbleTime=merges_hubble_time, pessimistic=pessimistic_CEE, noRLOFafterCEE=no_RLOF_after_CEE)
    COMPAS.setCOMPASData()
    COMPAS.set_sw_weights(weight_column)
    COMPAS.find_star_forming_mass_per_binary_sampling()

    assert np.log(np.min(COMPAS.initialZ)) != np.log(np.max(COMPAS.initialZ)), "You cannot perform cosmic integration with just one metallicity"

    # compute the chirp masses and symmetric mass ratios only for systems of interest
    chirp_masses = (COMPAS.mass1*COMPAS.mass2)**(3/5) / (COMPAS.mass1 + COMPAS.mass2)**(1/5)
    etas = COMPAS.mass1 * COMPAS.mass2 / (COMPAS.mass1 + COMPAS.mass2)**2
    n_binaries = len(chirp_masses)
    # another warning on poor input
    if max(chirp_masses)*(1+max_redshift_detection) < Mc_max:
        warnings.warn("Maximum chirp mass used for detectability calculation is below maximum binary chirp mass * (1+maximum redshift for detectability calculation)", stacklevel=2)

       
    # work out the metallicity distribution at each redshift and probability of drawing each metallicity in COMPAS
    if data_fname:
        # read in metallicity data
        Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals = readTNGdata(data_fname)

        # interpolate metallicity data
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new, Metalsnew, step_fit_logZ_new = interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, Metals, redshiftlimandstep=[0, max_redshift+redshift_step, redshift_step])
        
        # calculate the redshifts array and its equivalents
        redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes = calculate_redshift_related_params(max_redshift, max_redshift_detection, redshift_step, z_first_SF)

        # find the star forming mass per year per Gpc^3 and convert to total number formed per year per Gpc^3
        sfr = np.sum(SFRDnew, axis=0)*step_fit_logZ_new

        # Calculate the representative SF mass
        Average_SF_mass_needed = (COMPAS.mass_evolved_per_binary * COMPAS.n_systems)
        print('Average_SF_mass_needed = ', Average_SF_mass_needed) # print this, because it might come in handy to know when writing up results :)
        n_formed = sfr / Average_SF_mass_needed # Divide the star formation rate density by the representative SF mass

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, sfr)
        #plt.ylabel("SFRD(z)", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/SFR_data.png')
        #plt.clf()
        
        # assume a flat in log distribution in metallicity to find probability of drawing Z in COMPAS
        min_logZ_COMPAS = np.log(np.min(COMPAS.initialZ))
        max_logZ_COMPAS = np.log(np.max(COMPAS.initialZ))
        p_draw_metallicity = 1 / (max_logZ_COMPAS - min_logZ_COMPAS)
        
        print("Calculating rates using data")

        # calculate the formation and merger rates using what we computed above
        formation_rate, merger_rate = find_formation_and_merger_rates(n_binaries, redshifts, times, time_first_SF, metals_new, p_draw_metallicity,
                                                                     COMPAS.metallicitySystems, COMPAS.delayTimes, COMPAS.sw_weights, 
                                                                     Metal_distributions=Metalsnew, n_formed=n_formed)
        
        #print(formation_rate.shape, merger_rate.shape)

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, np.sum(formation_rate, axis=0))
        #plt.ylabel("Formation rate", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/full_formationrate_data.png')
        #plt.clf()

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, np.sum(merger_rate, axis=0))
        #plt.ylabel("Merger rate", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/mergerrate_data.png')
        #plt.clf()
        
    else:

        # calculate the redshifts array and its equivalents
        redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes = calculate_redshift_related_params(max_redshift, max_redshift_detection, redshift_step, z_first_SF)

        # find the star forming mass per year per Gpc^3 and convert to total number formed per year per Gpc^3
        sfr = find_sfr(redshifts, a = aSF, b = bSF, c = cSF, d = dSF) # functional form from Madau & Dickinson 2014

        # Calculate the representative SF mass
        Average_SF_mass_needed = (COMPAS.mass_evolved_per_binary * COMPAS.n_systems)
        print('Average_SF_mass_needed = ', Average_SF_mass_needed) # print this, because it might come in handy to know when writing up results :)
        n_formed = sfr / Average_SF_mass_needed # Divide the star formation rate density by the representative SF mass

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, sfr)
        #plt.ylabel("SFRD(z)", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/SFR_model.png')
        #plt.clf()
        
        dPdlogZ, metallicities, p_draw_metallicity = find_metallicity_distribution(redshifts, min_logZ_COMPAS = np.log(np.min(COMPAS.initialZ)),
                                                                                max_logZ_COMPAS = np.log(np.max(COMPAS.initialZ)),
                                                                                mu0=mu0, muz=muz, sigma_0=sigma0, sigma_z=sigmaz, alpha = alpha,
                                                                                min_logZ=min_logZ, max_logZ=max_logZ, step_logZ = step_logZ)

        print("Calculating rates using model")

        # calculate the formation and merger rates using what we computed above
        formation_rate, merger_rate = find_formation_and_merger_rates(n_binaries, redshifts, times, time_first_SF,
                                                                    metallicities, p_draw_metallicity, COMPAS.metallicitySystems,
                                                                    COMPAS.delayTimes, COMPAS.sw_weights, n_formed= n_formed, dPdlogZ=dPdlogZ)

        #print(formation_rate.shape, merger_rate.shape)

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, np.sum(formation_rate, axis=0))
        #plt.ylabel("Formation rate", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/full_formationrate_model.png')
        #plt.clf()

        #fig, ax = plt.subplots(figsize = (12,8))
        #plt.plot(redshifts, np.sum(merger_rate, axis=0))
        #plt.ylabel("Merger rate", fontsize=20)
        #plt.xlabel("Redshift", fontsize=20)
        #plt.savefig('/home/alevina1/figures/mergerrate_model.png')
        #plt.clf()

    print("Calculating snr")
    # create lookup tables for the SNR at 1Mpc as a function of the masses and the probability of detection as a function of SNR
    snr_grid_at_1Mpc, detection_probability_from_snr = compute_snr_and_detection_grids(sensitivity, snr_threshold, Mc_max, Mc_step,
                                                                                    eta_max, eta_step, snr_max, snr_step)

    print("calculating detection probability")
    # use lookup tables to find the probability of detecting each binary at each redshift
    detection_probability = find_detection_probability(chirp_masses, etas, redshifts, distances, n_redshifts_detection, n_binaries,
                                                        snr_grid_at_1Mpc, detection_probability_from_snr, Mc_step, eta_step, snr_step)

    print("computing detection rate")
    # finally, compute the detection rate using Neijssel+19 Eq. 2
    detection_rate = np.zeros(shape=(n_binaries, n_redshifts_detection))
    detection_rate = merger_rate[:, :n_redshifts_detection] * detection_probability \
                    * shell_volumes[:n_redshifts_detection] / (1 + redshifts[:n_redshifts_detection])
    
    print("rates calculated")

    return detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes


def append_rates(path, outfilename, detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes, n_redshifts_detection,
    maxz=5., sensitivity="O1", dco_type="BHBH", mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0., aSF = 0.01, bSF = 2.77, cSF = 2.90, dSF = 4.70,
    append_binned_by_z = False, redshift_binsize=0.1):
    """
        Append the formation rate, merger rate, detection rate and redshifts as a new group to your COMPAS output with weights hdf5 file

        Args:
            path                   --> [string] Path to the COMPAS file that contains the output
            outfilename            --> [string] Name of the hdf5 file that you want to write your rates to
            detection_rate         --> [2D float array] Detection rate for each binary at each redshift in 1/yr
            formation_rate         --> [2D float array] Formation rate for each binary at each redshift in 1/yr/Gpc^3
            merger_rate            --> [2D float array] Merger rate for each binary at each redshift in 1/yr/Gpc^3
            redshifts              --> [list of floats] List of redshifts
            COMPAS                 --> [Object]         Relevant COMPAS data in COMPASData Class
            Average_SF_mass_needed --> [float]          How much star forming mass your simulation represents on average
            shell_volumes          --> [list of floats] Equivalent of redshifts but converted to shell volumes
            n_redshifts_detection  --> [int]            Number of redshifts in list that should be used to calculate detection rates

            maxz                   --> [float] Maximum redshhift up to where we would like to store the data
            sensitivity            --> [string] Which detector sensitivity you used to calculate rates 
            dco_type               --> [string] Which DCO type you used to calculate rates 
            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)

            append_binned_by_z     --> [Bool] to save space, bin rates by redshiftbin and append binned rates
            redshift_binsize       --> [float] if append_binned_by_z, how big should your redshift bin be

        Returns:
            h_new                  --> [hdf5 file] Compas output file with a new group "rates" with the same shape as DoubleCompactObjects x redshifts
    """
    print('\nIn append rates: shape redshifts', np.shape(redshifts))
    print('shape COMPAS.sw_weights', np.shape(COMPAS.sw_weights) )
    print('COMPAS.DCOmask', COMPAS.DCOmask, ' was set for dco_type', dco_type )
    print('shape COMPAS COMPAS.DCOmask', np.shape(COMPAS.DCOmask), ' sums to ', np.sum(COMPAS.DCOmask) )
    print('path', path)

    #################################################
    #Open hdf5 file that we will read from
    print('path', path)
    with h5.File(path , 'r') as f_COMPAS:
        
        # Would you like to write your rates to a different file? 
        if path == outfilename:
            raise ValueError('you cant append directly to the input data, will change outfilename to %s'%(outfilename)+'_1')
            outfilename = outfilename+'_1'

        #'you want to save your output to a different file!'
        if os.path.exists(outfilename):
            print('file', outfilename, 'exists!! You will remove it')
            os.remove(outfilename)
            
        print('writing to ', outfilename)
        h_new = h5.File(outfilename, 'w')

        # The rate info is shaped as BSE_Double_Compact_Objects[COMPAS.DCOmask] , len(redshifts)
        try: 
            DCO             = f_COMPAS['BSE_Double_Compact_Objects']#
        except:
            DCO             = f_COMPAS['DoubleCompactObjects']#

        #################################################
        # Create a new group where we will store data
        new_rate_group = 'Rates_mu0{}_muz{}_alpha{}_sigma0{}_sigmaz{}_a{}_b{}_c{}_d{}'.format(mu0, muz, alpha, sigma0, sigmaz, aSF, bSF, cSF, dSF)

        if append_binned_by_z:
            new_rate_group  = new_rate_group + '_zBinned'

        if new_rate_group not in h_new:
            h_new.create_group(new_rate_group)
        else:
            print(new_rate_group, 'exists, we will overrwrite the data')


        #################################################
        # Bin rates by redshifts
        #################################################
        if append_binned_by_z:
            # Choose how you want to bin the redshift, these represent the left and right boundaries
            redshift_bins = np.arange(0, redshifts[-1]+redshift_binsize, redshift_binsize)
            print('new crude redshift_bins', redshift_bins)
            print('old fine redshifts', redshifts)
            fine_binsize    = np.diff(redshifts)[0] #Assunming your redshift bins are equally spaced!!
            print('fine_binsize', fine_binsize)
            #Assuming your crude redshift bin is made up of an integer number of fine z-bins!!!
            i_per_crude_bin = redshift_binsize/fine_binsize 
            print('i_per_crude_bin', i_per_crude_bin)
            i_per_crude_bin = int(i_per_crude_bin)

            ###################
            # convert crude redshift bins to volumnes and ensure all volumes are in Gpc^3
            crude_volumes = cosmology.comoving_volume(redshift_bins).to(u.Gpc**3).value
            # split volumes into shells and duplicate last shell to keep same length
            crude_shell_volumes    = np.diff(crude_volumes)
            # crude_shell_volumes    = np.append(crude_shell_volumes, crude_shell_volumes[-1])

            ###################
            # convert redshifts to volumnes and ensure all volumes are in Gpc^3
            fine_volumes       = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value
            fine_shell_volumes = np.diff(fine_volumes)
            fine_shell_volumes = np.append(fine_shell_volumes, fine_shell_volumes[-1])

            # Use digitize to assign the redshifts to a bin (detection list is shorter)
            # digitized     = np.digitize(redshifts, redshift_bins)
            digitized_det = np.digitize(redshifts[:n_redshifts_detection], redshift_bins)

            # Convert your merger_rate back to 1/yr by multiplying by the fine_shell_volumes
            N_dco_in_z_bin      = (merger_rate[:,:] * fine_shell_volumes[:])
            N_dco_in_z_bin_form   = (formation_rate[:,:] * fine_shell_volumes[:])
            print('fine_shell_volumes', fine_shell_volumes)

            # The number of merging BBHs that need a weight
            N_dco  = len(merger_rate[:,0])
            
            ####################
            # binned_merger_rate will be the (observed) weights, binned by redshhift
            binned_merger_rate    = np.zeros( (N_dco, len(redshift_bins)-1) )# create an empty list to fill
            binned_detection_rate = np.zeros( (N_dco, len(redshift_bins)-1) )# create an empty list to fill
            binned_formation_rate = np.zeros( (N_dco, len(redshift_bins)-1) )# create an empty list to fill

            # loop over all redshift redshift_bins
            for i in range(len(redshift_bins)-1):
                # print('redshifts[Bool_list[i]]', redshifts[Bool_list[i]])
                print('redshifts[[i*i_per_crude_bin:(i+1)*i_per_crude_bin]]', redshifts[i*i_per_crude_bin:(i+1)*i_per_crude_bin])

                # Sum the number of mergers per year, and divide by the new dz volume to get a density
                binned_merger_rate[:,i] = np.sum(N_dco_in_z_bin[:,i*i_per_crude_bin:(i+1)*i_per_crude_bin], axis = 1)/crude_shell_volumes[i]
                binned_formation_rate[:,i] = np.sum(N_dco_in_z_bin_form[:,i*i_per_crude_bin:(i+1)*i_per_crude_bin], axis = 1)/crude_shell_volumes[i]

                # only add detected rates for the 'detectable' redshifts
                if redshift_bins[i] < redshifts[n_redshifts_detection]:
                    # The detection rate was already multiplied by the shell volumes, so we can sum it directly
                    binned_detection_rate[:,i] = np.sum(detection_rate[:, digitized_det == i+1], axis = 1)

            #  To avoid huge filesizes, we don't really wan't All the data, 
            # so we're going to save up to some redshift
            z_index = np.digitize(maxz, redshift_bins) -1

            # The detection_rate is a smaller array, make sure you don't go beyond the end
            detection_index = z_index if z_index < n_redshifts_detection else n_redshifts_detection
            
            save_redshifts        = redshift_bins[:z_index]
            save_merger_rate      = binned_merger_rate[:,:z_index]
            save_formation_rate      = binned_formation_rate[:,:z_index]
            # save_detection_rate   = binned_detection_rate[:,:detection_index]

        else: 
            #  To avoid huge filesizes, we don't really wan't All the data, 
            # so we're going to save up to some redshift
            z_index = np.digitize(maxz, redshifts) -1

            # The detection_rate is a smaller array, make sure you don't go beyond the end
            detection_index = z_index if z_index < n_redshifts_detection else n_redshifts_detection

            print('You will only save data up to redshift ', maxz, ', i.e. index', z_index)
            save_redshifts        = redshifts[:z_index]
            save_merger_rate      = merger_rate[:,:z_index]
            save_formation_rate      = formation_rate[:,:z_index]
            # save_detection_rate   = detection_rate[:,:detection_index]

        print('save_redshifts', save_redshifts)
        print('shape of save_merger_rate ', np.shape(save_merger_rate))

        #################################################
        # Write the rates as a seperate dataset
        # re-arrange your list of rate parameters
        DCO_to_rate_mask     = COMPAS.DCOmask #save this bool for easy conversion between BSE_Double_Compact_Objects, and CI weights
        rate_data_list       = [DCO['SEED'][DCO_to_rate_mask], DCO_to_rate_mask , save_redshifts,  save_merger_rate, save_formation_rate]
        #, merger_rate[:,0], save_detection_rate, Average_SF_mass_needed]
        rate_list_names      = ['SEED', 'DCOmask', 'redshifts', 'merger_rate', 'formation_rate']
        #,'merger_rate_z0', 'detection_rate'+sensitivity, 'Average_SF_mass_needed']
        for i, data in enumerate(rate_data_list):
            print('Adding rate info {} of shape {}'.format(rate_list_names[i], np.shape(data)) )
            # Check if dataset exists, if so, just delete it
            if rate_list_names[i] in h_new[new_rate_group].keys():
                del h_new[new_rate_group][rate_list_names[i]]
            # write rates as a new data set
            dataNew     = h_new[new_rate_group].create_dataset(rate_list_names[i], data=data)

    #Always close your files again ;)
    h_new.close()
    print( ('Done with append_rates :) your new files are here: %s'%(outfilename)).replace('//', '/') )



def delete_rates(path, filename, mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0., append_binned_by_z=False):
    """
        Delete the group containing all the rate information from your COMPAS output with weights hdf5 file


        Args:
            path                   --> [string] Path to the COMPAS file that contains the output
            filename               --> [string] Name of the COMPAS file

            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)
            append_binned_by_z     --> [Bool] to save space, bin rates by redshiftbin and append binned rates

    """
    #################################################
    #Open hdf5 file that we will write on
    print('filename', filename)
    with h5.File(path +'/'+ filename, 'r+') as h_new:
        # The rate info is shaped as BSE_Double_Compact_Objects[COMPAS.DCOmask] , len(redshifts)
        try:
            DCO             = h_new['BSE_Double_Compact_Objects']#
        except:
            DCO             = h_new['DoubleCompactObjects']#

        #################################################
        # Name of the group that has the data stored
        new_rate_group = 'Rates_mu0{}_muz{}_alpha{}_sigma0{}_sigmaz{}_a{}'.format(mu0, muz, alpha, sigma0, sigmaz)
        if append_binned_by_z:
            new_rate_group  = new_rate_group + '_zBinned'

        if new_rate_group not in h_new:
            print(new_rate_group, 'Does not exist, nothing to do here...')
            #Always close your files again ;)
            h_new.close()
            return
        else:
            print('You want to remove this group, %s, from the hdf5 file, removing now..'%(new_rate_group))
            del h_new[new_rate_group]
            #Always close your files again ;)
            h_new.close()
            print('Done with delete_rates :) your files are here: ', path + '/' + filename )
            return




def plot_rates(save_dir, formation_rate, merger_rate, detection_rate, redshifts, chirp_masses, show_plot = False, mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0,aSF = 0.02,  bSF = 1.48, cSF = 4.45, dSF = 5.91):
    """
        Show a summary plot of the results, it also returns the summaries that it computes

        Args:
            save_dir                  --> [string] path where you would like to save your plot
            formation_rate            --> [2D float array] Formation rate for each binary at each redshift in 1/yr/Gpc^3
            merger_rate               --> [2D float array] Merger rate for each binary at each redshift in 1/yr/Gpc^3
            detection_rate            --> [2D float array] Detection rate for each binary at each redshift in 1/yr
            redshifts                 --> [list of floats] List of redshifts
            chirp_masses              --> [list of floats] Chrirp masses of merging DCO's

            show_plot                 --> [bool] Bool whether to show plot or not
            mu0                       --> [float]  metallicity dist: expected value at redshift 0
            muz                       --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                    --> [float]  metallicity dist: width at redshhift 0
            sigmaz                    --> [float]  metallicity dist: redshift evolution of width
            alpha                     --> [float]  metallicity dist: skewness (0 = lognormal)

        Returns:
            matplotlib figure

    """
    # sum things up across binaries
    total_formation_rate = np.sum(formation_rate, axis=0)
    total_merger_rate = np.sum(merger_rate, axis=0)
    total_detection_rate = np.sum(detection_rate, axis=0)
    
    # and across redshifts
    cumulative_detection_rate = np.cumsum(total_detection_rate)
    detection_rate_by_binary = np.sum(detection_rate, axis=1)

    ###########################
    #Start plotting

    # set some constants for the plots
    plt.rc('font', family='serif')
    fs = 20
    lw = 3

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes[0,0].plot(redshifts, total_formation_rate, lw=lw)
    axes[0,0].set_xlabel('Redshift', fontsize=fs)
    axes[0,0].set_ylabel(r'Formation rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=fs)

    axes[0,1].plot(redshifts, total_merger_rate, lw=lw)
    axes[0,1].set_xlabel('Redshift', fontsize=fs)
    axes[0,1].set_ylabel(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=fs)

    axes[1,0].plot(redshifts[:len(cumulative_detection_rate)], cumulative_detection_rate, lw=lw)
    axes[1,0].set_xlabel('Redshift', fontsize=fs)
    axes[1,0].set_ylabel(r'Cumulative detection rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}yr}]$', fontsize=fs)

    axes[1,1].hist(chirp_masses, weights=detection_rate_by_binary, bins=25, range=(0, 50))
    axes[1,1].set_xlabel(r'Chirp mass, $\mathcal{M}_c$', fontsize=fs)
    axes[1,1].set_ylabel(r'Mass distrbution of detections $[\rm \frac{\mathrm{d}N}{\mathrm{d}\mathcal{M}_c \mathrm{d}yr}]$', fontsize=fs)

    #########################
    #Plotvalues

    # Add text upper left corner
    axes[0,0].text(0.05,0.8, "mu0=%s \nmuz=%s \nsigma0=%s \nsigmaz=%s \nalpha=%s"%(mu0,muz,sigma0,sigmaz,alpha), transform=axes[0,0].transAxes, size = fs) 

    for ax in axes.flatten():
        ax.tick_params(labelsize=0.9*fs)

    print("Plotting!")
    # Save and show :)
    plt.savefig(save_dir +'/Rate_Info'+"mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s"%(mu0,muz,alpha,sigma0,sigmaz,aSF, bSF,cSF,dSF)+'.png', bbox_inches='tight') 
    if show_plot:
        plt.show()
    else:
        plt.close()




##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line options for the most commonly varied options
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", dest= 'path',  help="Path to the COMPAS file that contains the output",type=str, default = './')
    parser.add_argument("--filename", dest= 'fname',  help="Name of the COMPAS file",type=str, default = "COMPAS_Output.h5")
    parser.add_argument("--outfname", dest= 'outfname',  help="Name of the output file where you store the rates, default is append to COMPAS output",type=str, default = "COMPAS_Output.h5")
    parser.add_argument("--datafname", dest= 'data_fname', help="Name of file with metallicity and SFRD data", default = None)

    # For what DCO would you like the rate?  options: ALL, BHBH, BHNS NSNS
    parser.add_argument("--dco_type", dest= 'dco_type',  help="Which DCO type you used to calculate rates, one of: ['all', 'BBH', 'BHNS', 'BNS'] ",type=str, default = "BBH")
    parser.add_argument("--weight", dest= 'weight_column',  help="Name of column w AIS sampling weights, i.e. 'mixture_weight'(leave as None for unweighted samples) ",type=str, default = None)

    # Options for the redshift evolution and detector sensitivity
    parser.add_argument("--maxz", dest= 'max_redshift',  help="Maximum redshift to use in array",type=float, default=14)
    parser.add_argument("--zSF", dest= 'z_first_SF',  help="redshift of first star formation",type=float, default=10)
    parser.add_argument("--maxzdet", dest= 'max_redshift_detection',  help="Maximum redshift to calculate detection rates",type=float, default=1)
    parser.add_argument("--zstep", dest= 'redshift_step',  help="size of step to take in redshift",type=float, default=0.001)
    parser.add_argument("--sens", dest= 'sensitivity',  help="Which detector sensitivity to use: one of ['design', 'O1', 'O3']",type=str, default = "O3")
    parser.add_argument("--snr", dest= 'snr_threshold',  help="What SNR threshold required for a detection",type=float, default=8)

    # Parameters to calculate the representing SF mass (make sure these match YOUR simulation!)
    parser.add_argument("--m1min", dest= 'm1_min',  help="Minimum primary mass sampled by COMPAS",type=float, default=5.) 
    parser.add_argument("--m1max", dest= 'm1_max',  help="Maximum primary mass sampled by COMPAS",type=float, default=150.) 
    parser.add_argument("--m2min", dest= 'm2_min',  help="Minimum secondary mass sampled by COMPAS",type=float, default=0.1) 
    parser.add_argument("--fbin", dest= 'fbin',  help="Binary fraction used by COMPAS",type=float, default=0.7) 

    # Parameters determining dP/dZ and SFR(z), default options from Neijssel 2019
    parser.add_argument("--mu0", dest= 'mu0',  help="mean metallicity at redshhift 0",type=float, default=0.035)
    parser.add_argument("--muz", dest= 'muz',  help="redshift evolution of mean metallicity, dPdlogZ",type=float, default=-0.23)
    parser.add_argument("--sigma0", dest= 'sigma0',  help="variance in metallicity density distribution, dPdlogZ",type=float, default=0.39)
    parser.add_argument("--sigmaz", dest= 'sigmaz',  help="redshift evolution of variance, dPdlogZ",type=float, default=0.0)
    parser.add_argument("--alpha", dest= 'alpha',  help="skewness of mtallicity density distribution, dPdlogZ",type=float, default=0.0)
    parser.add_argument("--aSF", dest= 'aSF',  help="Parameter for shape of SFR(z)",type=float, default=0.01) 
    parser.add_argument("--bSF", dest= 'bSF',  help="Parameter for shape of SFR(z)",type=float, default=2.77)
    parser.add_argument("--cSF", dest= 'cSF',  help="Parameter for shape of SFR(z)",type=float, default=2.90)
    parser.add_argument("--dSF", dest= 'dSF',  help="Parameter for shape of SFR(z)",type=float, default=4.70)
 
     # Options for saving your data
    parser.add_argument("--dontAppend", dest= 'append_rates',  help="Prevent the script from appending your rates to the hdf5 file.", action='store_false', default=True)
    parser.add_argument("--BinAppend", dest= 'binned_rates',  help="Append your rates in more crude redshift bins to save space.", action='store_true', default=False)
    parser.add_argument("--redshiftBinSize", dest= 'zBinSize',  help="How big should the crude redshift bins be", type=float, default=0.05)
    parser.add_argument("--delete", dest= 'delete_rates',  help="Delete the rate group from your hdf5 output file (groupname based on dP/dZ parameters)", action='store_true', default=False)
    parser.add_argument("--cosmology", dest='Cosmology', help="Cosmology that is used for cosmic integration", type=str, default="Planck18")

    args = parser.parse_args()

    #####################################
    # Run the cosmic integration

    if args.Cosmology == "Planck18":
        print("USING PLANCK18 AS COSMOLOGY! if working with Illustris TNG data please use Planck15 instead")
    else:
        print("Using %s as cosmology!"%args.Cosmology)
    cosmology = getattr(importlib.import_module('astropy.cosmology'), args.Cosmology)

    print("Calculate detection rates")
    
    start_CI = time.time()
    detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes = find_detection_rate(args.path, filename=args.fname, dco_type=args.dco_type, weight_column=args.weight_column,
                            max_redshift=args.max_redshift, max_redshift_detection=args.max_redshift_detection, redshift_step=args.redshift_step, z_first_SF= args.z_first_SF,
                            m1_min=args.m1_min*u.Msun, m1_max=args.m1_max*u.Msun, m2_min=args.m2_min*u.Msun, fbin=args.fbin,
                            aSF = args.aSF, bSF = args.bSF, cSF = args.cSF, dSF = args.dSF, 
                            mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha, 
                            sensitivity=args.sensitivity, snr_threshold=args.snr_threshold, 
                            min_logZ=-12.0, max_logZ=0.0, step_logZ=0.01, Mc_max=300.0, Mc_step=0.1, eta_max=0.25, eta_step=0.01, snr_max=1000.0, snr_step=0.1, 
                            data_fname = args.data_fname)
    end_CI = time.time()
    
    print("rates calculated, now appending rates")

    #####################################
    # Append your freshly calculated merger rates to the hdf5 file
    
    print(args.append_rates)
    
    start_append = time.time()
    if args.append_rates:
        print("Appending rates!")
        n_redshifts_detection = int(args.max_redshift_detection / args.redshift_step)
        append_rates(args.path + '/' + args.fname, args.outfname, detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes, n_redshifts_detection,
            maxz=args.max_redshift_detection, sensitivity=args.sensitivity, dco_type=args.dco_type, mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha,
            aSF = args.aSF,  bSF = args.bSF , cSF = args.cSF , dSF = args.dSF ,
            append_binned_by_z = args.binned_rates, redshift_binsize=args.zBinSize)

    # or just delete this group if your hdf5 file is getting too big
    # !! Make sure to run h5repack from your terminal if you do this!! del doesn't actually free up space
    if args.delete_rates:
        delete_rates(args.path, args.fname, mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha, append_binned_by_z=False)

    end_append = time.time()

    print("Plotting now")
    
    #####################################
    # Plot your result
    start_plot = time.time()
    chirp_masses = (COMPAS.mass1*COMPAS.mass2)**(3./5.) / (COMPAS.mass1 + COMPAS.mass2)**(1./5.)
    print('almost finished, just plotting your results now')
    plot_rates(args.path, formation_rate, merger_rate, detection_rate, redshifts, chirp_masses, show_plot = False, mu0=args.mu0, muz=args.muz, sigma0=args.sigma0, sigmaz=args.sigmaz, alpha=args.alpha, aSF = args.aSF,  bSF = args.bSF , cSF = args.cSF , dSF = args.dSF ,)
    end_plot = time.time()

    print('CI took ', end_CI - start_CI, 's')
    print('Appending rates took ', end_append - start_append, 's')
    print('plot took ', end_plot - start_plot, 's')

