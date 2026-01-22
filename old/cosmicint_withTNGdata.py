import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn as sns
from colour import Color

from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo # Planck15 since that's what TNG uses

from scipy.stats import norm as NormDist

from Fit_model_TNG_SFRD import readTNGdata, interpolate_TNGdata
from TNG_BBHpop_properties import read_best_fits
import read_TNGdata as read

############################
# Custom scripts
sys.path.append('../')
import paths
from Fit_model_TNG_SFRD import readTNGdata

def calculate_redshift_related_params(max_redshift=10.0, max_redshift_detection=1.0, redshift_step=0.001, z_first_SF = 14.0):

    # create a list of redshifts and record lengths
    redshifts = np.arange(0, max_redshift + redshift_step, redshift_step)
    n_redshifts_detection = int(max_redshift_detection / redshift_step)

    # convert redshifts to times and ensure all times are in Myr
    times = cosmo.age(redshifts).to(u.Myr).value

    # and time of first Sf
    time_first_SF = cosmo.age(z_first_SF).to(u.Myr).value

    # convert redshifts to distances and ensure all distances are in Mpc (also avoid D=0 because division by 0)
    distances = cosmo.luminosity_distance(redshifts).to(u.Mpc).value
    distances[0] = 0.001

    # convert redshifts to volumnes and ensure all volumes are in Gpc^3
    volumes = cosmo.comoving_volume(redshifts).to(u.Gpc**3).value

    # split volumes into shells and duplicate last shell to keep same length
    shell_volumes = np.diff(volumes)
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])

    return redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes

def find_sfr(redshifts, a = 0.01, b =2.77, c = 2.90, d = 4.70):
    # get value in mass per year per cubic Mpc and convert to per cubic Gpc then return
    sfr = a * ((1+redshifts)**b) / (1 + ((1+redshifts)/c)**d) * u.Msun / u.yr / u.Mpc**3
    print(np.amax(sfr.value))
    return sfr.to(u.Msun / u.yr / u.Gpc**3).value


def find_metallicity_distribution(redshifts, metals = [], min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03),
                                  mu0=0.025, muz=-0.048, sigma_0=1.125, sigma_z=0.048, alpha =-1.767,
                                  min_logZ=-12.0, max_logZ=0.0, step_logZ =0.01):
    # Log-Linear redshift dependence of sigma
    sigma = sigma_0* 10**(sigma_z*redshifts)

    # Follow Langer & Norman 2007? in assuming that mean metallicities evolve in z as:
    mean_metallicities = mu0 * 10**(muz * redshifts)

    # Now we re-write the expected value of ou log-skew-normal to retrieve mu
    beta = alpha/(np.sqrt(1 + (alpha)**2))
    PHI  = NormDist.cdf(beta * sigma)
    mu_metallicities = np.log(mean_metallicities/(2.*PHI)) - (sigma**2)/2.
    
    if len(metals) == 0:
        # create a range of metallicities (the x-values, or random variables)
        log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
        metallicities     = np.exp(log_metallicities)
    else: 
        #use a pre-determined array of metals
        metallicities     = metals
        log_metallicities = np.log(metallicities)
        step_logZ         = np.diff(log_metallicities)[0]
        
    # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
    dPdlogZ = 2./(sigma[:,np.newaxis]) \
    * NormDist.pdf((log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis]) \
    * NormDist.cdf(alpha * (log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis] )

    # normalise the distribution over all metallicities
    norm = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ /norm[:,np.newaxis]

    # assume a flat in log distribution in metallicity to find probability of drawing Z in COMPAS
    p_draw_metallicity = 1 / (max_logZ_COMPAS - min_logZ_COMPAS)
    
    return dPdlogZ, metallicities, step_logZ, p_draw_metallicity


def calc_dPdlogZ(COMPAS_metallicity, metallicities, Metal_dist):
        
    #dPdlogZs = []
    #metals_form = []

    metallicity_ind = np.digitize(COMPAS_metallicity, metallicities)
    dPdlogZs =  Metal_dist[:, metallicity_ind]
    #dPdlogZs = metal_dist_form/np.trapz(metal_dist_form)

    """
    for z_ind, z in enumerate(redshifts):
        #convert everything to log metallicity
        log_binary_smetallicity = np.log10(binary_metallicity)
        log_data_metallicity = np.log10(data_metallicity)

        #find closest value in the metallicities in the data to the binary metallicity
        metallicity_ind = np.argmin(np.abs(log_data_metallicity - log_binary_metallicity)) #index
        metallicity_form = 10**log_data_metallicity[metallicity_ind] #value of closest metallicity

        #find the probability of getting that metallicity using the metallicity distribution at this redshift
        metal_dist_form = metallicity_dists[:, z_ind]
        norm_metal_dist_form = metal_dist_form/np.trapz(metal_dist_form)

        if showdist==True:
            plt.plot(log_data_metallicity, norm_metal_dist_form) #plot just to check
            plt.yscale('log')
            plt.title("z=%s, Z=%s"%(z, metallicity_form))
            plt.ylabel("PDF")
            plt.xlabel("logZ")
            plt.show()
            
        dPdlogZs.append(norm_metal_dist_form[metallicity_ind])
        metals_form.append(metallicity_form)
    """

    return dPdlogZs


def SFRfrom_TNGdata(path, tng, ver, rbox, fit_filename, showplot=True):

    #Method 1 of getting SFR: read in directly from data (inputs)
    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ = readTNGdata(loc = path, rbox=rbox, SFR=True)

    #Method 2: Interpolate 
    SFRDinterp, redshiftinterp, Lookbacktimesinterp, metalsinterp, step_fit_logZ_new = interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, ver, redshiftlimandstep=[0, 20, 0.05], saveplot=False)

    #Method 3: Model from best fit parameters (these are based on interpolated, method 2)
    best_fits = read_best_fits(fit_filename)[0]
    mu0, muz, omega0, omegaz, alpha0, sf_a, sf_b, sf_c, sf_d = best_fits
    sfr = find_sfr(redshiftinterp, a=sf_a, b=sf_b, c=sf_c, d=sf_d)


    #Try plotting all three ways of getting SFR from data, and also the model
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    plt.plot(redshiftinterp, np.sum(SFRDinterp*step_fit_logZ_new*mpc,axis=0), label='Interpolated TNG SFRD') #interpolated Sim_SFRD
    plt.plot(Redshifts, np.sum(Sim_SFRD*step_fit_logZ*mpc, axis=1), label='TNG SFRD') 
    plt.plot(redshiftinterp, sfr, label='model')
    plt.xlabel('Redshift', fontsize=15)
    plt.ylabel('SFRD (Msun/yr/Gpc^3)', fontsize=15)
    plt.yscale('log')
    plt.legend(fontsize=12);
    plt.savefig('figures/SFRD_z_comparisonTNG%s-%s.png'%(tng,ver))

    if showplot==True:
        plt.show()

def dPdlogZfrom_TNGdata(path, tng, ver, rbox, showplot=True, shownormalized=True):

    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals, MetalBins = readTNGdata(loc = path, rbox=rbox, metals=True)
    
    #Plot metallicity distribution for each redshift
    #Blue to green = high z to low z
    fig, ax = plt.subplots(figsize = (14,10))
    for i in range(100):
        plt.plot(Sim_center_Zbin/Zsun, Metals[i]*Sim_center_Zbin, label='Snapshot %s' %i, c='%s'%colors[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.title('dP/dZ', fontsize=20)
    plt.xlabel(r'$Z/Z_\odot$', fontsize=20)
    plt.ylabel(r'# SF gas cells with $Z/Z_\odot$', fontsize=20)
    #plt.legend();
    plt.savefig('figures/dPdZ_TNG%s-%s.png'%(tng, ver))

    if showplot==True:
        plt.show()

    if shownormalized == True:
        #Normalized metallicity distribution = probability of getting that Z at that redshift

        fig, ax = plt.subplots(figsize = (14,10))
        totalZ = np.trapz(Metals*Sim_center_Zbin)
        for i in range(100):
            plt.plot(Sim_center_Zbin/Zsun, Metals[i]*Sim_center_Zbin/totalZ[i], label='Snapshot %s' %i, c='%s'%colors[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$Z/Z_\odot$', fontsize=20)
        plt.ylabel(r'PDF', fontsize=20)
        plt.title('Normalized dP/dZ', fontsize=20)
        #plt.legend();
        plt.savefig('figures/dPdZ_normalized_TNG%s-%s.png'%(tng, ver))

        if showplot==True:
            plt.show()

def comparedPdlogZ(path, fit_params, rbox, COMPAS_metallicities, Average_SF_mass_needed, showplot=True, compareformationrates=True):
    
    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals, MetalBins = readTNGdata(loc = path, rbox=rbox, metals=True)
    SFRD_new, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new= interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, ver, saveplot=False)
    mu0, muz, omega0, omegaz, alpha0,sf_a, sf_b, sf_c, sf_d = fit_params

    #sfr using interpolated z range
    sfr2 = find_sfr(redshift_new, a=sf_a, b=sf_b, c=sf_c, d=sf_d)
    minlogZ = np.log(MetalBins[0])
    maxlogZ = np.log(MetalBins[-1])
    steplogZ = (maxlogZ-minlogZ)/len(Sim_center_Zbin)

    minlogZ2 = np.log(MetalBins[0])
    maxlogZ2 = np.log(MetalBins[-1])
    steplogZ2 = (maxlogZ-minlogZ)/len(Sim_center_Zbin)

    #using interpolated Z, z range
    dPdlogZ1, metallicities1, step_logZ1, p_draw_metallicity1 = find_metallicity_distribution(redshift_new, metals = metals_new,
                                    mu0=mu0, muz=muz, sigma_0=omega0, sigma_z=omegaz, alpha =alpha0,
                                    min_logZ=minlogZ, max_logZ=maxlogZ, step_logZ =steplogZ)

    #using Z, z range directly from TNG data
    dPdlogZ2, metallicities2, step_logZ2, p_draw_metallicity2 = find_metallicity_distribution(Redshifts, metals = Sim_center_Zbin,
                                    mu0=mu0, muz=muz, sigma_0=omega0, sigma_z=omegaz, alpha =alpha0,
                                    min_logZ=minlogZ, max_logZ=maxlogZ, step_logZ =steplogZ)

    #calculating Z from min, max, and step logZ
    dPdlogZ3, metallicities3, step_logZ3, p_draw_metallicity3 = find_metallicity_distribution(Redshifts,
                                    mu0=mu0, muz=muz, sigma_0=omega0, sigma_z=omegaz, alpha =alpha0,
                                    min_logZ=minlogZ, max_logZ=maxlogZ, step_logZ =steplogZ)
    
    #dPdlogZ calculared three ways; supposedly should give the same (or approx the same) result
    plt.plot(redshift_new, dPdlogZ1[:, np.digitize(COMPAS_metallicities[0], metals_new)]);
    plt.plot(Redshifts, dPdlogZ2[:, np.digitize(COMPAS_metallicities[0], Sim_center_Zbin)]);
    plt.plot(Redshifts, dPdlogZ3[:, np.digitize(COMPAS_metallicities[0], Sim_center_Zbin)]);

    if showplot==True:
        plt.show()

    if compareformationrates==True:
        # Formation rate using model and using the data
        # should at least be the same order of magnitude
        # looks like there is a normalization factor difference..
        sfr = np.sum(Sim_SFRD*mpc, axis=1)
        n_binaries = len(COMPAS_metallicities)

        fig, ax = plt.subplots(figsize = (10,7))

        model_form_rate = np.zeros(shape=(n_binaries, len(Redshifts)))
        model_form_rate[0, :] = sfr/Average_SF_mass_needed * dPdlogZ2[:, np.digitize(COMPAS_metallicities[0], Sim_center_Zbin)]

        model_form_rate2 = np.zeros(shape=(n_binaries, len(Redshifts)))
        model_form_rate2[0, :] = sfr/Average_SF_mass_needed * dPdlogZ3[:, np.digitize(COMPAS_metallicities[0], Sim_center_Zbin)]

        model_form_rate3 = np.zeros(shape=(n_binaries, len(redshift_new)))
        model_form_rate3[0, :] = sfr2/Average_SF_mass_needed * dPdlogZ1[:, np.digitize(COMPAS_metallicities[0], metals_new)]

        plt.plot(Redshifts, model_form_rate[0], label='Model, with Zrange')
        plt.plot(Redshifts, model_form_rate2[0], label='Model, no Zrange')
        plt.plot(redshift_new, model_form_rate3[0], label='Model, interpolated, with Z range')
        #plt.plot(Redshifts, formation_rate[0], label='TNG%s-%s data'%(tng, ver))
        plt.yscale('log')
        plt.xlabel('Redshift', fontsize=15)
        plt.ylabel('Formation rate (yr^-1 Gpc^-3)', fontsize=15)
        plt.legend(fontsize=12)
        plt.savefig('figures/formrate_modelvsdataTNG%s-%s'%(tng, ver))

        if showplot==True:
            plt.show()


def formationratefrom_TNGdata(path, rbox, COMPAS_metallicities, COMPAS_weights, COMPAS_delay_times, n_binaries, Average_SF_mass_needed, 
                              max_redshift, max_redshift_detection, z_first_SF, p_draw_metallicity, age_first_sfr, redshift_step, 
                              fit_params, showdist, plot_model=False, showplot=True):
    
    #Get TNG data 
    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ, Metals = read.readTNGdata(path)

    print(np.amax(Sim_SFRD))

    # interpolate metallicity data
    SFRDnew, redshift_new, Lookbacktimes_new, metallicities, Metal_distributions, step_fit_logZ_new = read.interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, Metals, redshiftlimandstep=[0, max_redshift+redshift_step, redshift_step])

    #calculate the redshifts array and its equivalents
    redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes = calculate_redshift_related_params(max_redshift, max_redshift_detection, redshift_step, z_first_SF)

    # find the star forming mass per year per Gpc^3 and convert to total number formed per year per Gpc^3
    sfr = np.sum((SFRDnew*u.Msun / u.yr / u.Mpc**3).to(u.Msun / u.yr / u.Gpc**3).value, axis=0) * step_fit_logZ_new

    # Calculate the representative SF mass
    print('Average_SF_mass_needed = ', Average_SF_mass_needed) # print this, because it might come in handy to know when writing up results :)
    n_formed = sfr / Average_SF_mass_needed # Divide the star formation rate density by the representative SF mass

    #fig, ax = plt.subplots(figsize = (12,8))
    plt.plot(redshifts, sfr)
    plt.ylabel("SFRD(z)", fontsize=20)
    plt.xlabel("Redshift", fontsize=20)
    plt.savefig('figures/SFR_data.png')
    plt.show()
    
    print("Calculating rates using data")

    # initalise rates to zero
    n_redshifts = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    formation_rate = np.zeros(shape=(n_binaries, n_redshifts))
    merger_rate = np.zeros(shape=(n_binaries, n_redshifts))

    # interpolate times and redshifts for conversion
    times_to_redshifts = interp1d(times, redshifts)

    #from sklearn import preprocessing

    normalized_metal_dists = []
    for z_ind, z in enumerate(redshifts):
        metal_dist_form = Metal_distributions[:, z_ind] * metallicities
        norm_metal_dist_form = metal_dist_form/np.linalg.norm(metal_dist_form)
        normalized_metal_dists.append(norm_metal_dist_form)

        """
        norm_metal_dist_form2 = metal_dist_form/np.trapz(metal_dist_form)
        norm_metal_dist_form3 = preprocessing.normalize([metal_dist_form])

        if z_ind==0:
            plt.plot(metallicities, norm_metal_dist_form, label='linalg norm')
            plt.plot(metallicities, norm_metal_dist_form2, label='trapz')
            plt.plot(metallicities, norm_metal_dist_form3[0, :], label='sklearn')
            plt.legend()
            plt.show()
        """
    
    normalized_metal_dists = np.array(normalized_metal_dists)

    for i in range(n_binaries):

        formation_rate[i, :] = n_formed * calc_dPdlogZ(COMPAS_metallicities[i], metallicities, 
                                normalized_metal_dists) / p_draw_metallicity * COMPAS_weights[i]

        if i==0:
            print("The binary metallicity is", COMPAS_metallicities[i])

            #fig, ax = plt.subplots(figsize = (12,8))
            plt.plot(redshifts, calc_dPdlogZ(COMPAS_metallicities[i], metallicities, normalized_metal_dists))
            plt.xlim(0, 10)
            plt.ylabel("dP/dlogZ", fontsize=20)
            plt.xlabel("Redshift", fontsize=20)
            plt.title('Z = %s'%COMPAS_metallicities[i])
            plt.savefig('figures/dPdlogZ_data.png')
            plt.show()

            #print(metallicities[211], len(metallicities))
            #print(np.digitize(COMPAS_metallicities[i], metallicities))
            #print(len(calc_dPdlogZ(COMPAS_metallicities[i], metallicities, normalized_metal_dists)))
            #print(calc_dPdlogZ(COMPAS_metallicities[i], metallicities, Metal_distributions))
        
        if plot_model==False:
            plt.plot(redshifts, formation_rate[i], label="Data")
            plt.xlim(0, 10)
            plt.ylabel("Formation rate")
            plt.xlabel("Redshift")
            #plt.show()
        
        if plot_model==True:
            # calculate formation rate (see Neijssel+19 Section 4) - note this uses dPdlogZ for *closest* metallicity
            form_rate_model = np.zeros(shape=(n_binaries, len(redshifts)))
            mu_0, mu_z, omega0, omegaz, alpha0,sf_a, sf_b, sf_c, sf_d = fit_params
            sfr = find_sfr(redshifts, sf_a, sf_b, sf_c, sf_d)
            plt.plot(redshifts, sfr)
            plt.ylabel("SFRD(z)", fontsize=20)
            plt.xlabel("Redshift", fontsize=20)
            plt.show()
            n_formed = sfr/Average_SF_mass_needed
            dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                find_metallicity_distribution(redshifts,  mu0 = mu_0 , muz = mu_z,
                                                sigma_0= omega0 , sigma_z=omegaz , alpha = alpha0)
                                                #min_logZ=np.log(metallicities[0]), max_logZ= np.log(metallicities[-1]), step_logZ=step_fit_logZ_new)
            plt.plot(redshifts, dPdlogZ[:, np.digitize(COMPAS_metallicities[i], metallicities)])
            #print('Model')
            #print(COMPAS_metallicities[i], len(metallicities))
            #print(np.digitize(COMPAS_metallicities[i], metallicities))
            #print(dPdlogZ[:, np.digitize(COMPAS_metallicities[i], metallicities)].shape)
            plt.xlim(0, 10)
            plt.ylabel("dP/dlogZ", fontsize=20)
            plt.xlabel("Redshift", fontsize=20)
            plt.title('Z = %s'%COMPAS_metallicities[i])
            plt.show()
            
            form_rate_model[i, :] = n_formed * dPdlogZ[:, np.digitize(COMPAS_metallicities[i], metallicities)]  / p_draw_metallicity * COMPAS_weights[i]
            plt.plot(redshifts, formation_rate[i], label="Data")
            plt.plot(redshifts, form_rate_model[i], label='Model')
            plt.xlim(0, 10)
            plt.ylabel("Formation rate")
            plt.xlabel("Redshift")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    tng=50
    ver = 1
    Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5"%(tng,ver))
    fit_filename = ['test_best_fit_parameters_TNG%s-%s.txt'%(tng,ver)]
    if tng==50:
        rbox=35
    elif tng==100:
        rbox=75
    elif tng==300:
        rbox=205
    SFRD_Z_z_fname = 'SFRD_TNG%s-%s.txt'%(tng,ver)
    Zsun = 0.014 # Solar metallicity
    cmap = sns.color_palette('rocket', as_cmap=True)

    fit_params = read_best_fits(fit_filename)[0]

    #Conversion to per Gpc^3
    gpc = 1*u.Gpc**3
    mpc = gpc.to(u.Mpc**3).value

    blue = Color("blue")
    colors = list(blue.range_to(Color("green"),100))
    for i, val in enumerate(colors):
        colors[i] = str(val)

    #SFRfrom_TNGdata(Cosmol_sim_location, tng, ver, rbox, fit_filename)
    
    #dPdlogZfrom_TNGdata(Cosmol_sim_location, tng, ver, rbox)

    #comparedPdlogZ(Cosmol_sim_location, fit_params, rbox, COMPAS_metallicities=[0.0003526597629725001], Average_SF_mass_needed=1010913214.2182984, 
    #               showplot=True, compareformationrates=True)

    formationratefrom_TNGdata(Cosmol_sim_location, rbox, COMPAS_metallicities=[0.0003526597629725001], COMPAS_weights=[0.0034379198705589455], 
                              COMPAS_delay_times=[16.263754997115225], n_binaries=1, Average_SF_mass_needed=1010913214.2182984, 
                              max_redshift =14, max_redshift_detection=14, z_first_SF=14,
                              p_draw_metallicity=0.1753222685894653, age_first_sfr=472.4939676954561, redshift_step=0.05, fit_params=fit_params, 
                              plot_model=True, showdist=False, showplot=True)

    """
    From one random COMPAS binary running the full cosmic integration
    times [13797.6158969, 13654.32095468, 13513.10090191]
    p draw metallicity 0.1753222685894653
    compas weights 0.0034379198705589455
    compas delay times 16.263754997115225
    age first sfr 472.4939676954561
    compas metallicities 0.0003526597629725001
    redshift step 0.01
    Average_SF_mass_needed =  1010913214.2182984 solMass
    """