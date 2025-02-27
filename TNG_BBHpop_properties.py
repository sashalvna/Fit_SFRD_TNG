import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib
from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses

############################
# Custom scripts
import paths
import init_values as In
import MassDistHelperFunctions as mfunc

## PLOT setttings
plt.rc('font', family='serif')
from matplotlib import rc
import matplotlib
plt.rc('font', family='serif', weight='bold')
plt.rc('text', usetex=True)
matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams.update({'font.weight': 'bold'})

def read_best_fits(fit_param_files):
    #Read in best fit parameters for each TNG into one array of arrays
    fit_param_vals = []
    for file in fit_param_files:
        mu0_best, muz_best, omega0_best, omegaz_best, alpha0_best, sf_a_best, sf_b_best, sf_c_best, sf_d_best = np.loadtxt(str(paths.data)+'/'+file,unpack=True, delimiter=',')
        fit_param_vals.append([mu0_best, muz_best, omega0_best, omegaz_best, alpha0_best, sf_a_best, sf_b_best, sf_c_best, sf_d_best])
    return np.array(fit_param_vals)


def plot_BBH_merger_rate(data_dir, rates, fit_param_vals, plot_zoomed=False, plot_logscale=False, showplot=True):

    fig, ax = plt.subplots(figsize = (12, 8))
    redshifts = []
    merger_rates = []
    for i, rfile in enumerate(rates):
        rate_key = 'Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(fit_param_vals[i][0], 3), 
                                                                                      np.round(fit_param_vals[i][1], 3), 
                                                                                      np.round(fit_param_vals[i][4], 3), 
                                                                                      np.round(fit_param_vals[i][2], 3), 
                                                                                      np.round(fit_param_vals[i][3], 3), 
                                                                                      np.round(fit_param_vals[i][5], 3), 
                                                                                      np.round(fit_param_vals[i][6], 3), 
                                                                                      np.round(fit_param_vals[i][7], 3), 
                                                                                      np.round(fit_param_vals[i][8], 3))
        
        with h5.File(data_dir + rfile ,'r') as File:
            redshift      = File[rate_key]['redshifts'][()]
            merger_rate    = File[rate_key]['merger_rate'][()]
    
        total_merger_rate = np.sum(merger_rate, axis=0)
        plt.plot(redshift, total_merger_rate, label='TNG%s'%labels[i], ls=linestyles[i], lw=4, color=line_colors[i])
        redshifts.append(redshift)
        merger_rates.append(total_merger_rate)
        print("The TNG%s merger rate at z=%s is: "%(labels[i], redshift[0]), total_merger_rate[0])
    
    ax.set_xlabel('Redshift', fontsize=25)
    ax.set_ylabel(r'Merger rate $(\rm Gpc^{-3} yr^{-1})$', fontsize=25)
    ax.set_xlim(0, 14)
    ax.set_ylim(10**-3, 10**3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    if plot_logscale == True:
        ax.set_yscale('log')
    fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=18, loc="upper right", frameon=False)

    ax2 = ax.twiny()
    redshift_tick_list = [0,0.1, 0.25, 0.5, 1.0, 10]
    lookbackt_tick_list = [cosmo.lookback_time(z).value for z in redshift_tick_list]
    ax2.set_xlabel('Lookback time (Gyr)', fontsize = 20)
    ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
    ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(length=5, width=1.5, which='major')

    plt.savefig('figures/merger_rates_TNG.png', bbox_inches='tight', dpi=300)

    if plot_zoomed == True:
        fig, axes = plt.subplots(1, 1, figsize=(10, 7))

        for i in range(len(merger_rates)):
            plt.plot(redshifts[i], merger_rates[i], label='TNG%s'%labels[i], ls=linestyles[i], lw=4, color=line_colors[i])

        plt.xlabel('Redshift', fontsize=20)
        plt.ylabel(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=20)
        fig.legend(bbox_to_anchor=(0.9, 0.3), fontsize=18, frameon=False)
        plt.xlim(9.5, 10)
        plt.yscale('log')
        plt.savefig('figures/merger_rates_TNG_zoomed.png', bbox_inches='tight')

    if showplot==True:
        plt.show()


def compare_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, plot_merger_rates=True, plot_logscale=False, showplot=True):

    fig, ax = plt.subplots(figsize = (12, 8))

    for i, rfile in enumerate(model_rates):
        rate_key = 'Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(fit_param_vals[i][0], 3), 
                                                                                      np.round(fit_param_vals[i][1], 3), 
                                                                                      np.round(fit_param_vals[i][4], 3), 
                                                                                      np.round(fit_param_vals[i][2], 3), 
                                                                                      np.round(fit_param_vals[i][3], 3), 
                                                                                      np.round(fit_param_vals[i][5], 3), 
                                                                                      np.round(fit_param_vals[i][6], 3), 
                                                                                      np.round(fit_param_vals[i][7], 3), 
                                                                                      np.round(fit_param_vals[i][8], 3))
        
        with h5.File(data_dir + rfile ,'r') as File:
            redshift      = File[rate_key]['redshifts'][()]
            formation_rate = File[rate_key]['formation_rate'][()]
            merger_rate    = File[rate_key]['merger_rate'][()]

        with h5.File(data_dir + data_rates[i] ,'r') as File:
            data_redshift      = File[rate_key]['redshifts'][()]
            data_formation_rate = File[rate_key]['formation_rate'][()]
            data_merger_rate    = File[rate_key]['merger_rate'][()]

        if plot_merger_rates == True:
            #Plot merger rates
            total_merger_rate = np.sum(merger_rate, axis=0)
            total_data_merger_rate = np.sum(data_merger_rate, axis=0)
            plt.plot(data_redshift, total_data_merger_rate, label='TNG%s'%labels[i], lw=8, c=data_colors[i])
            plt.plot(redshift, total_merger_rate, lw=2, c=model_colors[i], ls='--')
            print("The TNG%s model merger rate at z=%s is: "%(labels[i], redshift[0]), total_merger_rate[0])
            print("The TNG%s data merger rate at z=%s is: "%(labels[i], data_redshift[0]), total_data_merger_rate[0])

        else:
            #Plot formation rates
            total_formation_rate = np.sum(formation_rate, axis=0)
            total_data_formation_rate = np.sum(data_formation_rate, axis=0)

            plt.plot(data_redshift, total_data_formation_rate, label='TNG%s'%labels[i], lw=8, c=data_colors[i])
            plt.plot(redshift, total_formation_rate, lw=2, c=model_colors[i], ls='--')
            print("The TNG%s model formation rate at z=%s is: "%(labels[i], redshift[0]), total_formation_rate[0])
            print("The TNG%s data formation rate at z=%s is: "%(labels[i], data_redshift[0]), total_data_formation_rate[0])

    x = [-0.0001]
    y1 = [1]
    y2 = [1]
    plt.plot(x, y1, c='black', ls = '-', lw=5, label='Simulation')
    plt.plot(x, y2, c='black', ls = '--', lw=2, label='Analytical fit')
    ax.set_xlim(0, 14)
    ax.set_ylim(10**-3, 10**3)
    ax.set_xlabel('Redshift', fontsize=25)
    if plot_merger_rates == True:
        ax.set_ylabel(r'Merger rate $(\rm Gpc^{-3} yr^{-1})$', fontsize=25)
        fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=18, loc="upper right", frameon=False)
    else:
        ax.set_ylabel(r'Formation rate $(\rm Gpc^{-3} yr^{-1})$', fontsize=25)
        fig.legend(bbox_to_anchor=(0.15, 0.1), fontsize=18, loc="lower left", frameon=False)
    ax.set_xlim(0, 14)
    ax.set_ylim(10**-3, 10**3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    if plot_logscale == True:
        ax.set_yscale('log')

    ax2 = ax.twiny()
    redshift_tick_list = [0,0.1, 0.25, 0.5, 1.0, 10]
    lookbackt_tick_list = [cosmo.lookback_time(z).value for z in redshift_tick_list]
    ax2.set_xlabel('Lookback time (Gyr)', fontsize = 20)
    ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
    ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(length=5, width=1.5, which='major')

    if plot_merger_rates == True:
        plt.ylabel(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=25)
        plt.savefig('figures/merger_rates_datavsmodel_TNG.png', bbox_inches='tight', dpi=300)

    else:
        plt.ylabel(r'Formation rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=25)
        plt.savefig('figures/formation_rates_datavsmodel_TNG.png', bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def plot_BBH_mass_dist(rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = 0.2, showplot=True, show_reference_masses=True):
    #get rate file names and rate keys
    TNGpaths = []
    rate_keys = []

    for i in rates:
        TNGpaths.append('/'+i)
    for i in fit_param_vals:
        rate_keys.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    fig, ax = plt.subplots(figsize = (12, 8))
    bins = np.arange(0.,55,2.5)
    plot_lines = []
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm (M_{\odot})$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \left (\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr \mathrm{d}M_{\odot}} \right )$'
    #ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    # GWTC-3 Powerlaw + Peak Mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'o3only_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5'
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    with h5.File(input_fname, "r") as f:
        mass_ppd = f["ppd"]
        mass_lines = f["lines"]
        mass_1_ppd = np.trapz(mass_ppd, mass_ratio, axis=0)
        mass_1_lower = np.percentile(mass_lines["mass_1"], 5, axis=0)
        mass_1_upper = np.percentile(mass_lines["mass_1"], 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(mass_1, mass_1_ppd, lw=1.8, color=color_plpeak, zorder=1, label="GWTC-3")
    ax.fill_between(mass_1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)
    nplot=0

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath, i)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[rate_keys[i]]['redshifts'][()]
            DCO_mask = File[rate_keys[i]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[rate_keys[i]]['merger_rate'][()]

        CEcount = 'CE_Event_Counter'
        #first bring it to the same shape as the rate table
        merging_BBH    = DCO[DCO_mask]
        #apply the additional mask based on your prefs
        if np.logical_and(only_stable, only_CE):
            print("Both only_stable and only_CE, I assume you just want both")
            channel_bool = np.full(len(merging_BBH), True)
        elif only_stable:
            channel_bool = merging_BBH[CEcount] == 0
        elif only_CE:
            channel_bool = merging_BBH[CEcount] > 0
        else:
            raise ValueError("Both only_stable =%s and only_CE=%s, set at least one to true"%(only_stable,only_CE))
        # we exclude CHE systems
        not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
        BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)

        merging_BBH         = merging_BBH[BBH_bool * not_CHE  * channel_bool]
        Red_intr_rate_dens  = intrinsic_rate_density[BBH_bool* not_CHE * channel_bool, :]
    
        #Calculate average rate density per z-bin
        x_vals              = merging_BBH['M_moreMassive']
        i_redshift = np.where(redshifts == z)[0][0] # Rate at redshift 0.2
        Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
    
        # Get the Hist    
        hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

        # And the KDE
        kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

        m10.append(kernel(10)*sum(hist)) # append value at reference mass 
        m25.append(kernel(25)*sum(hist)) # append value at reference mass 
        m40.append(kernel(40)*sum(hist)) # append value at reference mass 
    
        x_KDE = np.arange(0.1,50.,0.1)
        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
        plot_lines.append(ax.plot(x_KDE, KDEy_vals, label = 'TNG'+labels[nplot], color=line_colors[nplot], lw= 4,  zorder =i+1,ls = '-'))
    
        nplot += 1
        
    #########################################
    # plot values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    
    # Channel
    plt.text(0.75, 0.60, '$\mathrm{%s \ channel}$\nz=%s'%(channel_string, z), ha = 'center', transform=ax.transAxes, size = 20)

    ax.set_xlabel(xlabel, fontsize = 25)
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.set_yscale('log')
    fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=15)
    fig.savefig('figures/massdist_%s_z%s.png'%(channel_string, z), bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def plot_BBH_mass_dist_over_z(rates, fit_param_vals, tng, model=True, only_stable = True, only_CE = True, channel_string='all', z = [0.2, 1, 2, 4, 6, 8], showplot=True):
    #get rate file names and rate keys
    TNGpaths = []
    rate_keys = []

    if tng==50:
        nplot=0
        colors = plt.cm.GnBu(np.linspace(0.3,1,len(z)))
    elif tng==100:
        nplot=1
        colors = plt.cm.PuRd(np.linspace(0.3,1,len(z)))
    else:
        nplot=2
        colors = plt.cm.YlGn(np.linspace(0.3,1,len(z)))

    for i in rates:
        TNGpaths.append('/'+i)
    for i in fit_param_vals:
        rate_keys.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    fig, ax = plt.subplots(figsize = (12, 8))
    bins = np.arange(0.,55,2.5)
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    # GWTC-3 Powerlaw + Peak Mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'o3only_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5'
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    with h5.File(input_fname, "r") as f:
        mass_ppd = f["ppd"]
        mass_lines = f["lines"]
        mass_1_ppd = np.trapz(mass_ppd, mass_ratio, axis=0)
        mass_1_lower = np.percentile(mass_lines["mass_1"], 5, axis=0)
        mass_1_upper = np.percentile(mass_lines["mass_1"], 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(mass_1, mass_1_ppd, lw=1.8, color=color_plpeak, zorder=1, label="GWTC-3")
    ax.fill_between(mass_1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[rate_keys[i]]['redshifts'][()]
            DCO_mask = File[rate_keys[i]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[rate_keys[i]]['merger_rate'][()]

        CEcount = 'CE_Event_Counter'
        #first bring it to the same shape as the rate table
        merging_BBH    = DCO[DCO_mask]
        #apply the additional mask based on your prefs
        if np.logical_and(only_stable, only_CE):
            print("Both only_stable and only_CE, I assume you just want both")
            channel_bool = np.full(len(merging_BBH), True)
        elif only_stable:
            channel_bool = merging_BBH[CEcount] == 0
        elif only_CE:
            channel_bool = merging_BBH[CEcount] > 0
        else:
            raise ValueError("Both only_stable =%s and only_CE=%s, set at least one to true"%(only_stable,only_CE))
        # we exclude CHE systems
        not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
        BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)

        merging_BBH         = merging_BBH[BBH_bool * not_CHE  * channel_bool]
        Red_intr_rate_dens  = intrinsic_rate_density[BBH_bool* not_CHE * channel_bool, :]

        for j, redshift in enumerate(z):

            print("Plotting redshift z = ", redshift)
    
            #Calculate average rate density per z-bin
            x_vals              = merging_BBH['M_moreMassive']
            i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
            Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
            Weights[Weights < 0] = 0
    
            # Get the Hist    
            hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

            # And the KDE
            kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)
    
            x_KDE = np.arange(0.1,50.,0.1)
            KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
            ax.plot(x_KDE, KDEy_vals, label = 'z = %s'%redshift, color=colors[j], lw= 4, ls = '-')
        
    #########################################
    # plot values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')
    
    # Channel
    plt.text(0.12, 0.92, 'TNG%s'%labels[nplot], ha = 'center', transform=ax.transAxes, size = 25)
    
    ax.set_xlabel(xlabel, fontsize = 25)
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.set_yscale('log')
    if channel_string=='all':
        ax.set_title('all channels', fontsize = 25)
    else:
        ax.set_title('%s channel'%channel_string, fontsize = 25)
    fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=18, frameon=False)
    fig.savefig('figures/massdist_TNG%s_%s_redshift_evol.png'%(tng, channel_string), bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()

def plot_BBH_mass_dist_over_z_allTNGs(rates, fit_param_vals, tngs, z = [0.2, 1, 2, 4, 6, 8], showplot=True):
    #get rate file names and rate keys
    TNGpaths = []
    rate_keys = []
    colors  = []
    nplot = []

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize = (14, 16))
    fig.subplots_adjust(wspace=0, hspace=0)

    for i in tngs:
        if i==50:
            nplot.append(0)
            colors.append(plt.cm.GnBu(np.linspace(0.3,1,len(z))))
        elif i==100:
            nplot.append(1)
            colors.append(plt.cm.PuRd(np.linspace(0.3,1,len(z))))
        else:
            nplot.append(2)
            colors.append(plt.cm.YlGn(np.linspace(0.3,1,len(z))))

    for i in rates:
        TNGpaths.append('/'+i)
    for i in fit_param_vals:
        rate_keys.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    bins = np.arange(0.,55,2.5)
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    # GWTC-3 Powerlaw + Peak Mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'o3only_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5'
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    with h5.File(input_fname, "r") as f:
        mass_ppd = f["ppd"]
        mass_lines = f["lines"]
        mass_1_ppd = np.trapz(mass_ppd, mass_ratio, axis=0)
        mass_1_lower = np.percentile(mass_lines["mass_1"], 5, axis=0)
        mass_1_upper = np.percentile(mass_lines["mass_1"], 95, axis=0)
    # plot the max posterior and the 95th percentile
    for i in range(3):
        for j in range(3):
            if i==j==0:
                ax[i, j].plot(mass_1, mass_1_ppd, lw=1.8, color=color_plpeak, zorder=1, label="GWTC-3")
            else:
                ax[i, j].plot(mass_1, mass_1_ppd, lw=1.8, color=color_plpeak, zorder=1)
            ax[i, j].fill_between(mass_1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))
    channels = ['all', 'stable', 'CE']

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[rate_keys[i]]['redshifts'][()]
            DCO_mask = File[rate_keys[i]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[rate_keys[i]]['merger_rate'][()]

        for j, channel in enumerate(channels):
            print('Plotting %s channel'%channel)

            if channel == 'all':
                only_stable=True
                only_CE=True
            elif channel == 'stable':
                only_stable=True
                only_CE=False
            else:
                only_stable=False
                only_CE=True

            CEcount = 'CE_Event_Counter'
            #first bring it to the same shape as the rate table
            merging_BBH    = DCO[DCO_mask]
            #apply the additional mask based on your prefs
            if np.logical_and(only_stable, only_CE):
                print("Both only_stable and only_CE, I assume you just want both")
                channel_bool = np.full(len(merging_BBH), True)
            elif only_stable:
                channel_bool = merging_BBH[CEcount] == 0
            elif only_CE:
                channel_bool = merging_BBH[CEcount] > 0
            else:
                raise ValueError("Both only_stable =%s and only_CE=%s, set at least one to true"%(only_stable,only_CE))
            # we exclude CHE systems
            not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
            BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)

            merging_BBH         = merging_BBH[BBH_bool * not_CHE  * channel_bool]
            Red_intr_rate_dens  = intrinsic_rate_density[BBH_bool* not_CHE * channel_bool, :]

            for m, redshift in enumerate(z):

                print("Plotting redshift z = ", redshift)
    
                #Calculate average rate density per z-bin
                x_vals              = merging_BBH['M_moreMassive']
                i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
                Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
                Weights[Weights < 0] = 0
    
                # Get the Hist    
                hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

                # And the KDE
                kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)
    
                x_KDE = np.arange(0.1,50.,0.1)
                KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
                ax[i, j].plot(x_KDE, KDEy_vals, color=colors[i][m], lw= 4, ls = '-')
    
                # plot values
                ax[i, j].set_xlim(x_lim)
                ax[i, j].set_ylim(y_lim)
                ax[i, j].tick_params(axis='both', which='major', labelsize=15)
                ax[i, j].xaxis.set_minor_locator(ticker.MultipleLocator(1))
                ax[i, j].tick_params(length=10, width=2, which='major')
                ax[i, j].tick_params(length=5, width=1, which='minor')
                ax[i, j].set_yscale('log')

                if j==3 and i!=3:
                    labels = ax[i,j].get_xticklabels()
                    labels[-1] = ''
                    ax[i,j].set_xticklabels(labels)
    
                # Channel
                #ax[i, j].text(0.12, 0.92, 'TNG%s'%labels[nplot[i]], ha = 'center', transform=ax.transAxes, size = 25)
    
    fig.supxlabel(xlabel, y=0.04, fontsize = 30)
    fig.supylabel(ylabel, x=0.03, fontsize = 30)

    #cbar = fig.colorbar(z, orientation='horizontal')
    #cbar.ax.tick_params(labelsize=20)

    cmap = plt.get_cmap('Greys',len(z))
    norm = matplotlib.colors.Normalize(vmin=0,vmax=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0], orientation='horizontal')
                
    fig.legend(bbox_to_anchor=(0.8, 1), fontsize=18, ncol = 4)
    fig.savefig('figures/massdist_allTNGs_redshift_evol.png', bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = 0.2, showplot=True, show_reference_masses=False):
    #get rate file names and rate keys
    TNGpaths = []
    rate_keys = []
    TNGpaths_data = []
    rate_keys_data = []

    for i in model_rates:
        TNGpaths.append('/'+i)
    for i in fit_param_vals:
        rate_keys.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    for i in data_rates:
        TNGpaths_data.append('/'+i)
    for i in fit_param_vals:
        rate_keys_data.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    fig, ax = plt.subplots(figsize = (12, 8))
    bins = np.arange(0.,55,2.5)
    plot_lines = []
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    # GWTC-3 Powerlaw + Peak Mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'o3only_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5'
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)
    with h5.File(input_fname, "r") as f:
        mass_ppd = f["ppd"]
        mass_lines = f["lines"]
        mass_1_ppd = np.trapz(mass_ppd, mass_ratio, axis=0)
        mass_1_lower = np.percentile(mass_lines["mass_1"], 5, axis=0)
        mass_1_upper = np.percentile(mass_lines["mass_1"], 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(mass_1, mass_1_ppd, lw=1.8, color=color_plpeak, zorder=1, label="GWTC-3")
    ax.fill_between(mass_1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)
    nplot=0

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    #Loop over TNGs
    nplot=0
    for i, tngpath in enumerate(TNGpaths_data):
        print('Path to TNG', tngpath, i)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[rate_keys_data[i]]['redshifts'][()]
            DCO_mask = File[rate_keys_data[i]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[rate_keys_data[i]]['merger_rate'][()]

        CEcount = 'CE_Event_Counter'
        #first bring it to the same shape as the rate table
        merging_BBH    = DCO[DCO_mask]
        #apply the additional mask based on your prefs
        if np.logical_and(only_stable, only_CE):
            print("Both only_stable and only_CE, I assume you just want both")
            channel_bool = np.full(len(merging_BBH), True)
        elif only_stable:
            channel_bool = merging_BBH[CEcount] == 0
        elif only_CE:
            channel_bool = merging_BBH[CEcount] > 0
        else:
            raise ValueError("Both only_stable =%s and only_CE=%s, set at least one to true"%(only_stable,only_CE))
        # we exclude CHE systems
        not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
        BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)

        merging_BBH         = merging_BBH[BBH_bool * not_CHE  * channel_bool]
        Red_intr_rate_dens  = intrinsic_rate_density[BBH_bool* not_CHE * channel_bool, :]
    
        #Calculate average rate density per z-bin
        x_vals              = merging_BBH['M_moreMassive']
        i_redshift = np.where(redshifts == z)[0][0] # Rate at redshift 0.2
        Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
        Weights[Weights < 0] = 0
    
        # Get the Hist    
        hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

        # And the KDE
        kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

        x_KDE = np.arange(0.1,50.,0.1)
        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
        plot_lines.append(ax.plot(x_KDE, KDEy_vals, color=data_colors[nplot], label = 'TNG '+labels[nplot], lw=8, ls = '-'))
    
        nplot += 1

    ####################################################
    #Loop over TNGs
    nplot = 0
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath, i)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[rate_keys[i]]['redshifts'][()]
            DCO_mask = File[rate_keys[i]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[rate_keys[i]]['merger_rate'][()]

        CEcount = 'CE_Event_Counter'
        #first bring it to the same shape as the rate table
        merging_BBH    = DCO[DCO_mask]
        #apply the additional mask based on your prefs
        if np.logical_and(only_stable, only_CE):
            print("Both only_stable and only_CE, I assume you just want both")
            channel_bool = np.full(len(merging_BBH), True)
        elif only_stable:
            channel_bool = merging_BBH[CEcount] == 0
        elif only_CE:
            channel_bool = merging_BBH[CEcount] > 0
        else:
            raise ValueError("Both only_stable =%s and only_CE=%s, set at least one to true"%(only_stable,only_CE))
        # we exclude CHE systems
        not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
        BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)

        merging_BBH         = merging_BBH[BBH_bool * not_CHE  * channel_bool]
        Red_intr_rate_dens  = intrinsic_rate_density[BBH_bool* not_CHE * channel_bool, :]
    
        #Calculate average rate density per z-bin
        x_vals              = merging_BBH['M_moreMassive']
        i_redshift = np.where(redshifts == z)[0][0] # Rate at redshift 0.2
        Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
        Weights[Weights < 0] = 0
    
        # Get the Hist    
        hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

        # And the KDE
        kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

        x_KDE = np.arange(0.1,50.,0.1)
        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
        ax.plot(x_KDE, KDEy_vals, color=model_colors[nplot], lw= 2, ls='--')
    
        nplot += 1
        
    #########################################
    # plot values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')

    #legend
    x = [-0.0001]
    y1 = [1]
    y2 = [1]
    plt.plot(x, y1, c='black', ls = '-', lw=6, label='Simulation')
    plt.plot(x, y2, c='black', ls = '--', lw=2, label='Analytical fit')

    ax.set_xlabel(xlabel, fontsize = 25)
    ax.set_ylabel(ylabel, fontsize = 25)
    ax.set_yscale('log')
    if channel_string=='all':
        plt.text(0.03, 0.88, '$\mathrm{%s \ channels}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)
    else:
        plt.text(0.03, 0.88, '$\mathrm{%s \ channel}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)

    fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=18, frameon=False)
    fig.savefig('figures/massdist_modelvsdata_%s_z%s.png'%(channel_string, z), bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()

if __name__ == "__main__":

    #TNG data setup (will eventually turn into arguments but currently just uncomment the ones that you want)
    In.init()
    data_dir    =  str(paths.data) +'/'
    save_loc = str(paths.figures) + '/'
    COMPASfilename = 'COMPAS_Output_wWeights.h5'

    filenames = ['SFRMetallicityFromGasWithMetalsTNG50-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG100-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG300-1.hdf5',
                 'SFRMetallicityFromGasTNG100-2.hdf5', 'SFRMetallicityFromGasTNG50-2.hdf5', 'SFRMetallicityFromGasTNG50-3.hdf5'] 
    fit_param_files = ['test_best_fit_parameters_TNG50-1.txt', 'test_best_fit_parameters_TNG100-1.txt', 'test_best_fit_parameters_TNG300-1.txt']
    rates = ['Rate_info_TNG50-1.h5', 'Rate_info_TNG100-1.h5', 'Rate_info_TNG300-1.h5']
    model_rates = ['Rate_info_TNG50-1.h5', 'Rate_info_TNG100-1.h5', 'Rate_info_TNG300-1.h5']
    data_rates = ['data_Rate_info_TNG50-1.h5', 'data_Rate_info_TNG100-1.h5', 'data_Rate_info_TNG300-1.h5']

    #Plot setup
    labels = ['50-1', '100-1', '300-1']
    linestyles = ['-', '-', '-', '-', '-', '-']
    lineweights = [4, 4, 4, 4, 4, 4]
    data_colors = [plt.cm.GnBu(0.8), plt.cm.PuRd(0.6), plt.cm.YlGn(0.4)]
    model_colors = ['midnightblue', plt.cm.PuRd(0.9), plt.cm.YlGn(0.8)]
    line_colors = [plt.cm.GnBu(0.99), plt.cm.PuRd(0.7), plt.cm.YlGn(0.5)]
    line_styles = ['solid', 'dashed', 'dotted']
    tmin = 0.0
    tmax = 13.7
    z_of_massdist = [1, 2, 4, 6, 8]

    #Read in SFRD model parameters for each TNG
    fit_param_vals = read_best_fits(fit_param_files)

    #Plot merger rates for all TNGs in one plot
    #plot_BBH_merger_rate(data_dir, rates, fit_param_vals, plot_zoomed=False,  plot_logscale=True, showplot=True)

    #Compare model and data merger and formation rates
    #compare_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, plot_merger_rates=False, plot_logscale=True, showplot=True)
    #compare_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, plot_merger_rates=True, plot_logscale=True, showplot=True)

    #Plot primary mass distribution for all TNGs in one plot
    #plot_BBH_mass_dist(rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = z_of_massdist, showplot=True, show_reference_masses=False)
    #plot_BBH_mass_dist(rates, fit_param_vals, only_stable = True, only_CE = False, channel_string='stable', z = z_of_massdist, showplot=True, show_reference_masses=False)
    #plot_BBH_mass_dist(rates, fit_param_vals, only_stable = False, only_CE = True, channel_string='CE', z = z_of_massdist, showplot=True, show_reference_masses=False)

    plot_BBH_mass_dist_over_z_allTNGs(data_rates, fit_param_vals, tngs=[50, 100, 300], z = [0.2], showplot=True)

    #plot_BBH_mass_dist_over_z([data_rates[0]], [fit_param_vals[0]], tng=50, only_stable = True, only_CE = True, channel_string='all', z = [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2], showplot=False)
    #plot_BBH_mass_dist_over_z([data_rates[0]], [fit_param_vals[0]], tng=50, only_stable = True, only_CE = False, channel_string='stable', z = [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2], showplot=False)
    #plot_BBH_mass_dist_over_z([data_rates[0]], [fit_param_vals[0]], tng=50, only_stable = False, only_CE = True, channel_string='CE', z = [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2], showplot=False)

    #Compare model and data mass distributions

    #for z in z_of_mass_dist:
    #    compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = z, showplot=False, show_reference_masses=False)
    #    compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = True, only_CE = False, channel_string='stable', z = z, showplot=False, show_reference_masses=False)
    #    compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = False, only_CE = True, channel_string='CE', z = z, showplot=False, show_reference_masses=False)
    
