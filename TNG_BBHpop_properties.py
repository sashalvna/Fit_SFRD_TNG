import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib
from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses
from scipy.interpolate import interp1d
import astropy.units as u
import seaborn as sns
import popsummary

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
        mu0_best, muz_best, omega0_best, omegaz_best, alpha0_best, sf_a_best, sf_b_best, sf_c_best, sf_d_best = np.loadtxt(str(paths.data)+'/' + file,unpack=True, delimiter=',')
        fit_param_vals.append([mu0_best, muz_best, omega0_best, omegaz_best, alpha0_best, sf_a_best, sf_b_best, sf_c_best, sf_d_best])
    return np.array(fit_param_vals)


def plot_BBH_merger_rate(data_dir, rates, fit_param_vals, plot_zoomed=False, plot_logscale=False, showplot=True):

    fig, ax = plt.subplots(figsize = (12, 8))
    redshifts = []
    merger_rates = []
    for i, rfile in enumerate(rates):
        with h5.File(data_dir + rfile ,'r') as File:
            redshift      = File[list(File.keys())[0]]['redshifts'][()]
            merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]
    
        total_merger_rate = np.sum(merger_rate, axis=0)
        plt.plot(redshift, total_merger_rate, label='TNG%s'%labels[i], ls=linestyles[i], lw=4, color=data_colors[i])
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
    redshift_tick_list = [0, 1, 2, 6, 10, 14]#[0,0.1, 0.25, 0.5, 1.0, 10]
    ax2.set_xticks([z for z in redshift_tick_list])
    ax2.set_xticklabels(['${:.1f}$'.format(cosmo.lookback_time(z).value) for z in redshift_tick_list], fontsize = 24)
    ax2.set_xlabel('Lookback time [Gyr]', fontsize = 20)
    #ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
    #ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(length=5, width=1.5, which='major')

    plt.savefig('figures/merger_rates_TNG.pdf', format="pdf", bbox_inches='tight', dpi=300)

    if plot_zoomed == True:
        fig, axes = plt.subplots(1, 1, figsize=(10, 7))

        for i in range(len(merger_rates)):
            plt.plot(redshifts[i], merger_rates[i], label='TNG%s'%labels[i], ls=linestyles[i], lw=4, color=data_colors[i])

        plt.xlabel('Redshift', fontsize=20)
        plt.ylabel(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=20)
        fig.legend(bbox_to_anchor=(0.9, 0.3), fontsize=18, frameon=False)
        plt.xlim(9.5, 10)
        plt.yscale('log')
        plt.savefig('figures/merger_rates_TNG_zoomed.pdf', format="pdf", bbox_inches='tight')

    if showplot==True:
        plt.show()


def compare_BBH_data_and_model_rates(data_dir, model_rates, data_rates, error_ylim=[], plot_merger_rates=True, plot_logscale=False, showplot=True, showLVK=True):

    fig = plt.figure(layout='constrained',figsize=[13, 10])
    ax = fig.add_subplot()
    ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)

    for i, rfile in enumerate(model_rates):
        with h5.File(data_dir + rfile ,'r') as File:
            redshift      = File[list(File.keys())[0]]['redshifts'][()]
            formation_rate = File[list(File.keys())[0]]['formation_rate'][()]
            merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]

        with h5.File(data_dir + data_rates[i] ,'r') as File:
            data_redshift      = File[list(File.keys())[0]]['redshifts'][()]
            data_formation_rate = File[list(File.keys())[0]]['formation_rate'][()]
            data_merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]

        if plot_merger_rates == True:
            #Plot merger rates
            total_merger_rate = np.sum(merger_rate, axis=0)
            total_data_merger_rate = np.sum(data_merger_rate, axis=0)
            ax.plot(data_redshift, total_data_merger_rate, label='TNG%s'%labels[i], lw=8, c=data_colors[i])
            ax.plot(redshift, total_merger_rate, lw=3, c=model_colors[i], ls='--')
            fractionalerr = total_merger_rate/total_data_merger_rate
            ax_x.plot(data_redshift, fractionalerr, lw=4, c=data_colors[i])
            print("The TNG%s model merger rate at z=%s is: "%(labels[i], redshift[0]), total_merger_rate[0])
            print("The TNG%s data merger rate at z=%s is: "%(labels[i], data_redshift[0]), total_data_merger_rate[0])

        else:
            #Plot formation rates
            total_formation_rate = np.sum(formation_rate, axis=0)
            total_data_formation_rate = np.sum(data_formation_rate, axis=0)
            ax.plot(data_redshift, total_data_formation_rate, label='TNG%s'%labels[i], lw=8, c=data_colors[i])
            ax.plot(redshift, total_formation_rate, lw=3, c=model_colors[i], ls='--')
            fractionalerr = total_formation_rate/total_data_formation_rate
            ax_x.plot(data_redshift, fractionalerr, lw=4, c=data_colors[i])
            print("The TNG%s model formation rate at z=%s is: "%(labels[i], redshift[0]), total_formation_rate[0])
            print("The TNG%s data formation rate at z=%s is: "%(labels[i], data_redshift[0]), total_data_formation_rate[0])


    if showLVK == True:
        lvkdata = plt.gca()
        lvkdata.add_patch(matplotlib.patches.Rectangle((0.02, 14), 0.4, 12, facecolor="none", ec='gray', lw=3, zorder=6, label='GWTC-4'))

    ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    x = [-0.0001]
    y1 = [1]
    y2 = [1]
    ax.plot(x, y1, c='black', ls = '-', lw=5, label='TNG simulation')
    ax.plot(x, y2, c='black', ls = '--', lw=3, label='Analytical fit')
    ax.set_xlim(0, 14)
    ax.set_ylim(1e-3, 5e2)
    ax.set_xlabel('Redshift $z$', fontsize=35)
    if plot_merger_rates == True:
        ax.set_ylabel(r'$\mathcal{R}(z) \ [\rm Gpc^{-3} \ yr^{-1}]$', fontsize=35)
        ax_x.set_ylabel(r'$\mathcal{R}(z)_\mathrm{fit}/\mathcal{R}(z)_\mathrm{sim}$', fontsize=20)
        #fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=25, loc="upper right", frameon=False)
        fig.legend(bbox_to_anchor=(0.15, 0.1), fontsize=25, loc="lower left", frameon=False)
    else:
        ax.set_ylabel(r'Formation rate $[\rm Gpc^{-3} \ yr^{-1}]$', fontsize=35)
        ax_x.set_ylabel(r'$\mathcal{R_\mathrmform}}(z)_\mathrm{fit}/\mathcal{R_\mathrmform}}(z)_\mathrm{sim}$', fontsize=25)
        fig.legend(bbox_to_anchor=(0.15, 0.1), fontsize=25, loc="lower left", frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    if plot_logscale == True:
        ax.set_yscale('log')

    ax_x.set_ylim(error_ylim[0], error_ylim[1])
    ax_x.tick_params(axis='y', which='major', labelsize=20)
    ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
    ax_x.tick_params(length=10, width=2, which='major')
    ax_x.tick_params(length=5, width=1, which='minor')
    ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_x.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
    ax_x.set_yscale('log')

    ax2 = ax_x.twiny()
    redshift_tick_list = [0, 1, 2, 6, 10, 14]#[0,0.1, 0.25, 0.5, 1.0, 10]
    ax2.set_xticks([z for z in redshift_tick_list])
    ax2.set_xticklabels(['${:.1f}$'.format(cosmo.lookback_time(z).value) for z in redshift_tick_list], fontsize = 25)
    ax2.set_xlabel('Lookback time [Gyr]', fontsize = 35, labelpad=15)
    ax2.tick_params(axis='both', which='major', labelsize=23)
    ax2.tick_params(length=10, width=3, which='major')

    if plot_merger_rates == True:
        plt.savefig('figures/merger_rates_datavsmodel_TNG.pdf', format="pdf", bbox_inches='tight', dpi=300)

    else:
        plt.savefig('figures/formation_rates_datavsmodel_TNG.pdf', format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def residuals_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, ylim = [], plot_merger_rates=True, showplot=True):

    fig, ax = plt.subplots(figsize = (13, 8))

    #gridline for reference
    xs = np.arange(0, 20)
    ys = np.zeros(len(xs))
    plt.plot(xs, ys, c='lightgray', ls='-', lw=3)

    for i, rfile in enumerate(model_rates):
        with h5.File(data_dir + rfile ,'r') as File:
            formation_rate = File[list(File.keys())[0]]['formation_rate'][()]
            merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]

        with h5.File(data_dir + data_rates[i] ,'r') as File:
            data_redshift      = File[list(File.keys())[0]]['redshifts'][()]
            data_formation_rate = File[list(File.keys())[0]]['formation_rate'][()]
            data_merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]

        if plot_merger_rates == True:
            #Plot merger rates
            total_merger_rate = np.sum(merger_rate, axis=0)
            total_data_merger_rate = np.sum(data_merger_rate, axis=0)
            percenterr = abs(total_data_merger_rate - total_merger_rate)/total_merger_rate * 100
            plt.plot(data_redshift, percenterr, label='TNG%s'%labels[i], lw=4, c=data_colors[i])

        else:
            #Plot formation rates
            total_formation_rate = np.sum(formation_rate, axis=0)
            total_data_formation_rate = np.sum(data_formation_rate, axis=0)
            percenterr = abs(total_data_formation_rate - total_formation_rate)/total_formation_rate * 100
            plt.plot(data_redshift, percenterr, label='TNG%s'%labels[i], lw=4, c=data_colors[i])

    ax.set_xlim(0, 13.5)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('Redshift $z$', fontsize=35)
    if plot_merger_rates == True:
        ax.set_ylabel(r'$\mathcal{R}(z)$ \% error', fontsize=35)
        fig.legend(bbox_to_anchor=(0.15, 0.88), fontsize=25, loc="upper left", frameon=False)
        #fig.legend(bbox_to_anchor=(0.15, 0.1), fontsize=25, loc="lower left", frameon=False)
    else:
        ax.set_ylabel(r'Formation rate \% error', fontsize=35)
        fig.legend(bbox_to_anchor=(0.15, 0.88), fontsize=25, loc="upper left", frameon=False)
        #fig.legend(bbox_to_anchor=(0.15, 0.1), fontsize=25, loc="lower left", frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_yscale('log')
    
    ax2 = ax.twiny()
    redshift_tick_list = [0, 1, 2, 6, 10, 14]#[0,0.1, 0.25, 0.5, 1.0, 10]
    ax2.set_xticks([z for z in redshift_tick_list])
    ax2.set_xticklabels(['${:.1f}$'.format(cosmo.lookback_time(z).value) for z in redshift_tick_list], fontsize = 24)
    ax2.set_xlabel('Lookback time [Gyr]', fontsize = 35, labelpad=15)
    #ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
    #ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
    ax2.tick_params(axis='both', which='major', labelsize=23)
    ax2.tick_params(length=10, width=3, which='major')

    if plot_merger_rates == True:
        plt.savefig('figures/merger_rates_res_datavsmodel_TNG.pdf', format="pdf", bbox_inches='tight', dpi=300)
    else:
        plt.savefig('figures/formation_rates_res_datavsmodel_TNG.pdf', format="pdf", bbox_inches='tight', dpi=300)

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


    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
    ax.fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)

    """
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

    #Callister&Farr 2024 non-parametric fit to LVK
    input_fname = data_dir + "ar_lnm1_q_summary.hdf"
    with h5.File(input_fname, "r") as f:
        m1s = f['posterior/m1s'][()]
        dR_dlnm1s = f['posterior/dR_dlnm1s'][()]

    ax.plot(m1s,np.median(dR_dlnm1s,axis=1),color='gray',lw=1.8, zorder=1, label='GWTC-3')
    ax.fill_between(m1s, np.quantile(dR_dlnm1s,0.05,axis=1), np.quantile(dR_dlnm1s,0.95,axis=1), alpha=0.14,color='gray',zorder=0)
    """
    nplot=0

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath, i)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
    
        x_KDE = np.arange(0.1,50.,0.1)
        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
        plot_lines.append(ax.plot(x_KDE, KDEy_vals, label = 'TNG'+labels[nplot], color=data_colors[nplot], lw= 4,  zorder =i+1,ls = '-'))
    
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
    fig.savefig('figures/massdist_%s_z%s.pdf'%(channel_string, z), format="pdf", bbox_inches='tight', dpi=300)

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

    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
    ax.fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
    fig.savefig('figures/massdist_TNG%s_%s_redshift_evol.pdf'%(tng, channel_string), format="pdf", bbox_inches='tight', dpi=300)

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
            colors.append(cmap_blue(np.linspace(0,1,len(z))))
        elif i==100:
            nplot.append(1)
            colors.append(cmap_pink(np.linspace(0,1,len(z))))
        else:
            nplot.append(2)
            colors.append(cmap_green(np.linspace(0,1,len(z))))

    for i in rates:
        TNGpaths.append('/'+i)
    for i in fit_param_vals:
        rate_keys.append('Rates_mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s_zBinned'%(np.round(i[0], 3), np.round(i[1], 3), np.round(i[4], 3), np.round(i[2], 3), np.round(i[3], 3), np.round(i[5],3), np.round(i[6], 3), np.round(i[7], 3), np.round(i[8],3)))

    bins = np.arange(0.,55,2.5)
    x_lim=(0.,49)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'
    channel_names = [r'all channels', r'stable channel', r'CE channel']

    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    for i in range(3):
        for j in range(3):
            if i==j==0:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
            else:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1)
            ax[i, j].fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak, zorder=0)
    """
    #Callister&Farr 2024 non-parametric fit to LVK
    input_fname = data_dir + "ar_lnm1_q_summary.hdf"
    with h5.File(input_fname, "r") as f:
        m1s = f['posterior/m1s'][()]
        dR_dlnm1s = f['posterior/dR_dlnm1s'][()]
    for i in range(3):
        for j in range(3):
            if i==j==0:
                ax[i, j].plot(m1s,np.median(dR_dlnm1s,axis=1),color='gray',lw=1.8, zorder=1, label='GWTC-3')
            else:
                ax[i, j].plot(m1s,np.median(dR_dlnm1s,axis=1),color='gray',lw=1.8, zorder=1)
            ax[i,j].fill_between(m1s, np.quantile(dR_dlnm1s,0.05,axis=1), np.quantile(dR_dlnm1s,0.95,axis=1), alpha=0.14,color='gray',zorder=0)
    """

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))
    channels = ['all', 'stable', 'CE']

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(TNGpaths):
        print('Path to TNG', tngpath)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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

                if i==0:
                    ax[i, j].set_title(channel_names[j], fontsize = 25)

                # plot values
                ax[i, j].set_xlim(x_lim)
                ax[i, j].set_ylim(y_lim)
                ax[i, j].tick_params(axis='both', which='major', labelsize=20)
                ax[i, j].xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax[i, j].xaxis.set_minor_locator(ticker.MultipleLocator(5))
                #if i==2 and j!=2:
                #    ax[i,j].set_xticklabels(['0', '10', '20', '30', '40', ''])
                ax[i, j].tick_params(length=15, width=3, which='major')
                ax[i, j].tick_params(length=10, width=2, which='minor')
                ax[i, j].set_yscale('log')
    
    fig.supxlabel(xlabel, y=0.05, fontsize = 30)
    fig.supylabel(ylabel, x=0.03, fontsize = 30)

    x = [-0.0001]
    y1 = [0.0001]
    y2 = [0.0001]
    y3 = [0.0001]
    plt.plot(x, y1, c=data_colors[0], ls = '-', lw=6, label=r'TNG50-1')
    plt.plot(x, y2, c=data_colors[1], ls = '-', lw=6, label=r'TNG100-1')
    plt.plot(x, y3, c=data_colors[2], ls = '-', lw=6, label=r'TNG300-1')

    bounds = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]
    bounds = [0.05, 0.35, 0.65, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    boundlabels = ['0.2', '0.5', '1', '2', '3', '4', '5', '6', '7', '8']
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap_gray.N)
    cbar_ax = fig.add_axes([0.53, 0.93, 0.35, 0.02]) #location, size of colorbar
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_gray.reversed()), cax=cbar_ax, ticks=[0.2, 0.5, 1.05, 2, 3, 4, 5, 6, 7, 8], orientation='horizontal')
    cbar.set_ticklabels(boundlabels)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'$z$', fontsize=25)
    cbar.ax.xaxis.set_label_position('top')
                
    fig.legend(bbox_to_anchor=(0.5, 0.98), fontsize=22, ncol = 2)
    fig.savefig('figures/massdist_allTNGs_redshift_evol_model.pdf', format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def plot_BBH_mass_dist_formation_channels(data_rates, model_rates, tng, version, z = [0.2, 1, 2, 4, 6, 8], showplot=True):
    #get rate file names and rate keys
    paths = ['/'+data_rates, '/'+model_rates]
    colors  = []

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize = (14, 16))
    fig.subplots_adjust(wspace=0, hspace=0)

    colormap = sns.color_palette('rocket_r', as_cmap=True)
    colors.append(colormap(np.linspace(0.2, 0.9, len(z))))
    #colors.append(colormap2(np.linspace(0.2, 0.9, len(z))))
    #colors.append(cmap_gray(np.linspace(0, 1, len(z))))
    colors.append(colormap(np.linspace(0.2, 0.9, len(z))))
    linestyles = ['-', '--']
    linewidths = [4, 3]

    bins = np.arange(0.,55,2.5)
    x_lim=(0.,49)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3} \ yr^{-1} \ M^{-1}_{\odot}]}$'
    titles = [r'TNG simulation', r'Analytical fit']
    channel_names = [r'all channels', r'stable channel', r'CE channel']
    
    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    for i in range(3):
        for j in range(2):
            if i==j==0:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
            else:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1)
            ax[i, j].fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak, zorder=0)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))
    channels = ['all', 'stable', 'CE']

    ####################################################
    #Loop over TNGs
    for i, tngpath in enumerate(paths):

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
                line, = ax[j, i].plot(x_KDE, KDEy_vals, color=colors[i][m], lw= linewidths[i], ls = linestyles[i])
                if i == 1:
                    line.set_dashes([5, 1])

                if j==0:
                    ax[j, i].set_title(titles[i], fontsize = 25)

                # plot values
                ax[j, i].set_xlim(x_lim)
                ax[j, i].set_ylim(y_lim)
                ax[j, i].tick_params(axis='both', which='major', labelsize=25)
                ax[j, i].xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax[j, i].xaxis.set_minor_locator(ticker.MultipleLocator(5))
                ax[j, i].tick_params(length=15, width=3, which='major')
                ax[j, i].tick_params(length=10, width=2, which='minor')
                ax[j, i].set_yscale('log')
                ax[j, i].text(0.73, 0.79, channel_names[j], ha = 'center', transform=ax[j,i].transAxes, size = 22)

                #if i==0:
                #    ax[j, i].set_ylabel(channel_names[j], fontsize = 25)
    
    fig.supxlabel(xlabel, y=0.04, fontsize = 30)
    fig.supylabel(ylabel, x=0.01, fontsize = 30)

    #x = [-0.0001]
    #y1 = [0.0001]
    #y2 = [0.0001]
    #plt.plot(x, y1, c='black', ls = '-', lw=4, label=r'TNG simulation')
    #plt.plot(x, y2, c='black', ls = '--', lw=3, label=r'Analytical fit', dashes=[5, 1])

    bounds = [0.05, 0.35, 0.65, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    boundlabels = ['0.2', '0.5', '1', '2', '3', '4', '5', '6', '7', '8']
    norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N * 0.9)
    cbar_ax = fig.add_axes([0.32, 0.93, 0.35, 0.02]) #location, size of colorbar
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap.reversed()), cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0.2, 0.5, 1.05, 2, 3, 4, 5, 6, 7, 8])
    cbar.set_ticklabels(boundlabels)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.tick_params(which="minor",size=0)
    cbar.set_label(r'$z_\mathrm{merger}$', fontsize=25, labelpad=10)
    cbar.ax.xaxis.set_label_position('top')

    #norm = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
    #cbar_ax = fig.add_axes([0.53, 0.95, 0.35, 0.02]) #location, size of colorbar
    #cbar2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap.reversed()), cax=cbar_ax, orientation='horizontal',)
    #cbar2.ax.tick_params(labelbottom=False, labeltop=False) 

    #fig.legend(bbox_to_anchor=(0.92, 0.89), fontsize=22, frameon=False)
    fig.legend(bbox_to_anchor=(0.5, 0.88), fontsize=22, frameon=False)
    #fig.legend(bbox_to_anchor=(0.5, 0.98), fontsize=22, ncol = 2)
    fig.savefig('figures/massdist_TNG%s_%s_formationchannels.pdf'%(tng, version), format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()



def compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = 0.2, showplot=True):
    #get rate file names and rate keys
    TNGpaths = []
    TNGpaths_data = []

    for i in model_rates:
        TNGpaths.append('/'+i)
    for i in data_rates:
        TNGpaths_data.append('/'+i)

    fig, ax = plt.subplots(figsize = (12, 8))
    bins = np.arange(0.,55,2.5)
    plot_lines = []
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    ax.plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
    ax.fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak,zorder=0)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    #Loop over TNGs
    nplot=0
    for i, tngpath in enumerate(TNGpaths_data):
        print('Path to TNG', tngpath, i)

        #Reading Rate data 
        with h5.File(data_dir + tngpath ,'r') as File:
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
            redshifts = File[list(File.keys())[0]]['redshifts'][()]
            DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
            intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
        ax.plot(x_KDE, KDEy_vals, color=model_colors[nplot], lw= 3, ls='--')
    
        nplot += 1
        
    #########################################
    # plot values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    #legend
    x = [-0.0001]
    y1 = [1]
    y2 = [1]
    plt.plot(x, y1, c='black', ls = '-', lw=6, label='TNG simulation')
    plt.plot(x, y2, c='black', ls = '--', lw=3, label='Analytical fit')

    ax.set_xlabel(xlabel, fontsize = 30)
    ax.set_ylabel(ylabel, fontsize = 30)
    ax.set_yscale('log')
    if channel_string=='all':
        plt.text(0.03, 0.88, '$\mathrm{%s \ channels}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)
    else:
        plt.text(0.03, 0.88, '$\mathrm{%s \ channel}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)

    fig.legend(bbox_to_anchor=(0.92, 0.88), fontsize=30, frameon=False)
    fig.savefig('figures/massdist_modelvsdata_%s_z%s.pdf'%(channel_string, z),format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()

def compare_BBH_data_and_model_mass_dist_over_z(model_rates, data_rates, only_stable = True, only_CE = True, channel_string='all', z = [0.2, 1, 2, 4, 6, 8], showplot=True):
    #get rate file names and rate keys
    TNGpaths = []
    TNGpaths_data = []

    for i in model_rates:
        TNGpaths.append('/'+i)
    for i in data_rates:
        TNGpaths_data.append('/'+i)
    
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize = (14, 16))
    fig.subplots_adjust(wspace=0, hspace=0)

    bins = np.arange(0.,55,2.5)
    x_lim=(0.,50)
    y_lim = (1e-2,40)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3} \ yr^{-1} \ M^{-1}_{\odot}]}$'

    # GWTC-4 Spline mass distribution
    color_plpeak = 'grey'#'#1f78b4'
    input_fname = data_dir+'BBHMassSpinRedshift_BSplineIID.h5'
    bptp_o4 = popsummary.popresult.PopulationResult(input_fname)
    with h5.File(input_fname, "r") as f:
        param = 'rate_vs_mass_1_at_z0-2' #'mass_1'
        dat = bptp_o4.get_rates_on_grids(param)
        bptp_m1 = dat[0][0]
        bptp_m1_pdfs = dat[1]
        mass_1_lower = np.percentile(bptp_m1_pdfs, 5, axis=0)
        mass_1_upper = np.percentile(bptp_m1_pdfs, 95, axis=0)
    # plot the max posterior and the 95th percentile
    for i in range(3):
        for j in range(2):
            if i==j==0:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1, label="GWTC-4")
            else:
                ax[i, j].plot(bptp_m1, np.median(bptp_m1_pdfs, axis=0), lw=1.8, color=color_plpeak, zorder=1)
            ax[i, j].fill_between(bptp_m1, mass_1_lower, mass_1_upper, alpha=0.14,color=color_plpeak, zorder=0)
    """

    #Callister&Farr 2024 non-parametric fit to LVK
    input_fname = data_dir + "ar_lnm1_q_summary.hdf"
    with h5.File(input_fname, "r") as f:
        m1s = f['posterior/m1s'][()]
        dR_dlnm1s = f['posterior/dR_dlnm1s'][()]
    for i in range(3):
        for j in range(2):
            if i==j==0:
                ax[i, j].plot(m1s,np.median(dR_dlnm1s,axis=1),color='gray',lw=1.8, zorder=1, label='GWTC-3')
            else:
                ax[i, j].plot(m1s,np.median(dR_dlnm1s,axis=1),color='gray',lw=1.8, zorder=1)
            ax[i,j].fill_between(m1s, np.quantile(dR_dlnm1s,0.05,axis=1), np.quantile(dR_dlnm1s,0.95,axis=1), alpha=0.14,color='gray',zorder=0)
    """
    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    #Loop over TNGs
    for m, redshift in enumerate(z):
        print("Plotting redshift z = ", redshift)

        if m%2==0: 
            a2=0
        else:
            a2=1
        if m < 2:
            a1 = 0
        elif (m >= 2) and (m < 4):
            a1 = 1
        else:
            a1 = 2

        plot_lines = []
        nplot=0
    
        for i, tngpath in enumerate(TNGpaths_data):
            print('Path to TNG', tngpath, i)

            #Reading Rate data 
            with h5.File(data_dir + tngpath ,'r') as File:
                redshifts = File[list(File.keys())[0]]['redshifts'][()]
                DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
                intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
            i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
            Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
            Weights[Weights < 0] = 0
    
            # Get the Hist    
            hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

            # And the KDE
            kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

            x_KDE = np.arange(0.1,50.,0.1)
            KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
            plot_lines.append(ax[a1, a2].plot(x_KDE, KDEy_vals, color=data_colors[nplot], lw=6, ls = '-'))
    
            nplot += 1

        ####################################################
        #Loop over TNGs
        nplot = 0
        for i, tngpath in enumerate(TNGpaths):
            print('Path to TNG', tngpath, i)

            #Reading Rate data 
            with h5.File(data_dir + tngpath ,'r') as File:
                redshifts = File[list(File.keys())[0]]['redshifts'][()]
                DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
                intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
            i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
            Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
            Weights[Weights < 0] = 0
    
            # Get the Hist    
            hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

            # And the KDE
            kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

            x_KDE = np.arange(0.1,50.,0.1)
            KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
            ax[a1, a2].plot(x_KDE, KDEy_vals, color=model_colors[nplot], lw= 3, ls='--')
    
            nplot += 1

        # plot values
        ax[a1, a2].set_xlim(x_lim)
        ax[a1, a2].set_ylim(y_lim)
        ax[a1, a2].tick_params(axis='both', which='major', labelsize=25)
        ax[a1, a2].xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax[a1, a2].tick_params(length=15, width=3, which='major')
        ax[a1, a2].tick_params(length=10, width=2, which='minor')
        ax[a1, a2].set_yscale('log')

        if a1==2 and a2!=2:
            ax[a1, a2].set_xticklabels(['0', '10', '20', '30', '40', ''])

        ax[a1, a2].text(30, 10, '$z_\mathrm{merger}=%s$'%(redshift), ha = 'left', size = 25)
        
    #########################################

    fig.supxlabel(xlabel, y=0.04, fontsize = 30)
    fig.supylabel(ylabel, x=0.01, fontsize = 30)
    if channel_string=='all':
        fig.suptitle('all channels', fontsize=30)
    elif channel_string=='stable':
        fig.suptitle('stable channel', fontsize=30)
    else:
        fig.suptitle('CE channel', fontsize=30)

    #legend
    x = [-0.0001]
    y1 = [0.0001]
    y2 = [0.0001]
    y3 = [0.0001]
    plt.plot(x, y1, c=data_colors[0], ls = '-', lw=6, label=r'TNG50-1')
    plt.plot(x, y2, c=data_colors[1], ls = '-', lw=6, label=r'TNG100-1')
    plt.plot(x, y3, c=data_colors[2], ls = '-', lw=6, label=r'TNG300-1')
    y1 = [1]
    y2 = [1]
    plt.plot(x, y1, c='black', ls = '-', lw=6, label='TNG simulation')
    plt.plot(x, y2, c='black', ls = '--', lw=3, label='Analytical fit')

    #ax.set_xlabel(xlabel, fontsize = 30)
    #ax.set_ylabel(ylabel, fontsize = 30)
    #ax.set_yscale('log')
    #if channel_string=='all':
    #    plt.text(0.03, 0.88, '$\mathrm{%s \ channels}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)
    #else:
    #    plt.text(0.03, 0.88, '$\mathrm{%s \ channel}$\nz=%s'%(channel_string, z), ha = 'left', transform=ax.transAxes, size = 25)

    fig.legend(bbox_to_anchor=(0.86, 0.96), fontsize=25, ncol = 3)
    fig.savefig('figures/massdist_modelvsdata.pdf', format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def residuals_BBH_data_and_model_mass_dist(model_rates, data_rates, only_stable = True, only_CE = True, channel_string='all', z = [0.2, 1, 2, 4, 6, 8], showplot=True):
    #get rate file names and rate keys
    TNGpaths = []
    TNGpaths_data = []

    for i in model_rates:
        TNGpaths.append('/'+i)

    for i in data_rates:
        TNGpaths_data.append('/'+i)

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize = (14, 16))
    fig.subplots_adjust(wspace=0, hspace=0)

    bins = np.arange(0.,55,2.5)
    x_lim=(0.,50)
    y_lim = (1.5e-1,1e2)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1}}}_\mathrm{fit}/\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1}}}_\mathrm{sim}$'

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))

    #Loop over TNGs
    for m, redshift in enumerate(z):
        print("Plotting redshift z = ", redshift)

        if m%2==0: 
            a2=0
        else:
            a2=1
        if m < 2:
            a1 = 0
        elif (m >= 2) and (m < 4):
            a1 = 1
        else:
            a1 = 2

        nplot=0
    
        for i, tngpath in enumerate(TNGpaths_data):
            print('Path to TNG', tngpath, i)

            #Reading Rate data 
            with h5.File(data_dir + tngpath ,'r') as File:
                redshifts = File[list(File.keys())[0]]['redshifts'][()]
                DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
                intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
            i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
            Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
            Weights[Weights < 0] = 0
    
            # Get the Hist    
            hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

            # And the KDE
            kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

            x_KDE = np.arange(0.1,50.,0.1)
            KDEy_vals_data =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
            nplot += 1

        ####################################################
        #Loop over TNGs
        nplot = 0
        for i, tngpath in enumerate(TNGpaths):
            print('Path to TNG', tngpath, i)

            #Reading Rate data 
            with h5.File(data_dir + tngpath ,'r') as File:
                redshifts = File[list(File.keys())[0]]['redshifts'][()]
                DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
                intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]

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
            i_redshift = np.where(redshifts == redshift)[0][0] # Rate at redshift 0.2
            Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
            Weights[Weights < 0] = 0
    
            # Get the Hist    
            hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

            # And the KDE
            kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

            x_KDE2 = np.arange(0.1,50.,0.1)
            KDEy_vals_sim =  kernel(x_KDE2)*sum(hist) #re-normalize the KDE
            ax[a1, a2].plot(x_KDE, KDEy_vals_sim/KDEy_vals_data, color=data_colors[nplot], lw=4, ls= '-')
            nplot += 1

        # plot values
        ax[a1, a2].set_xlim(x_lim)
        ax[a1, a2].set_ylim(y_lim)
        ax[a1, a2].tick_params(axis='both', which='major', labelsize=25)
        ax[a1, a2].xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax[a1, a2].tick_params(length=15, width=3, which='major')
        ax[a1, a2].tick_params(length=10, width=2, which='minor')
        ax[a1, a2].set_yscale('log')

        if a1==2 and a2!=2:
            ax[a1, a2].set_xticklabels(['0', '10', '20', '30', '40', ''])

        ax[a1, a2].text(5, 40, '$z_\mathrm{merger}=%s$'%(redshift), ha = 'left', size = 25)
        ax[a1, a2].axhline(y=1, linewidth=1, color='gray', zorder=0)
        
    #########################################

    fig.supxlabel(xlabel, y=0.04, fontsize = 30)
    fig.supylabel(ylabel, x=0.01, fontsize = 30)
    if channel_string=='all':
        fig.suptitle('all channels', fontsize=30, y=0.96)
    elif channel_string=='stable':
        fig.suptitle('stable channel', fontsize=30, y=0.96)
    else:
        fig.suptitle('CE channel', fontsize=30, y=0.96)

    #legend
    x = [-0.0001]
    y1 = [0.0001]
    y2 = [0.0001]
    y3 = [0.0001]
    plt.plot(x, y1, c=data_colors[0], ls = '-', lw=4, label=r'TNG50-1')
    plt.plot(x, y2, c=data_colors[1], ls = '-', lw=4, label=r'TNG100-1')
    plt.plot(x, y3, c=data_colors[2], ls = '-', lw=4, label=r'TNG300-1')

    fig.legend(bbox_to_anchor=(0.82, 0.94), fontsize=25, ncol = 3)
    fig.savefig('figures/massdist_modelvsdata_res.pdf', format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()

def plot_BBH_mass_Z_z(tngpath, COMPASpath, tng, data_rates=None, only_stable = True, only_CE = True, channel_string='all', z_merger=0.2, max_redshift=14, 
                      redshift_step=0.05, z_first_SF=14, z_form = [0, 1, 2, 6, 10, 14], Z_zams = [0.1, 0.01, 0.001, 0.0001], showplot=True, plot_total_dist=True):
    
    fig, ax = plt.subplots(figsize = (12, 8))
    bins = np.arange(0.,55,2.5)
    x_lim=(0.,50)
    y_lim = (1e-4, 1e2)
    xlabel = r'$M_{\mathrm{BH, 1}} \ \rm [M_{\odot}]$'
    ylabel = r'$\frac{d\mathcal{R}}{dM_{\mathrm{BH, 1} }} \ \mathrm{[Gpc^{-3}yr^{-1}M^{-1}_{\odot}]}$'

    colors = colormap(np.linspace(0.1, 0.8, len(z_form)-1))

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))
    ####################################################
    #Reading Rate data 
    with h5.File(data_dir + '/' + tngpath ,'r') as File:
        redshifts = File[list(File.keys())[0]]['redshifts'][()]
        DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
        intrinsic_rate_density = File[list(File.keys())[0]]['merger_rate'][()]
        seeds = File[list(File.keys())[0]]['SEED'][()]
        #print(list(File[list(File.keys())[0]].keys()))

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
    merging_BBH_seeds =  seeds[BBH_bool * not_CHE  * channel_bool]

    if plot_total_dist==True:
        #Calculate average rate density per z-bin
        x_vals              = merging_BBH['M_moreMassive']
        i_redshift = np.where(redshifts == z_merger)[0][0] # Rate at redshift 0.2
        Weights             = Red_intr_rate_dens[:, i_redshift]#crude_rate_density[:,0]
        Weights[Weights < 0] = 0
    
        # Get the Hist    
        hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

        # And the KDE
        kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)

        x_KDE = np.arange(0.1,50.,0.1)
        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
        #ax.plot(x_KDE, KDEy_vals, color='black', lw=2, ls = '-', zorder=0, label='total')
        ax.fill_between(x_KDE, x_KDE*0, KDEy_vals, alpha=0.1,color='gray',zorder=0)

    #read in formation redshift and metallicity for all binaries
    with h5.File(data_dir + '/' + COMPASpath ,'r') as File:
            metallicities_compas = File['BSE_Double_Compact_Objects']['Metallicity@ZAMS(1)'][()] #same metallicity for both stars
            COMPAS_delay_times = File['BSE_Double_Compact_Objects']['Coalescence_Time'][()]#Myr
            time = File['BSE_Double_Compact_Objects']['Time'][()] #Myr
            seeds_compas = File['BSE_Double_Compact_Objects']['SEED'][()]

    #filter formation redshift and metallicity by the seeds we want
    mask_merging_DCOs = np.isin(seeds_compas, merging_BBH_seeds)
    seeds_DCOs = seeds_compas[mask_merging_DCOs]

    metallicities_compas = metallicities_compas[mask_merging_DCOs]
    COMPAS_delay_times = COMPAS_delay_times[mask_merging_DCOs]
    time = time[mask_merging_DCOs]

    #from fastcosmicintegration, need to calculate merger rate and redshift
    redshifts = np.arange(0, max_redshift + redshift_step, redshift_step)
    times = cosmo.age(redshifts).to(u.Myr).value
    times_to_redshifts = interp1d(times, redshifts)
    age_first_sfr = cosmo.age(z_first_SF).to(u.Myr).value

    age_merger = cosmo.age(z_merger).to(u.Myr).value
    age_of_formation = age_merger - COMPAS_delay_times
    age_of_formation_mask = (age_of_formation >= age_first_sfr) #remove binaries that cannot merge at z=0.2 (too long delay time)
    age_of_formation_masked = age_of_formation[age_of_formation_mask]
    z_of_formation = times_to_redshifts(age_of_formation_masked)

    #remove all binaries that cannot merge at z=0.2
    seeds_DCOs = seeds_DCOs[age_of_formation_mask]
    metallicities_compas = metallicities_compas[age_of_formation_mask]
    merging_BBH_seeds = merging_BBH_seeds[age_of_formation_mask]
    merging_BBH = merging_BBH[age_of_formation_mask]
    Red_intr_rate_dens = Red_intr_rate_dens[age_of_formation_mask]

    for i, redshift_form in enumerate(z_form):
        if i < len(z_form)-1:

            print("Plotting formation redshift %s <= z < %s"%(redshift_form, z_form[i+1]))

            #filter by formation redshift bin
            z_mask_low = (z_of_formation >= redshift_form)
            z_mask_high = (z_of_formation < z_form[i+1])
            if z_form[i+1] <= z_merger:
                print("Can't have binaries with z_formation <= z_merger, skipping redshift bin")
                continue
            z_mask = ((z_mask_low==True) & (z_mask_high==True))
            z_seeds = seeds_DCOs[z_mask]
            metallicities_compas_masked = metallicities_compas[z_mask]

            z_seeds_mask = np.isin(merging_BBH_seeds, z_seeds)
            z_seeds_DCOs = merging_BBH_seeds[z_seeds_mask]    

            if len(z_seeds_DCOs) > 0:

                counter=0
                for j, metalbin in enumerate(Z_zams):
                    if j < len(Z_zams)-1:
                        #for each metallicity, plot mass distribution
                        #to do this, filter data by metallicity at ZAMS
                        print("Plotting metallicity bin %s <= Z < %s"%(metalbin, Z_zams[j+1]))

                        #create mask for Z bin
                        Z_mask_low = (metallicities_compas_masked <= metalbin)
                        Z_mask_high = (metallicities_compas_masked > Z_zams[j+1])
                        Z_mask = ((Z_mask_low==True) & (Z_mask_high==True))
                        Z_seeds = z_seeds[Z_mask]

                        Z_seeds_mask = np.isin(seeds_DCOs, Z_seeds)
                        merging_BBH_masked = merging_BBH[Z_seeds_mask]
                        Red_intr_rate_dens_masked = Red_intr_rate_dens[Z_seeds_mask]

                        #Calculate average rate density per z-bin at the merger redshift
                        x_vals  = merging_BBH_masked['M_moreMassive']
                        i_redshift = np.where(redshifts == z_merger)[0][0] # Rate at redshift of merger
                        Weights = Red_intr_rate_dens_masked[:, i_redshift]#crude_rate_density[:,0]
                        Weights[Weights < 0] = 0
                        if np.sum(Weights) == 0:
                            print("No binaries merging at z=%s in metallicity bin %s <= Z < %s"%(z_merger, Z_zams[j+1], metalbin))
                            continue

                        # Get the Hist    
                        hist, bin_edge = np.histogram(x_vals, weights = Weights, bins=bins)

                        # And the KDE
                        try: kernel = stats.gaussian_kde(x_vals, bw_method='scott', weights=Weights)
                        except ValueError: continue
    
                        x_KDE = np.arange(0.1,50.,0.1)
                        KDEy_vals =  kernel(x_KDE)*sum(hist) #re-normalize the KDE
                        if counter == 0:
                            ax.plot(x_KDE, KDEy_vals, label = r'%s $\leq z_{form} <$ %s'%(redshift_form, z_form[i+1]), color = colors[i], lw= 3, ls = linestyles2[j])
                        else:
                            ax.plot(x_KDE, KDEy_vals, color = colors[i], lw= 3, ls = linestyles2[j])
                        counter+=1

            else:
                print("No binaries in redshift bin %s <= z < %s"%(redshift_form, z_form[i+1]))
                continue

    #########################################
    # plot values
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    x = [-0.0001]
    y1 = [0.0001]
    plt.plot(x, y1, c='black', ls = '-', lw=3, label=r'$%s \leq Z < %s$'%(Z_zams[1], Z_zams[0]))
    if len(Z_zams) >= 3:
        plt.plot(x, y1, c='black', ls = '--', lw=3, label=r'$%s \leq Z < %s$'%(Z_zams[2], Z_zams[1],))
        if len(Z_zams) >= 4:
            plt.plot(x, y1, c='black', ls = ':', lw=3, label=r'$%s \leq Z < %s$'%(Z_zams[3], Z_zams[2]))
    
    # Channel
    if tng==50:
        plt.text(0.12, 0.92, 'TNG%s'%labels[0], ha = 'center', transform=ax.transAxes, size = 30)
    elif tng==100:
        plt.text(0.12, 0.92, 'TNG%s'%labels[1], ha = 'center', transform=ax.transAxes, size = 30)
    elif tng==300:
        plt.text(0.12, 0.92, 'TNG%s'%labels[2], ha = 'center', transform=ax.transAxes, size = 30)
    
    ax.set_xlabel(xlabel, fontsize = 30)
    ax.set_ylabel(ylabel, fontsize = 30)
    ax.set_yscale('log')
    fig.legend(bbox_to_anchor=(0.91, 0.88), fontsize=18, frameon=False)
    if data_rates==True:
        if channel_string=='all':
            ax.set_title('all channels, $z_\mathrm{merger}$ = %s, TNG simulation'%z_merger, fontsize = 30)
        else:
            ax.set_title('%s channel, $z_\mathrm{merger}$ = %s, TNG simulation'%(channel_string, z_merger), fontsize = 30)
        fig.savefig('figures/massdist_TNG%s_%s_%s_Z_z_data.pdf'%(tng, channel_string, z_merger), format="pdf", bbox_inches='tight', dpi=300)
    else:
        if channel_string=='all':
            ax.set_title('all channels, $z_\mathrm{merger}$ = %s, analytical fit'%z_merger, fontsize = 30)
        else:
            ax.set_title('%s channel, $z_\mathrm{merger}$ = %s, analytical fit'%(channel_string, z_merger), fontsize = 30)
        fig.savefig('figures/massdist_TNG%s_%s_%s_Z_z.pdf'%(tng, channel_string, z_merger), format="pdf", bbox_inches='tight', dpi=300)

    if showplot==True:
        plt.show()


def plot_BBH_rate_Z_z(tngpath, COMPASpath, tng, z_merger=0.2, z_formation=False, xlim=[], ylim=[], levels=[], nlevels=15, 
                      max_redshift=14, redshift_step=0.05, z_first_SF=14, plotredshift=True, showplot=True):
    
    levels = np.logspace(-2, 3, nlevels)
    cmap = sns.color_palette('rocket', as_cmap=True)
    
    fig = plt.figure(layout='constrained',figsize=[12,8])
    ax = fig.add_subplot()
    ax_histx = ax.inset_axes([0, 1.0, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.0, 0, 0.2, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    DCO = mfunc.read_data(loc = data_dir +str(COMPASfilename))
    ####################################################
    #Reading Rate data 
    with h5.File(data_dir + '/' + tngpath ,'r') as File:
        redshifts = File[list(File.keys())[0]]['redshifts'][()]
        DCO_mask = File[list(File.keys())[0]]['DCOmask'][()] # Mask from DCO to merging systems  
        merger_rate = File[list(File.keys())[0]]['merger_rate'][()]
        formation_rate = File[list(File.keys())[0]]['formation_rate'][()]
        seeds = File[list(File.keys())[0]]['SEED'][()]

    #first bring it to the same shape as the rate table
    merging_BBH    = DCO[DCO_mask]
    not_CHE  = merging_BBH['Stellar_Type@ZAMS(1)'] != 16
    BBH_bool = np.logical_and(merging_BBH['Stellar_Type(1)'] == 14, merging_BBH['Stellar_Type(2)'] == 14)
    merging_BBH         = merging_BBH[BBH_bool * not_CHE]
    merger_rate_dens  = merger_rate[BBH_bool, :]
    formation_rate_dens  = formation_rate[BBH_bool, :]
    merging_BBH_seeds =  seeds[BBH_bool]

    #read in delay times and time of merger and metallicity for all binaries
    with h5.File(data_dir + '/' + COMPASpath ,'r') as File:
            metallicities_compas = File['BSE_Double_Compact_Objects']['Metallicity@ZAMS(1)'][()] #same metallicity for both stars
            COMPAS_delay_times = File['BSE_Double_Compact_Objects']['Coalescence_Time'][()]#Myr
            time = File['BSE_Double_Compact_Objects']['Time'][()] #Myr
            seeds_compas = File['BSE_Double_Compact_Objects']['SEED'][()]

    #filter formation redshift and metallicity by the seeds we want
    mask_merging_DCOs = np.isin(seeds_compas, merging_BBH_seeds)
    seeds_DCOs = seeds_compas[mask_merging_DCOs]
    metallicities_compas = metallicities_compas[mask_merging_DCOs]
    log_metals = np.log10(metallicities_compas)
    metal_bins = np.logspace(min(log_metals), max(log_metals), 60)
    center_metalbins  = (metal_bins[:-1] + metal_bins[1:])/2.

    if plotredshift == True:
        xvals = redshifts
        xlim_min = z_merger
    else:
        xvals = [cosmo.lookback_time(z).value for z in redshifts]
        xlim_min = cosmo.lookback_time(z_merger).value

    if z_formation==True:
        rates = formation_rate_dens
    else:
        rates = merger_rate_dens

    #calculate formation redshift
    redshifts_new = np.arange(0, max_redshift + redshift_step, redshift_step)
    COMPAS_delay_times = COMPAS_delay_times[mask_merging_DCOs]
    times = cosmo.age(redshifts_new).to(u.Myr).value
    times_to_redshifts = interp1d(times, redshifts_new)
    age_first_sfr = cosmo.age(z_first_SF).to(u.Myr).value

    age_merger = cosmo.age(z_merger).to(u.Myr).value
    age_of_formation = age_merger - COMPAS_delay_times
    age_of_formation_mask = (age_of_formation >= age_first_sfr) #remove binaries that cannot merge at z=0.2 (too long delay time)
    age_of_formation_masked = age_of_formation[age_of_formation_mask]
    z_of_formation = times_to_redshifts(age_of_formation_masked)

    #remove all binaries that cannot merge at z=0.2
    seeds_DCOs = seeds_DCOs[age_of_formation_mask]
    metallicities_compas = metallicities_compas[age_of_formation_mask]
    merging_BBH_seeds = merging_BBH_seeds[age_of_formation_mask]
    merging_BBH = merging_BBH[age_of_formation_mask]
    rates = rates[age_of_formation_mask]

    Zzbinned_rates = []
    for i, metalbin in enumerate(metal_bins):
        if i < len(metal_bins)-1:
            #filter by formation redshift bin
            Z_mask_low = (metallicities_compas >= metalbin)
            Z_mask_high = (metallicities_compas < metal_bins[i+1])
            Z_mask = ((Z_mask_low==True) & (Z_mask_high==True))

            #get binaries that are in this metallicity bin
            Z_seeds = seeds_DCOs[Z_mask]
            Z_seeds_mask = np.isin(merging_BBH_seeds, Z_seeds)

            z_of_formation_masked = z_of_formation[Z_seeds_mask]

            redshift_bins = np.linspace(min(redshifts_new), max(redshifts_new), len(redshifts_new))

            #bin binaries by those redshifts
            hist, bin_edges = np.histogram(z_of_formation_masked, bins=redshift_bins, weights=np.sum(rates[Z_seeds_mask], axis=1))
            Zzbinned_rates.append(hist)
            
    Zzbinned_rates = np.array(Zzbinned_rates)

    plt.hist(cosmo.lookback_time(z_of_formation).to(u.Gyr).value)
    plt.show()

    #rate, xedges, yedges, image = plt.hist2d(z_of_formation, metallicities_compas/Zsun, bins=(redshift_bins, metal_bins/Zsun), cmap=cmap)

    #Zzbinned_rates[Zzbinned_rates <= 0] = 1e-4
    #rate = plt.contourf(xvals, center_metalbins/Zsun, Zzbinned_rates, levels=levels, cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1e3), extend='min')
    #ax_histx.plot(xvals, np.sum(Zzbinned_rates, axis=0), color='darkorange', lw=3)
    #ax_histy.plot(np.sum(Zzbinned_rates, axis=1), center_metalbins/Zsun, color='darkorange', lw=3)

    #########################################
    # plot values
    ax.set_yscale('log')
    if plotredshift == True:
        ax.set_xlabel('$z_\mathrm{formation}$', y=0.04, fontsize=30)
    else:
        ax.set_xlabel('Lookbacktime [Gyr]', y=0.04, fontsize=30)
    ax.set_ylabel(r'$Z/Z_{\rm{\odot}}$', x=0.03, fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')

    #Set limits for horizontal (lookback time or redshift) and vertical (metallicity) axes
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any
    else:
        ax.set_xlim(xlim_min, max(xvals))
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1]) 

    #sideplot formatting
    ax_histx.tick_params(axis='y', which='major', labelsize=15)
    ax_histx.tick_params(axis='x', which='both', direction='in')
    #ax_histx.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax_histx.tick_params(length=10, width=2, which='major')
    ax_histx.tick_params(length=5, width=1, which='minor')
    #ax_histx.set_ylim(0, 400)
    ax_histx.set_ylabel(r'$\mathcal{R}(z)$', fontsize=20)
    #histxlabels = ax_histx.get_yticklabels()
    #histxlabels[0] = ''
    #ax_histx.set_yticklabels(histxlabels)
    ax_histx.set_yscale('log')

    ax_histy.tick_params(axis='x', which='major', labelsize=15)
    ax_histy.tick_params(axis='y', which='both', direction='in')
    ax_histy.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax_histy.tick_params(length=10, width=2, which='major')
    ax_histy.tick_params(length=5, width=1, which='minor')
    #ax_histy.set_xlim(0, 1000)
    ax_histy.set_xlabel(r'$\mathcal{R}(Z_{\rm{i}})$', fontsize=20)

    #Set up the colorbar
    #fig.subplots_adjust(right=0.81)
    #cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar_ax = fig.add_axes([1.01, 0.1, 0.03, 0.7])
    cbar = fig.colorbar(rate, cax=cbar_ax, ticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    cbar.ax.tick_params(labelsize=20)
    if z_formation==True:
        cbar.set_label(r'Formation rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', rotation=270, fontsize=30, labelpad=40);
    else:
        cbar.set_label(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', rotation=270, fontsize=30, labelpad=40);
    
    # Channel
    if tng==50:
        plt.text(0.02, 0.02, 'TNG%s'%labels[0], color='white', transform=ax.transAxes, size = 35)
    elif tng==100:
        plt.text(0.02, 0.02, 'TNG%s'%labels[1], color='white', transform=ax.transAxes, size = 35)
    elif tng==300:
        plt.text(0.02, 0.02, 'TNG%s'%labels[2], color='white',  transform=ax.transAxes, size = 35)
    
    #fig.legend(bbox_to_anchor=(0.9, 0.88), fontsize=18, frameon=False)
    if z_formation==True:
        if plotredshift==True:
            fig.savefig('figures/BBHrate_redshift_TNG%s_Z_zform.png'%(tng), bbox_inches='tight', dpi=300)
        else:
            fig.savefig('figures/BBHrates_lookbackt_TNG%s_Z_zform.png'%(tng), bbox_inches='tight', dpi=300)
    else:
        if plotredshift==True:
            fig.savefig('figures/BBHrates_redshift_TNG%s_Z_z_data.png'%(tng),bbox_inches='tight', dpi=300)
        else:
            fig.savefig('figures/BBHrates_lookbackt_TNG%s_Z_z.png'%(tng),bbox_inches='tight', dpi=300)
    if showplot==True:
        plt.show()

        

if __name__ == "__main__":

    #TNG data setup (will eventually turn into arguments but currently just uncomment the ones that you want)
    In.init()
    data_dir    =  str(paths.data) +'/'
    save_loc = str(paths.figures) + '/'
    COMPASfilename = 'COMPAS_Output_wWeights.h5'
    Zsun = 0.014 #Solar metallicity

    filenames = ['SFRMetallicityFromGasWithMetalsTNG50-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG100-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG300-1.hdf5',
                 'SFRMetallicityFromGasTNG100-2.hdf5', 'SFRMetallicityFromGasTNG50-2.hdf5', 'SFRMetallicityFromGasTNG50-3.hdf5'] 
    fit_param_files = ['test_best_fit_parameters_TNG50-1_TEST.txt', 'test_best_fit_parameters_TNG100-1_TEST.txt', 'test_best_fit_parameters_TNG300-1_TEST.txt']
    fit_param_files_data = ['test_best_fit_parameters_TNG50-1.txt', 'test_best_fit_parameters_TNG100-1.txt', 'test_best_fit_parameters_TNG300-1.txt']
    rates = ['TEST_Rate_info_TNG50-1.h5', 'TEST_Rate_info_TNG100-1.h5', 'TEST_Rate_info_TNG300-1.h5']
    model_rates = ['TEST_Rate_info_TNG50-1.h5', 'TEST_Rate_info_TNG100-1.h5', 'TEST_Rate_info_TNG300-1.h5']
    data_rates = ['data_Rate_info_TNG50-1.h5', 'data_Rate_info_TNG100-1.h5', 'data_Rate_info_TNG300-1.h5']

    #Plot setup
    labels = ['50-1', '100-1', '300-1']
    linestyles = ['-', '-', '-', '-', '-', '-']
    linestyles2 = ['-', '--', ':', '-.']
    lineweights = [4, 4, 4, 4, 4, 4]

    line_styles = ['solid', 'dashed', 'dotted']
    tmin = 0.0
    tmax = 13.7
    z_of_mass_dist = [0.2, 1, 2, 4, 6, 8]
    cmap_blue = matplotlib.colors.LinearSegmentedColormap.from_list("blue_cmap", ['#40E9E0', '#1C7EB7', '#100045'])
    cmap_pink = matplotlib.colors.LinearSegmentedColormap.from_list("pink_cmap", ['#F76FDD', '#C13277', '#490013'])
    cmap_green = matplotlib.colors.LinearSegmentedColormap.from_list("green_cmap", ['#CCE666', '#79B41C', '#004011'])
    #cmap_gray = matplotlib.colors.LinearSegmentedColormap.from_list("gray_cmap", ['#D3D2D2', "#0F0F0F"])
    cmap_gray = matplotlib.colors.LinearSegmentedColormap.from_list("gray_cmap", ["#C9C8D7", "#151618"])
    data_colors = ["#0067A6", '#C01874', '#98CB4F', '#D3D2D2']
    model_colors = ['#0C0034', '#4B0012', '#005B2F', '#787878']
    #colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    #colors = ["#0067A6", '#79B41C', "#ECA120", "#ED6544",  '#C01874']
    colormap = sns.color_palette('viridis_r', as_cmap=True)
    

    #Read in SFRD model parameters for each TNG
    fit_param_vals = read_best_fits(fit_param_files)

    #Compare model and data merger and formation rates
    #compare_BBH_data_and_model_rates(data_dir, model_rates, data_rates, error_ylim=[1e-2, 1e4], plot_merger_rates=True, plot_logscale=True, showplot=True, showLVK=True)

    #residuals_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, ylim = [1e-1, 1e5], plot_merger_rates=False, showplot=True)
    #residuals_BBH_data_and_model_rates(data_dir, model_rates, data_rates, fit_param_vals, ylim = [1e-1, 1e5], plot_merger_rates=True, showplot=True)

    #plot_BBH_mass_dist_over_z_allTNGs(model_rates, fit_param_vals, tngs=[50, 100, 300], z = [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2], showplot=False)
    #compare_BBH_data_and_model_mass_dist_over_z(model_rates, data_rates, only_stable = True, only_CE = True, channel_string='all', z = [0.2, 1, 2, 4, 6, 8], showplot=False)
    
    #compare_BBH_data_and_model_mass_dist(model_rates, data_rates, fit_param_vals, only_stable = True, only_CE = True, channel_string='all', z = 0.2, showplot=True)

    residuals_BBH_data_and_model_mass_dist(model_rates, data_rates, only_stable = True, only_CE = True, channel_string='all', z = [0.2, 1, 2, 4, 6, 8], showplot=True)
    
    #plot_BBH_mass_dist_formation_channels(data_rates[1], model_rates[1], 100, 1, z = [8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2], showplot=True)

    #plot_BBH_mass_Z_z(data_rates[1], COMPASfilename, tng=100, z_form = [0.2, 0.5, 1, 2, 6, 10], Z_zams = [0.1, 0.01, 0.001, 0.0001], 
    #              data_rates=True, only_stable = True, only_CE = True, channel_string='all',  z_merger=0.2, showplot=True)
    """
    plot_BBH_mass_Z_z(model_rates[0], COMPASfilename, [fit_param_vals[0]], tng=50, data_rates=False, only_stable = True, only_CE = False, channel_string='stable',  
                      z_merger=0.2, z_form = [4, 8, 12, 14], Z_zams = [0.0001, 0.001, 0.01, 0.1], showplot=True)
    plot_BBH_mass_Z_z(model_rates[0], COMPASfilename, [fit_param_vals[0]], tng=50, data_rates=False, only_stable = False, only_CE = True, channel_string='CE',  
                      z_merger=0.2, z_form = [4, 8, 12, 14], Z_zams = [0.0001, 0.001, 0.01, 0.1], showplot=True)
    plot_BBH_mass_Z_z(data_rates[0], COMPASfilename, [fit_param_vals[0]], tng=50, data_rates=True, only_stable = True, only_CE = True, channel_string='all',  
                      z_merger=0.2, z_form = [4, 8, 12, 14], Z_zams = [0.0001, 0.001, 0.01, 0.1], showplot=True)
    plot_BBH_mass_Z_z(data_rates[0], COMPASfilename, [fit_param_vals[0]], tng=50, data_rates=True, only_stable = True, only_CE = False, channel_string='stable',  
                      z_merger=0.2, z_form = [4, 8, 12, 14], Z_zams = [0.0001, 0.001, 0.01, 0.1], showplot=True)
    plot_BBH_mass_Z_z(data_rates[0], COMPASfilename, [fit_param_vals[0]], tng=50, data_rates=True, only_stable = False, only_CE = True, channel_string='CE',  
                      z_merger=0.2, z_form = [4, 8, 12, 14], Z_zams = [0.0001, 0.001, 0.01, 0.1], showplot=True)
    """

   #plot_BBH_rate_Z_z(data_rates[0], COMPASfilename, tng=50, xlim=[], ylim=[], nlevels=30, z_formation=False, plotredshift=True, showplot=True)