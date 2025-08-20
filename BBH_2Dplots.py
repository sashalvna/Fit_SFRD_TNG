import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import matplotlib.ticker as ticker
import matplotlib
import h5py as h5

from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses

############################
# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
import paths
import Fit_model_TNG_SFRD as fitmodel
from TNG_BBHpop_properties import read_best_fits
from Fit_model_TNG_SFRD import readTNGdata

## PLOT setttings
plt.rc('font', family='serif')
from matplotlib import rc
import matplotlib
plt.rc('font', family='serif', weight='bold')
plt.rc('text', usetex=True)
matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams.update({'font.weight': 'bold'})


def BBH_2Dplot_sidepanels(data_dir, rates_file, tng=[], ver=[], model=None, xlim=[], ylim=[10**-2, 10**2], levels = [], 
                          nlevels=20, plottype='data', plotredshift=True, plotregions=False, showplot=True):

    """
    plottype (str): type of SFRD 2D plot, options: 'data', 'percenterr'
    """

    #Set up the plot
    fig = plt.figure(layout='constrained',figsize=[14,10])
    ax = fig.add_subplot()
    ax_histx = ax.inset_axes([0, 1.02, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    #if plotredshift == True:
    #    xvals = Redshifts
    #else:
    #    xvals = Lookbacktimes


    #Read in rates data
    with h5.File(data_dir + rates_file ,'r') as File:
            redshift      = File[list(File.keys())[0]]['redshifts'][()]
            merger_rate    = File[list(File.keys())[0]]['merger_rate'][()]
            seed      = File[list(File.keys())[0]]['SEED'][()]

    print(seed.shape, redshift.shape, merger_rate.shape)

    """
    if plottype == 'data':
        if len(levels) > 0:
            levels = np.linspace(levels[0], levels[1], nlevels)
        else:
            levels = np.linspace(np.amin(SFRD), np.amax(SFRD), nlevels)
        cmap = sns.color_palette('rocket', as_cmap=True)
    if plottype == 'percenterr':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['tab:red', 'white', 'tab:blue'])
    """

    """
    #Plot data and is model is given plot model
    if plottype=='data':

        data = ax.contourf(xvals, metals/Zsun, SFRD, levels=levels, cmap=cmap)
        if len(model) > 0:
            ax.text(0.02, 0.02, "TNG%s-%s, analytical fit"%(tng, ver), transform=ax.transAxes, fontsize=20, color='white')

        ax_histx.plot(xvals, np.sum(SFRD, axis=0))
        ax_histy.plot(np.sum(SFRD, axis=1), metals/Zsun)
        ax_histx.set_ylim(0, 6)
        ax_histy.set_xlim(0, 4)

    if plottype=='percenterr':
        SFRD[SFRD < 1e-7] = 1e-7
        model.T[model.T < 1e-7] = 1e-7
        percenterr = (SFRD-model.T)/model.T
        levels = np.linspace(-2, 2, nlevels)
        data = ax.contourf(xvals, metals/Zsun, percenterr, levels=levels, cmap=cmap)
        ax.text(0.02, 0.02, "TNG%s-%s, analytical fit"%(tng, ver), transform=ax.transAxes, fontsize=20, color='black')

        ax_histx.plot(xvals, np.zeros(len(xvals)), color='lightgray')
        ax_histy.plot(np.zeros(len(np.sum(percenterr, axis=1))), np.sum(percenterr, axis=1), color='lightgray')
        ax_histx.plot(xvals, np.sum(percenterr, axis=0))
        ax_histy.plot(np.sum(percenterr, axis=1), metals/Zsun)

        ax_histx.set_ylim(-4000, 4000)
        ax_histy.set_xlim(-500, 2000)
    
    if len(model) > 0:
        #Model contours
        if tng==50:
            clevels = [1e-5, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05] 
        elif tng==100:
            clevels = [1e-5, 1e-3, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.033] 
        elif tng==300:
            clevels = [1e-5, 1e-3, 0.005, 0.01, 0.015,0.02, 0.024] 
        if plottype=='data':
            modelplot = ax.contour(xvals, metals/Zsun, model.T, levels=clevels, colors='white')
            #ax.clabel(modelplot, fontsize=11, inline=True)
        elif plottype=='percenterr':
            modelplot = ax.contour(xvals, metals/Zsun, model.T, levels=clevels, colors='black', linestyles='dashed')

    if plotregions==True:
        lowz_bin = [4, 8]
        midz_bin = [10, 12]
        highz_bin = [12, 14]
        zbins = np.array([lowz_bin, midz_bin, highz_bin])

        lowZ_bin = [0.0001, 0.001]
        midZ_bin = [0.001, 0.01]
        highZ_bin = [0.01, 0.1]
        Zbins = np.array([lowZ_bin, midZ_bin, highZ_bin])
        Zbins_Zsun = Zbins/Zsun

        Zzbin_colors = ['darkgreen', 'mediumseagreen', 'lightgreen',
                        'darkorange', 'orange', 'gold',
                        'purple', 'mediumvioletred', 'pink']
        counter = 0

        for zbin in zbins:
            for Zbin in Zbins_Zsun:
                rect = matplotlib.patches.Rectangle((zbin[0], Zbin[0]), zbin[1]-zbin[0], Zbin[1]-Zbin[0], linewidth=2, edgecolor=Zzbin_colors[counter], facecolor='none')
                ax.add_patch(rect)
                counter+=1

    ax.set_yscale('log')
    if plotredshift == True:
        ax.set_xlabel('Redshift', y=0.04, fontsize=30)
    else:
        ax.set_xlabel('Lookbacktime (Gyr)', y=0.04, fontsize=30)
    ax.set_ylabel(r'$Z/Z_{\rm{\odot}}$', x=0.03, fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(length=10, width=2, which='major')
    ax.tick_params(length=5, width=1, which='minor')

    #Set limits for horizontal (lookback time or redshift) and vertical (metallicity) axes
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any
    else:
        ax.set_xlim(min(xvals), max(xvals))
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1]) #defaults for TNG data are Z=10**-2 to Z=10**1

    #Set up the colorbar
    fig.subplots_adjust(right=0.81)
    cbar_ax = fig.add_axes([1.01, 0.1, 0.03, 0.7])
    cbar = fig.colorbar(data, cax=cbar_ax, format=ticker.FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=20)

    #Set the colorbar labels and save the plot
    if plottype=='data':
        if plotredshift==False:
            if plotregions==False:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s-%s data'%(tng, ver), rotation=270, fontsize=30, labelpad=35);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s_sidepanels.png'%(tng, ver), bbox_inches='tight', dpi=300)
            else:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s-%s data'%(tng, ver), rotation=270, fontsize=30, labelpad=35);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s_sidepanels_regions.png'%(tng, ver), bbox_inches='tight', dpi=300)
        else:
            if plotregions==False:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s data'%tng, rotation=270, fontsize=30, labelpad=35);
                fig.savefig('figures/SFRD_Z_z_TNG%s_redshift_sidepanels.png'%tng, bbox_inches='tight', dpi=300)
            else:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s data'%tng, rotation=270, fontsize=30, labelpad=35);
                fig.savefig('figures/SFRD_Z_z_TNG%s_redshift_sidepanels_regions.png'%tng, bbox_inches='tight', dpi=300)
    elif plottype=='percenterr':
        if plotredshift==False:
            if plotregions==False:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ $\frac{\mathcal{S}(Z_{\rm{i}},z)-1}{x}$, TNG%s-%s'%(tng, ver), rotation=270, fontsize=30, labelpad=45);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s_percenterr_sidepanels.png'%(tng, ver), bbox_inches='tight')
            else:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ $\frac{\mathcal{S}(Z_{\rm{i}},z)-1}{x}$, TNG%s-%s'%(tng, ver), rotation=270, fontsize=30, labelpad=45);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s_percenterr_sidepanels_regions.png'%(tng, ver), bbox_inches='tight')
        else:
            if plotregions==False:
                cbar.set_label(r'$\frac{\mathcal{S}(Z_{\rm{i}},z)_\mathrm{sim} - \mathcal{S}(Z_{\rm{i}},z)_\mathrm{model}}{\mathcal{S}(Z_{\rm{i}},z)_\mathrm{model}}$, TNG%s'%tng, rotation=270, fontsize=30, labelpad=45);
                fig.savefig('figures/SFRD_Z_z_TNG%s_percenterr_redshift_sidepanels.png'%tng, bbox_inches='tight', dpi=300)
            else:
                cbar.set_label(r'$\frac{\mathcal{S}(Z_{\rm{i}},z)_\mathrm{sim} - \mathcal{S}(Z_{\rm{i}},z)_\mathrm{model}}{\mathcal{S}(Z_{\rm{i}},z)_\mathrm{model}}$, TNG%s'%tng, rotation=270, fontsize=30, labelpad=45);
                fig.savefig('figures/SFRD_Z_z_TNG%s_percenterr_redshift_sidepanels_regions.png'%tng, bbox_inches='tight', dpi=300)
    
    if showplot==True:
        plt.show()
    """


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    tngs=[50, 100, 300, 100] 
    vers = [1, 1, 1, 2]
    Zsun = 0.014 #Solar metallicity

    #TNG data setup (will eventually turn into arguments but currently just uncomment the ones that you want)
    data_dir    =  str(paths.data) +'/'
    save_loc = str(paths.figures) + '/'
    COMPASfilename = 'COMPAS_Output_wWeights.h5'

    filenames = ['SFRMetallicityFromGasWithMetalsTNG50-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG100-1.hdf5', 'SFRMetallicityFromGasWithMetalsTNG300-1.hdf5'] 
    fit_param_files = ['test_best_fit_parameters_TNG50-1_TEST.txt', 'test_best_fit_parameters_TNG100-1_TEST.txt', 'test_best_fit_parameters_TNG300-1_TEST.txt']
    #fit_param_files_data = ['test_best_fit_parameters_TNG50-1.txt', 'test_best_fit_parameters_TNG100-1.txt', 'test_best_fit_parameters_TNG300-1.txt']
    model_rates = ['TEST_Rate_info_TNG50-1.h5', 'TEST_Rate_info_TNG100-1.h5', 'TEST_Rate_info_TNG300-1.h5']
    data_rates = ['data_Rate_info_TNG50-1.h5', 'data_Rate_info_TNG100-1.h5', 'data_Rate_info_TNG300-1.h5']

    #Read in SFRD model parameters for each TNG
    fit_param_vals = read_best_fits(fit_param_files)

    """
    #Read the TNG data and interpolate it
    SFRDsTNG= []
    redshiftsTNG = []
    LookbacktimesTNG = []
    metalsTNG = []
    param_filenames = []

    for n, tng in enumerate(tngs):
        Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5"%(tng,vers[n]))
        if vers[n] == 2:
            Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasTNG%s-%s.hdf5"%(tng,vers[n]))
        fit_filename = 'test_best_fit_parameters_TNG%s-%s_TEST.txt'%(tng,vers[n])
        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205
        Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ = readTNGdata(loc = Cosmol_sim_location, rbox=rbox, metals=False)
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = fitmodel.interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], redshiftlimandstep=[0, 20, 0.05], saveplot=False)
        SFRDsTNG.append(SFRDnew)
        redshiftsTNG.append(redshift_new)
        LookbacktimesTNG.append(Lookbacktimes_new)
        metalsTNG.append(metals_new)
        param_filenames.append(fit_filename)

        print(max(np.sum(Sim_SFRD, axis=0)), max(np.sum(Sim_SFRD, axis=1)))
        print(max(np.sum(SFRDnew, axis=0)), max(np.sum(SFRDnew, axis=1)))

    #Read in best fit SFRD parameters
    bestfits = read_best_fits(param_filenames)

    #Calculate the model
    models = []
    for n, fit_params in enumerate(bestfits):
        sfr = Z_SFRD.Madau_Dickinson2014(redshiftsTNG[n], a=fit_params[5], b=fit_params[6], c=fit_params[7], d=fit_params[8]).value # Msun year-1 Mpc-3 
        dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                    Z_SFRD.skew_metallicity_distribution(redshiftsTNG[n] , mu_0 = fit_params[0], mu_z = fit_params[1],
                                                  omega_0= fit_params[2] , omega_z=fit_params[3] , alpha = fit_params[4], 
                                                  metals=metalsTNG[n])
        models.append(sfr[:,np.newaxis] * dPdlogZ)

    """

    #SFRDplot_2D(metalsTNG, LookbacktimesTNG, SFRDsTNG, tngs, vers, ylim=[10**-4, 50], nlevels=17, model=models, plottype='percenterr', plotregions=True)
    BBH_2Dplot_sidepanels(data_dir, model_rates[0], tngs[0], vers[0], xlim=[14, 0], ylim=[10**-4, 50], nlevels=17,  plottype='data', plotredshift=True, plotregions=True)

    
