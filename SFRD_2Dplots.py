import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import matplotlib.ticker as ticker
import matplotlib

from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses

############################
# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
import paths
import Fit_model_TNG_SFRD as fitmodel
from TNG_BBHpop_properties import read_best_fits
from Fit_model_TNG_SFRD import readTNGdata


def SFRDplot_2D(metals, Lookbacktimes, SFRD, tngs=[], ver=[], model=[], xlim=[], ylim=[10**-1, 10**1], levels = [], plottype='data', nlevels=20, cmap='rocket', showplot=True):

    """
    plottype (str): type of SFRD 2D plot, options: 'data', 'percenterr'
    """

    if plottype == 'data':
        if len(levels) > 0:
            levels = np.linspace(levels[0], levels[1], nlevels)
        else:
            levels = np.linspace(np.amin(SFRD), np.amax(SFRD), nlevels)
        cmap = sns.color_palette('rocket', as_cmap=True)
    if plottype == 'percenterr':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['white', 'tab:blue'])

    if len(tngs) == 1:
        w = 14
        h = 10
    elif len(tngs) == 2:
        w = 18
        h = 10
    elif len(tngs) > 4:
        w = 14
        h = 16
    else:
        w = 14
        h = 12
    fig, ax = plt.subplots(figsize = (w,h))
    fig.subplots_adjust(wspace=0, hspace=0)

    for n, tng in enumerate(tngs):
        
        #Set up configuration of subplots depending on how many plotting
        if len(tngs) == 1: 
            ax = plt.subplot(1, 1, n + 1)
        elif len(tngs) > 1 and len(tngs) <= 2:
            ax = plt.subplot(1, len(tngs), n + 1)
        elif len(tngs) > 2:
            ax = plt.subplot(ceil(len(tngs)/2), 2, n + 1)

        #Plot data and is model is given plot model
        if plottype=='data':
            data = ax.contourf(Lookbacktimes[n], metals[n]/Zsun, SFRD[n], levels=levels, cmap=cmap)
            ax.text(0.02, 0.02, "TNG%s-%s, model"%(tng, ver[n]), transform=ax.transAxes, fontsize=15, color='white')

        if plottype=='percenterr':
            SFRD[n][SFRD[n] < 1e-7] = 0
            model[n].T[model[n].T < 1e-7] = 0
            percenterr = abs(model[n].T-SFRD[n])/SFRD[n]
            levels = np.linspace(0.01, 0.4, nlevels)
            data = ax.contourf(Lookbacktimes[n], metals[n]/Zsun, percenterr, levels=levels, cmap=cmap)
            ax.text(0.02, 0.02, "TNG%s-%s, model"%(tng, ver[n]), transform=ax.transAxes, fontsize=15, color='black')

        if len(model[n]) > 0:
            #Model contours
            if tng==50:
                clevels = [1e-3, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05] 
            elif tng==100:
                clevels = [1e-3, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.033] 
            elif tng==300:
                clevels = [1e-3, 0.005, 0.01, 0.015,0.02, 0.024] 
            if plottype=='data':
                modelplot = ax.contour(Lookbacktimes[n], metals[n]/Zsun, model[n].T, levels=clevels, colors='white')
                ax.clabel(modelplot, fontsize=12, inline=True)
            elif plottype=='percenterr':
                modelplot = ax.contour(Lookbacktimes[n], metals[n]/Zsun, model[n].T, levels=clevels, colors='black', linestyles='dashed')

        #Set yscale, TNG version label on each plot, and axis labels
        ax.set_yscale('log')
        fig.supxlabel('Lookback time', y=0.05, fontsize=20)
        fig.supylabel(r'$\mathrm{Metallicity}, \ Z/Z_{\rm{\odot}}$', x=0.07, fontsize=20)

        #Set redshift ticks; make sure they don't overlap
        ax2 = ax.twiny()
        if n==1:
            redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
        elif len(tngs) == 1:
            redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
        else:
            redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6]
        
        #Hide overlapping tick labels
        if n>0:
            nbins = len(ax.get_yticklabels())
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins, prune='upper'))

        #Hide tick labels depending on how many subplots are in the figure, and their configuration
        if n%2==0:
            if len(tngs) > 2 and n<len(tngs)-2: #if left column with more than two subplots, but not last row, hide ticks and labels
                ax.tick_params(axis='x',  which='both', bottom=False, labelbottom=False)
        else:
            if len(tngs) > 1 and len(tngs) <= 2: #if right subplot with only one row, hide left ticks on the left side
                ax.tick_params(axis='y',  which='both', left=False, labelleft=False)
            elif len(tngs) > 2 and n<len(tngs)-2: #if right subplot with more than one row, hide left and bottom ticks
                ax.tick_params(axis='both',  which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
            else: #if last right subplot, hide left label
                ax.tick_params(axis='both',  which='both', left=False, labelleft=False)

        #Set redshift axis labels for only first row of subplots
        if n < 2: 
            ax2.set_xlabel('Redshift', fontsize = 20)
            ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list])
            ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list])
        else:
            ax2.set_xlabel('', fontsize = 20)
            ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list*0])
            ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list*0])

        #Set limits for horizontal (lookback time) and vertical (metallicity) axes
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Lookbacktimes[n]), max(Lookbacktimes[n]))
            ax2.set_xlim(min(Lookbacktimes[n]), max(Lookbacktimes[n]))

        if len(ylim) > 0:
            ax.set_ylim(ylim[0], ylim[1]) #defaults for TNG data are Z=10**-1 to Z=10**1

    
    #Set up the colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(data, cax=cbar_ax, format=ticker.FormatStrFormatter('%.2g'))

    #Set the colorbar labels and save the plot
    if plottype=='data':
        if len(tngs)==1:
            if ver[0]>1:
                cbar.set_label('SFRD(Z, z) TNG%s-%s data'%(tngs[0], ver[0]), rotation=270, fontsize=20, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s.png'%(tngs[0], ver[0]), bbox_inches='tight')
            else:
                cbar.set_label('SFRD(Z, z) TNG%s data'%tngs[0], rotation=270, fontsize=20, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s.png'%tngs[0], bbox_inches='tight')
        else:
            cbar.set_label('SFRD(Z, z), TNG data', rotation=270, fontsize=20, labelpad=30);
            fig.savefig('figures/SFRD_Z_z_TNG.png', bbox_inches='tight')
    elif plottype=='percenterr':
        if len(tngs)==1:
            if ver[0]>1:
                cbar.set_label('SFRD(Z, z) percent error, TNG%s-%s data'%(tngs[0], ver[0]), rotation=270, fontsize=20, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%sdiff.png'%(tngs[0], ver[0]), bbox_inches='tight')
            else:
                cbar.set_label('SFRD(Z, z) percent error, TNG%s data'%tngs[0], rotation=270, fontsize=20, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%sdiff.png'%tngs[0], bbox_inches='tight')
        else:
            cbar.set_label('SFRD(Z, z) percent error', rotation=270, fontsize=20, labelpad=30);
            fig.savefig('figures/SFRD_Z_z_TNGdiff.png', bbox_inches='tight')
    
    if showplot==True:
        plt.show()


def SFRDmodels_2D(metals, Lookbacktimes, SFRD, tngs=[], ver=[], model=[], xlim=[], ylim=[10**-1, 10**1], levels = [], showplot=True):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    #Model contours
    if tng==50:
        clevels = [1e-3, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05] 
    elif tng==100:
        clevels = [1e-3, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.033] 
    elif tng==300:
        clevels = [1e-3, 0.005, 0.01, 0.015,0.02, 0.024] 
    
    for n, tng in enumerate(tngs):
        modelplot = ax.contour(Lookbacktimes[n], metals[n]/Zsun, model[n].T, levels=clevels, colors='white')
        ax.clabel(modelplot, fontsize=12, inline=True)
    
        #Set yscale, TNG version label on each plot, and axis labels
        ax.set_yscale('log')
        fig.supxlabel('Lookback time', y=0.05, fontsize=20)
        fig.supylabel(r'$\mathrm{Metallicity}, \ Z/Z_{\rm{\odot}}$', x=0.07, fontsize=20)

    #Set redshift ticks; make sure they don't overlap
    ax2 = ax.twiny()
    if n==1:
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
    elif len(tngs) == 1:
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
    else:
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6]


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    tngs=[50, 100, 300, 100] 
    vers = [1, 1, 1, 2]
    Zsun = 0.014 #Solar metallicity

    #Read the TNG data and interpolate it
    SFRDsTNG= []
    redshiftsTNG = []
    LookbacktimesTNG = []
    metalsTNG = []
    param_filenames = []

    for n, tng in enumerate(tngs):
        Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5"%(tng,vers[n]))
        fit_filename = 'test_best_fit_parameters_TNG%s-%s.txt'%(tng,vers[n])
        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205
        Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ = readTNGdata(loc = Cosmol_sim_location, rbox=rbox)
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new = fitmodel.interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], saveplot=False)
        SFRDsTNG.append(SFRDnew)
        redshiftsTNG.append(redshift_new)
        LookbacktimesTNG.append(Lookbacktimes_new)
        metalsTNG.append(metals_new)
        param_filenames.append(fit_filename)

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

    SFRDplot_2D(metalsTNG, LookbacktimesTNG, SFRDsTNG, tngs, vers, model=models, plottype='data')
