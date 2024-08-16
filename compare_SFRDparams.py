import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses
import seaborn as sns
from math import ceil
import matplotlib.ticker as ticker
import matplotlib

# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
from TNG_BBHpop_properties import read_best_fits
from Fit_model_TNG_SFRD import readTNGdata, interpolate_TNGdata

def compare_params(tngs=[50, 100, 300], vers=[1, 1, 1], showplot=True):
    
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)

    ticks =  ["mean metallicity at z=0", "redshift evolution of mean", "variance in metallicity density distribution", 
              "redshift evolution of variance", "skewness","SFR scaling", "SFR upward slope", "SFR peak location", "SFR downward slope"]
    x = [0,1,2,3,4,5,6,7,8] #9 parameters

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    for n, fits in enumerate(fit_params):
        ax[0].plot(x, fits, label="TNG%s-%s"%(tngs[n], vers[n]), lw=2)

        if n != 1:
            ax[1].plot(x, fit_params[n]/fit_params[1], label='TNG%s-%s/TNG%s-%s'%(tngs[n], vers[n], tngs[1], vers[1]), lw=2)
    
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ticks, rotation=90, fontsize=13)
    ax[0].legend(fontsize=13)
    ax[0].set_title('SFRD fit parameters', fontsize=18)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ticks, rotation=90, fontsize=13)
    ax[1].legend(fontsize=13)
    ax[1].set_title('Percent error in SFRD fit parameters', fontsize=18)

    fig.savefig('figures/fitparams_percenterror.png', bbox_inches='tight')
    
    if showplot==True:
        plt.show()


def compare_SFR(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], plotmodel=True, plotredshift=True, show_MD17=True, showplot=True):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)

    data_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    fit_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkred', 'darkpurple', 'darkbrown']

    fig, ax = plt.subplots(figsize=(12,8))

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        #Plot the TNG data
        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=True)

        if plotredshift == True:
            xvals = Sim_Redshifts
        else:
            xvals = Sim_Lookbacktimes
        plt.plot(xvals, np.sum(Sim_SFRD, axis=1), label="TNG%s-%s"%(tng, vers[n]), lw=5, c=data_colors[n], alpha=0.6)

        #Plot the TNG model
        if plotmodel==True:
            
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value
            plt.plot(xvals, sfr, lw=2, c=fit_colors[n], ls='--')

    if show_MD17 == True:
        #default Madau & Fragos 17
        ax.plot(xvals, Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=0.01, b=2.6, c=3.2, d=6.2), 
                 label = 'Madau & Fragos 2017', #+'\n'+'$a=%.2f, b=%.2f, c=%.2f, d=%.2f$'% (0.01,2.6,3.2,6.2), 
                 c = 'darkgrey', ls = ':',lw=2)
        
    x = [-0.0001]
    y1 = [0.0001]
    y2 = [0.0001]
    plt.plot(x, y1, c='black', ls = '-', lw=5, label='Data')
    plt.plot(x, y2, c='black', ls = '--', lw=2, label='Model')
    ax.set_ylabel("SFRD(z)", fontsize = 20)
    ax.legend(fontsize = 15)
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])

    if plotredshift:
        ax.set_xlabel("Redshift", fontsize = 20)
        if plotmodel:
            fig.savefig('figures/SFR_redshift_fit.png',  bbox_inches='tight')
        else:
            fig.savefig('figures/SFR_redshift.png',  bbox_inches='tight')
    else:
        ax.set_xlabel("Lookback time (Gyr)", fontsize = 20)
        if plotmodel:
            fig.savefig('figures/SFR_lookbackt_fit.png',  bbox_inches='tight')
        else:
            fig.savefig('figures/SFR_lookbackt.png',  bbox_inches='tight')
    
    if showplot==True:
        plt.show()


def compare_dPdlogZ(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], levels = [0, 0.5], nlevels=20, showplot=True):
    
    #Set up figure size based on number of subplots
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

    Zsun = 0.014

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)

    levels = np.linspace(levels[0], levels[1], nlevels)
    cmap = sns.color_palette('rocket', as_cmap=True)

    for n, tng in enumerate(tngs):
        
        #Set up configuration of subplots depending on how many plotting
        if len(tngs) == 1: 
            ax = plt.subplot(1, 1, n + 1)
        elif len(tngs) > 1 and len(tngs) <= 2:
            ax = plt.subplot(1, len(tngs), n + 1)
        elif len(tngs) > 2:
            ax = plt.subplot(ceil(len(tngs)/2), 2, n + 1)

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=True)
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new = interpolate_TNGdata(Sim_Redshifts, Sim_Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], saveplot=False)

        dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                Z_SFRD.skew_metallicity_distribution(redshift_new , mu_0 = fit_params[n][0], mu_z = fit_params[n][1],
                                                  omega_0= fit_params[n][2] , omega_z=fit_params[n][3] , alpha = fit_params[n][4], 
                                                  metals=metals_new)

        #Model plot
        modelplot = ax.contourf(Lookbacktimes_new, metals_new/Zsun, dPdlogZ.T, levels=levels, cmap=cmap)
        ax.text(0.02, 0.02, "TNG%s-%s"%(tng, vers[n]), transform=ax.transAxes, fontsize=15, color='white')
        
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
            ax.set_xlim(min(Lookbacktimes_new), max(Lookbacktimes_new))
            ax2.set_xlim(min(Lookbacktimes_new), max(Lookbacktimes_new))

        if len(ylim) > 0:
            ax.set_ylim(ylim[0], ylim[1]) #defaults for TNG data are Z=10**-1 to Z=10**1

    
    #Set up the colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(modelplot, cax=cbar_ax, format=ticker.FormatStrFormatter('%.2g'))

    #Set the colorbar labels and save the plot
    if len(tngs)==1:
        if vers[0]>1:
            cbar.set_label('dP/dlogZ TNG%s-%s model'%(tngs[0], vers[0]), rotation=270, fontsize=20, labelpad=30);
            fig.savefig('figures/dPdlogZ_TNG%s_%s.png'%(tngs[0], vers[0]), bbox_inches='tight')
        else:
            cbar.set_label('dP/dlogZ TNG%s model'%tngs[0], rotation=270, fontsize=20, labelpad=30);
            fig.savefig('figures/dPdlogZ_TNG%s.png'%tngs[0], bbox_inches='tight')
    else:
        cbar.set_label('dP/dlogZ, TNG model', rotation=270, fontsize=20, labelpad=30);
        fig.savefig('figures/dPdlogZ_TNG.png', bbox_inches='tight')
    
    if showplot==True:
        plt.show()


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    path = '/Users/sashalvna/Research/Fit_SFRD_TNG/data/'
    tngs=[50, 100, 300] 
    vers = [1, 1, 1]

    compare_params(tngs, vers)
    compare_SFR(path, tngs, vers, plotmodel=True, plotredshift=True, xlim=[0, 10])
    compare_dPdlogZ(path, tngs, vers, ylim=[1e-2, 1e1], levels = [0, 0.55], nlevels=30)