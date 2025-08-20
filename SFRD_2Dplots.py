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

## PLOT setttings
plt.rc('font', family='serif')
from matplotlib import rc
import matplotlib
plt.rc('font', family='serif', weight='bold')
plt.rc('text', usetex=True)
matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams.update({'font.weight': 'bold'})


def SFRDplot_2D(metals, Lookbacktimes, SFRD, tngs=[], ver=[], model=None, xlim=[], ylim=[10**-1, 10**1], levels = [], plottype='data', modelplot=True, plotregions=False, nlevels=20, cmap='rocket', showplot=True):

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
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['tab:red', 'white', 'tab:blue'])

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
            if model:
                ax.text(0.02, 0.02, "TNG%s-%s, analytical fit"%(tng, ver[n]), transform=ax.transAxes, fontsize=20, color='white')

        if plottype=='percenterr':
            #SFRD[n][SFRD[n] < 1e-7] = 1e-7
            #model[n].T[model[n].T < 1e-7] = 1e-7
            percenterr = (model[n].T-SFRD[n])/SFRD[n]
            levels = np.linspace(-2, 2, nlevels)
            data = ax.contourf(Lookbacktimes[n], metals[n]/Zsun, percenterr, levels=levels, cmap=cmap)
            print(max(Lookbacktimes[n]))
            ax.text(0.02, 0.02, "TNG%s-%s, analytical fit"%(tng, ver[n]), transform=ax.transAxes, fontsize=20, color='black')

        if model:
            #Model contours
            if tng==50:
                clevels = [1e-5, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05] 
            elif tng==100:
                clevels = [1e-5, 1e-3, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.033] 
            elif tng==300:
                clevels = [1e-5, 1e-3, 0.005, 0.01, 0.015,0.02, 0.024] 
            if plottype=='data':
                modelplot = ax.contour(Lookbacktimes[n], metals[n]/Zsun, model[n].T, levels=clevels, colors='white')
                #ax.clabel(modelplot, fontsize=11, inline=True)
            elif plottype=='percenterr':
                modelplot = ax.contour(Lookbacktimes[n], metals[n]/Zsun, model[n].T, levels=clevels, colors='black', linestyles='dashed')

        #Set yscale, TNG version label on each plot, and axis labels
        ax.set_yscale('log')
        fig.supxlabel('Lookback time (Gyr)', y=0.04, fontsize=30)
        fig.supylabel(r'$Z/Z_{\rm{\odot}}$', x=0.03, fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(length=10, width=2, which='major')
        ax.tick_params(length=5, width=1, which='minor')

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
            labels = ax.get_yticklabels()
            labels[-1] = ''
            ax.set_yticklabels(labels)

        #Hide tick labels depending on how many subplots are in the figure, and their configuration
        if n%2==0:
            if len(tngs) > 2 and n<len(tngs)-2: #if left column with more than two subplots, but not last row, hide ticks and labels
                ax.tick_params(axis='x',  which='both', bottom=False, labelbottom=False)
        else:
            if len(tngs) > 1 and len(tngs) <= 2: #if right subplot with only one row, hide left ticks on the left side
                ax.tick_params(axis='both',  which='both', left=False, labelleft=False)
            elif len(tngs) > 2 and n<len(tngs)-2: #if right subplot with more than one row, hide left and bottom ticks
                ax.tick_params(axis='both',  which='both', left=False, labelleft=False, bottom=False, labelbottom=False)
            else: #if last right subplot, hide left label
                ax.tick_params(axis='both',  which='both', left=False, labelleft=False)

        #Set redshift axis labels for only first row of subplots
        if n < 2: 
            ax2.set_xlabel('Redshift', fontsize = 25, labelpad=5)
            ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list])
            ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list])
        else:
            ax2.set_xlabel('', fontsize = 25)
            ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list*0])
            ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list*0])

        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(length=5, width=1.5, which='major')

        #Set limits for horizontal (lookback time) and vertical (metallicity) axes
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Lookbacktimes[n]), max(Lookbacktimes[n]))
            ax2.set_xlim(min(Lookbacktimes[n]), max(Lookbacktimes[n]))

        if len(ylim) > 0:
            ax.set_ylim(ylim[0], ylim[1]) #defaults for TNG data are Z=10**-1 to Z=10**1
            
        if plotregions==True:

            #lowz_bin = [cosmo.lookback_time(4).value, cosmo.lookback_time(8).value]
            #midz_bin = [cosmo.lookback_time(10).value, cosmo.lookback_time(12).value]
            #highz_bin = [cosmo.lookback_time(12).value, cosmo.lookback_time(14).value]
            lowz_bin = [4, 8]
            midz_bin = [10, 12]
            highz_bin = [12, 14]
            zbins = np.array([lowz_bin, midz_bin, highz_bin])
            zbins_lookbackt = cosmo.lookback_time(zbins).value

            lowZ_bin = [0.0001, 0.001]
            midZ_bin = [0.001, 0.01]
            highZ_bin = [0.01, 0.1]
            Zbins = np.array([lowZ_bin, midZ_bin, highZ_bin])
            Zbins_Zsun = Zbins/Zsun

            Zzbin_colors = ['darkgreen', 'mediumseagreen', 'lightgreen',
                            'darkorange', 'orange', 'gold',
                            'purple', 'mediumvioletred', 'pink']
            counter = 0

            for zbin in zbins_lookbackt:
                for Zbin in Zbins_Zsun:
                    rect = matplotlib.patches.Rectangle((zbin[0], Zbin[0]), zbin[1]-zbin[0], Zbin[1]-Zbin[0], linewidth=2, edgecolor=Zzbin_colors[counter], facecolor='none')
                    ax.add_patch(rect)
                    counter+=1

            #for limits in plotregions:
            #    rect = matplotlib.patches.Rectangle((limits[0], limits[1]), limits[2], limits[3], linewidth=1, edgecolor='gray', facecolor='none')
            #    ax.add_patch(rect)

    
    #Set up the colorbar
    fig.subplots_adjust(right=0.81)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(data, cax=cbar_ax, format=ticker.FormatStrFormatter('%.3f'))
    cbar.ax.tick_params(labelsize=20)

    #Set the colorbar labels and save the plot
    if plottype=='data':
        if len(tngs)==1:
            if ver[0]>1:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s-%s data'%(tngs[0], ver[0]), rotation=270, fontsize=30, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%s.png'%(tngs[0], ver[0]), bbox_inches='tight', dpi=300)
            else:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG%s data'%tngs[0], rotation=270, fontsize=30, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s.png'%tngs[0], bbox_inches='tight', dpi=300)
        else:
            cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ TNG data', rotation=270, fontsize=30, labelpad=30);
            fig.savefig('figures/SFRD_Z_z_TNG_1.png', bbox_inches='tight', dpi=300, transparent=True)
    elif plottype=='percenterr':
        if len(tngs)==1:
            if ver[0]>1:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ percent error, TNG%s-%s data'%(tngs[0], ver[0]), rotation=270, fontsize=30, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%s_%sdiff.png'%(tngs[0], ver[0]), bbox_inches='tight')
            else:
                cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ percent error, TNG%s data'%tngs[0], rotation=270, fontsize=30, labelpad=30);
                fig.savefig('figures/SFRD_Z_z_TNG%sdiff.png'%tngs[0], bbox_inches='tight', dpi=300)
        else:
            cbar.set_label(r'$\mathcal{S}(Z_{\rm{i}},z)$ percent error', rotation=270, fontsize=30, labelpad=30);
            fig.savefig('figures/SFRD_Z_z_TNGdiff_regions.png', bbox_inches='tight', dpi=300)
    
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


def SFRD_2Dplot_sidepanels(metals, Redshifts, Lookbacktimes, SFRD, step_fit_logZ, tng=[], ver=[], model=None, xlim=[], ylim=[10**-1, 10**1], levels = [], 
                           nlevels=20, plottype='data', plotredshift=True, plotregions=False, showplot=True):

    """
    plottype (str): type of BBH 2D plot, options: 'data', 'percenterr'
    """

    if plottype == 'data':
        if len(levels) > 0:
            levels = np.linspace(levels[0], levels[1], nlevels)
        else:
            levels = np.linspace(np.amin(SFRD), np.amax(SFRD), nlevels)
        cmap = sns.color_palette('rocket', as_cmap=True)
    if plottype == 'percenterr':
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ['tab:red', 'white', 'tab:blue'])

    fig = plt.figure(layout='constrained',figsize=[14,10])
    ax = fig.add_subplot()
    ax_histx = ax.inset_axes([0, 1.02, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.02, 0, 0.2, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if plotredshift == True:
        xvals = Redshifts
    else:
        xvals = Lookbacktimes

    #Plot data and is model is given plot model
    if plottype=='data':

        data = ax.contourf(xvals, metals/Zsun, SFRD, levels=levels, cmap=cmap)
        if len(model) > 0:
            ax.text(0.02, 0.02, "TNG%s-%s, analytical fit"%(tng, ver), transform=ax.transAxes, fontsize=20, color='white')

        print('hi', step_fit_logZ)
        print('hiii', xvals[1]-xvals[0])

        ax_histx.plot(xvals, np.sum(SFRD, axis=0)*step_fit_logZ)
        ax_histy.plot(np.sum(SFRD, axis=1)*(xvals[1]-xvals[0]), metals/Zsun)
        #ax_histx.set_ylim(0, 6)
        #ax_histy.set_xlim(0, 4)

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
            modelplot = ax.contour(xvals, metals/Zsun, model.T, levels=levels, colors='white')
            ax_histx.plot(xvals, np.sum(model.T, axis=0)*step_fit_logZ)
            ax_histy.plot(np.sum(model.T, axis=1)*(xvals[1]-xvals[0]), metals/Zsun)
            #ax.clabel(modelplot, fontsize=11, inline=True)
        elif plottype=='percenterr':
            modelplot = ax.contour(xvals, metals/Zsun, model.T, levels=levels, colors='black', linestyles='dashed')

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
    step_fit_logZ_TNG = []
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
        Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ = readTNGdata(loc = Cosmol_sim_location, rbox=rbox, SFR=False, metals=False)
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = fitmodel.interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], redshiftlimandstep=[0, 14.1, 0.05], saveplot=False)
        SFRDsTNG.append(SFRDnew)
        redshiftsTNG.append(redshift_new)
        LookbacktimesTNG.append(Lookbacktimes_new)
        metalsTNG.append(metals_new)
        step_fit_logZ_TNG.append(step_fit_logZ_new)
        param_filenames.append(fit_filename)

        print(max(np.sum(Sim_SFRD, axis=1)))
        print(max(np.sum(Sim_SFRD, axis=0)), max(np.sum(Sim_SFRD, axis=1)))
        print(max(np.sum(SFRDnew, axis=1)), max(np.sum(SFRDnew, axis=0)))
        print(max(np.sum(SFRDnew, axis=1)*step_fit_logZ_new), max(np.sum(SFRDnew, axis=0)*0.05))

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


    #SFRDplot_2D(metalsTNG, LookbacktimesTNG, SFRDsTNG, tngs, vers, ylim=[10**-4, 50], nlevels=17, model=models, plottype='percenterr', plotregions=True)
    SFRD_2Dplot_sidepanels(metalsTNG[0], redshiftsTNG[0], LookbacktimesTNG[0], SFRDsTNG[0], step_fit_logZ_TNG[0], tngs[0], vers[0], xlim=[14, 0], ylim=[10**-4, 50], nlevels=17, model=models[0], plottype='data', plotredshift=False, plotregions=False)

    
