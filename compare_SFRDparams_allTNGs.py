import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15  as cosmo #Planck 2015 since that's what TNG uses
import seaborn as sns
from math import ceil
import matplotlib.ticker as ticker

# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
from TNG_BBHpop_properties import read_best_fits
from Fit_model_TNG_SFRD import readTNGdata, interpolate_TNGdata

## PLOT setttings
plt.rc('font', family='serif')
import matplotlib
plt.rc('font', family='serif', weight='bold')
plt.rc('text', usetex=True)
matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams.update({'font.weight': 'bold'})

def compare_SFR(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], error_ylim=[], plotmodel=True, plotredshift=True, show_MD17=True, 
                plotlogscale=False, showfractionalerr=True, showplot=True, transparent=False):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_%s.txt'%(tng))
    fit_params = read_best_fits(fit_param_files)
    
    if showfractionalerr==True:
        fig = plt.figure(layout='constrained',figsize=[13, 10])
        ax = fig.add_subplot()
        ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(13, 9))

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if 'TNG50' in tng:
            rbox=35
        elif 'TNG100' in tng:
            rbox=75
        elif 'TNG30' in tng:
            rbox=205
        elif tng=='COLIBRE':
            rbox = 200
        else:
            rbox=12
        print(rbox)

        #Plot the TNG data
        if tng=='COLIBRE':
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'sfr_metallicity_redshift_L200_m6_Thermal_feh.hdf5', rbox=rbox, SFR=False, metals=False)
            Sim_SFRD = Sim_SFRD[:, 1:]
            Sim_center_Zbin = Sim_center_Zbin[1:]
        try:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGasWithMetals%s.hdf5'%(tng), rbox=rbox, SFR=False, metals=False)
        except:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGas%s.hdf5'%tng, rbox=rbox, SFR=False, metals=False)

        if plotredshift == True:
            xvals = Sim_Redshifts
        else:
            xvals = Sim_Lookbacktimes

        if vers[n]==1:
            linestyle='-'
            z_order=1
        if vers[n]==2:
            linestyle='--'
            z_order=2
        if vers[n]==3:
            linestyle=':'
            z_order=3
        ax.plot(xvals, np.sum(Sim_SFRD, axis=1)*step_fit_logZ, lw=6, ls=linestyle, c=data_colors_all[n], zorder=z_order, label=tng)

        #Plot the TNG model
        if plotmodel==True:
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value
            ax.plot(xvals, sfr, lw=3, c=fit_colors[n], ls='--')

        if showfractionalerr==True:
            fractionalerr =  sfr/(np.sum(Sim_SFRD, axis=1)*step_fit_logZ)
            ax_x.plot(xvals, fractionalerr, lw=4, c=data_colors[n])
            ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    if plotmodel==True:
        x = [-0.0001]
        y1 = [0.0001] 
        y2 = [0.0001]
        plt.plot(x, y1, c='black', ls = '-', lw=6, label=r'$\mathrm{Simulation}$')
        plt.plot(x, y2, c='black', ls = '--', lw=3, label=r'$\mathrm{Analytical \ fit}$')
    ax.tick_params(axis='both', which='major', labelsize=25) 
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(r'$\mathcal{S}(z) \ [\rm M_{\odot} \ yr^{-1}  \ Mpc^{-3}]$', fontsize = 35)

    if show_MD17 == True:
        #default Madau & Fragos 17
        ax.plot(xvals, Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=0.01, b=2.6, c=3.2, d=6.2), 
            label = r'M\&F 2017', #+'\n'+'$a=%.2f, b=%.2f, c=%.2f, d=%.2f$'% (0.01,2.6,3.2,6.2), 
            c = 'gray',lw=3, zorder=0)

    ax.legend(fontsize = 25, frameon=False)
    if plotlogscale==True:
        ax.set_yscale('log')
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    if showfractionalerr==True:
        ax_x.tick_params(axis='y', which='major', labelsize=20)
        ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
        ax_x.tick_params(length=10, width=2, which='major')
        ax_x.tick_params(length=5, width=1, which='minor')
        ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax_x.set_ylabel(r'$\mathcal{S}(z)_\mathrm{fit}/\mathcal{S}(z)_\mathrm{sim}$', fontsize = 25)
        ax_x.set_yscale('log')
        #ax_x.legend(fontsize = 25, frameon=False)
        if len(error_ylim) > 0:
            ax_x.set_ylim(error_ylim[0], error_ylim[1])

    if plotredshift:
        ax.set_xlabel(r"Redshift $z$", fontsize = 35)

        if showfractionalerr==True:
            ax2 = ax_x.twiny()
        else:
            ax2 = ax.twiny()
        redshift_tick_list = [0, 1, 2, 6, 10, 14]#[0,0.1, 0.25, 0.5, 1.0, 10]
        ax2.set_xticks([z for z in redshift_tick_list])
        ax2.set_xticklabels(['${:.1f}$'.format(cosmo.lookback_time(z).value) for z in redshift_tick_list], fontsize = 25)
        #lookbackt_tick_list = [cosmo.lookback_time(z).value for z in redshift_tick_list]
        ax2.set_xlabel('Lookback time [Gyr]', fontsize = 35, labelpad=15)
        #ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
        #ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
        ax2.tick_params(axis='both', which='major', labelsize=23)
        ax2.tick_params(length=10, width=3, which='major')

        #Set limits for horizontal axis
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))
            ax2.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))

        fig.savefig('figures/SFR_redshift_allTNGs.pdf', format="pdf",  bbox_inches='tight', dpi=300, transparent=transparent)
    else:
        ax.set_xlabel(r"Lookback time (Gyr)", fontsize = 35)
        
        ax2 = ax_x.twiny()
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
        ax2.set_xlabel('Redshift $z$', fontsize = 35, labelpad=15)
        ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list])
        ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list])
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(length=10, width=3, which='major')

        #Set limits for horizontal axis
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))
            ax2.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))

        fig.savefig('figures/SFR_lookbackt_allTNGs.pdf',  format="pdf", bbox_inches='tight', dpi=300, transparent=transparent)
    
    if showplot==True:
        plt.show()


def compare_SFR(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], error_ylim=[], plotmodel=True, plotredshift=True, show_MD17=True, 
                plotlogscale=False, showfractionalerr=True, showplot=True, transparent=False):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_%s.txt'%(tng))
    fit_params = read_best_fits(fit_param_files)
    
    if showfractionalerr==True:
        fig = plt.figure(layout='constrained',figsize=[13, 10])
        ax = fig.add_subplot()
        ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(13, 9))

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if 'TNG50' in tng:
            rbox=35
        elif 'TNG100' in tng:
            rbox=75
        elif 'TNG30' in tng:
            rbox=205
        elif tng=='COLIBRE':
            rbox = 200
        else:
            rbox=12
        print(rbox)

        #Plot the TNG data
        if tng=='COLIBRE':
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'sfr_metallicity_redshift_L200_m6_Thermal_feh.hdf5', rbox=rbox, SFR=False, metals=False)
            Sim_SFRD = Sim_SFRD[:, 1:]
            Sim_center_Zbin = Sim_center_Zbin[1:]
        try:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGasWithMetals%s.hdf5'%(tng), rbox=rbox, SFR=False, metals=False)
        except:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGas%s.hdf5'%tng, rbox=rbox, SFR=False, metals=False)

        if plotredshift == True:
            xvals = Sim_Redshifts
        else:
            xvals = Sim_Lookbacktimes

        if vers[n]==1:
            linestyle='-'
            z_order=1
        if vers[n]==2:
            linestyle='--'
            z_order=2
        if vers[n]==3:
            linestyle=':'
            z_order=3
        ax.plot(xvals, np.sum(Sim_SFRD, axis=1)*step_fit_logZ, lw=6, ls=linestyle, c=data_colors_all[n], zorder=z_order, label=tng)

        #Plot the TNG model
        if plotmodel==True:
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value
            ax.plot(xvals, sfr, lw=3, c=fit_colors[n], ls='--')

        if showfractionalerr==True:
            fractionalerr =  sfr/(np.sum(Sim_SFRD, axis=1)*step_fit_logZ)
            ax_x.plot(xvals, fractionalerr, lw=4, c=data_colors[n])
            ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    if plotmodel==True:
        x = [-0.0001]
        y1 = [0.0001] 
        y2 = [0.0001]
        plt.plot(x, y1, c='black', ls = '-', lw=6, label=r'$\mathrm{Simulation}$')
        plt.plot(x, y2, c='black', ls = '--', lw=3, label=r'$\mathrm{Analytical \ fit}$')
    ax.tick_params(axis='both', which='major', labelsize=25) 
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(r'$\mathcal{S}(z) \ [\rm M_{\odot} \ yr^{-1}  \ Mpc^{-3}]$', fontsize = 35)

    if show_MD17 == True:
        #default Madau & Fragos 17
        ax.plot(xvals, Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=0.01, b=2.6, c=3.2, d=6.2), 
            label = r'M\&F 2017', #+'\n'+'$a=%.2f, b=%.2f, c=%.2f, d=%.2f$'% (0.01,2.6,3.2,6.2), 
            c = 'gray',lw=3, zorder=0)

    ax.legend(fontsize = 25, frameon=False)
    if plotlogscale==True:
        ax.set_yscale('log')
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    if showfractionalerr==True:
        ax_x.tick_params(axis='y', which='major', labelsize=20)
        ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
        ax_x.tick_params(length=10, width=2, which='major')
        ax_x.tick_params(length=5, width=1, which='minor')
        ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax_x.set_ylabel(r'$\mathcal{S}(z)_\mathrm{fit}/\mathcal{S}(z)_\mathrm{sim}$', fontsize = 25)
        ax_x.set_yscale('log')
        #ax_x.legend(fontsize = 25, frameon=False)
        if len(error_ylim) > 0:
            ax_x.set_ylim(error_ylim[0], error_ylim[1])

    if plotredshift:
        ax.set_xlabel(r"Redshift $z$", fontsize = 35)

        if showfractionalerr==True:
            ax2 = ax_x.twiny()
        else:
            ax2 = ax.twiny()
        redshift_tick_list = [0, 1, 2, 6, 10, 14]#[0,0.1, 0.25, 0.5, 1.0, 10]
        ax2.set_xticks([z for z in redshift_tick_list])
        ax2.set_xticklabels(['${:.1f}$'.format(cosmo.lookback_time(z).value) for z in redshift_tick_list], fontsize = 25)
        #lookbackt_tick_list = [cosmo.lookback_time(z).value for z in redshift_tick_list]
        ax2.set_xlabel('Lookback time [Gyr]', fontsize = 35, labelpad=15)
        #ax2.set_xticks([cosmo.lookback_time(z).value for z in lookbackt_tick_list])
        #ax2.set_xticklabels(['${:.0f}$'.format(z) for z in lookbackt_tick_list])
        ax2.tick_params(axis='both', which='major', labelsize=23)
        ax2.tick_params(length=10, width=3, which='major')

        #Set limits for horizontal axis
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))
            ax2.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))

        fig.savefig('figures/SFR_redshift_all.pdf', format="pdf",  bbox_inches='tight', dpi=300, transparent=transparent)
    else:
        ax.set_xlabel(r"Lookback time (Gyr)", fontsize = 35)
        
        ax2 = ax_x.twiny()
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
        ax2.set_xlabel('Redshift $z$', fontsize = 35, labelpad=15)
        ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list])
        ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list])
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(length=10, width=3, which='major')

        #Set limits for horizontal axis
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))
            ax2.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))

        fig.savefig('figures/SFR_lookbackt_allTNGs.pdf',  format="pdf", bbox_inches='tight', dpi=300, transparent=transparent)
    
    if showplot==True:
        plt.show()

def compare_Zdist(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], error_ylim=[], plotmodel=True, plotlogscale=False, showplot=True, 
                  showfractionalerr=True, transparent=False):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_%s.txt'%(tng))
    fit_params = read_best_fits(fit_param_files)
    
    if showfractionalerr==True:
        fig = plt.figure(layout='constrained',figsize=[13, 10])
        ax = fig.add_subplot()
        ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=[13, 9])

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if 'TNG50' in tng:
            rbox=35
        elif 'TNG100' in tng:
            rbox=75
        elif 'TNG30' in tng:
            rbox=205
        elif tng=='COLIBRE':
            rbox = 200
        else:
            rbox=12
        print(rbox)

        #Plot the TNG data
        #if vers[n]==1:
        if tng=='COLIBRE':
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'sfr_metallicity_redshift_L200_m6_Thermal_feh.hdf5', rbox=rbox, SFR=False, metals=False)
            Sim_SFRD = Sim_SFRD[:, 1:]
            Sim_center_Zbin = Sim_center_Zbin[1:]
        try:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGasWithMetals%s.hdf5'%(tng), rbox=rbox, SFR=False, metals=False)
        except:
            Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(tng, loc = path+'SFRMetallicityFromGas%s.hdf5'%(tng), rbox=rbox, SFR=False, metals=False)
        #SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = interpolate_TNGdata(Sim_Redshifts, Sim_Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], redshiftlimandstep=[0, 14.1, 0.05], saveplot=False)
        #scale = max(np.sum(Sim_SFRD, axis=0))/max(np.sum(SFRDnew.T, axis=0))

        if vers[n]==1:
            linestyle='-'
            z_order=1
        if vers[n]==2:
            linestyle='--'
            z_order=2
        if vers[n]==3:
            linestyle=':'
            z_order=3
        ax.plot(Sim_center_Zbin/Zsun, np.sum(Sim_SFRD, axis=0), lw=6, ls=linestyle, zorder=z_order, c=data_colors_all[n], label=tng)

        #Plot the TNG model
        if plotmodel==True:
            fit_param = fit_params[n]
            #metals_bins = 10**(np.arange( -10, 0., 0.2 ))
            #metals  = (metals_bins[:-1] + metals_bins[1:])/2.
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_param[5], b=fit_param[6], c=fit_param[7], d=fit_param[8]).value # Msun year-1 Mpc-3 
            dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                    Z_SFRD.skew_metallicity_distribution(Sim_Redshifts , mu_0 = fit_param[0], mu_z = fit_param[1],
                                                    omega_0= fit_param[2] , omega_z=fit_param[3] , alpha = fit_param[4], 
                                                    metals=Sim_center_Zbin, min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03))
            model = sfr[:,np.newaxis] * dPdlogZ
            ax.plot(Sim_center_Zbin/Zsun, np.sum(model.T, axis=1), lw=3, c=fit_colors[n], ls='--') 

        if showfractionalerr==True:
            fractionalerr =  np.sum(model, axis=0)/np.sum(Sim_SFRD, axis=0) 
            ax_x.plot(Sim_center_Zbin/Zsun, fractionalerr, lw=4, c=data_colors[n])
            ax_x.axvline(x=1, linewidth=2, color='gray', zorder=0)
            ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    ax.axvline(x=1, linewidth=2, color='gray', zorder=0)
    ax.text(x=0.55, y= 0.002, s='$Z_\odot$', fontsize=30, color='gray')

    if plotmodel==True:
        x = [-0.0001]
        y1 = [0.0001]
        y2 = [0.0001]
        plt.plot(x, y1, c='black', ls = '-', lw=6, label=r'$\mathrm{Simulation}$')
        plt.plot(x, y2, c='black', ls = '--', lw=3, label=r'$\mathrm{Analytical \ fit}$')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel(r"Z/Z$_\odot$", fontsize = 35)
    ax.set_ylabel(r'$\mathcal{S}(Z) \ [\rm M_{\odot} \ yr^{-1} \ Mpc^{-3}]$', fontsize = 35)

    ax.legend(fontsize = 25, frameon=False)
    if plotlogscale==True:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    if showfractionalerr==True:
        ax_x.tick_params(axis='y', which='major', labelsize=20)
        ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
        ax_x.tick_params(length=10, width=2, which='major')
        ax_x.tick_params(length=5, width=1, which='minor')
        ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax_x.set_ylabel(r'$\mathcal{S}(Z)_\mathrm{fit}/\mathcal{S}(Z)_\mathrm{sim}$', fontsize = 25)
        ax_x.set_xscale('log')
        ax_x.set_yscale('log')
        #ax_x.legend(fontsize = 25, frameon=False)
        if len(error_ylim) > 0:
            ax_x.set_ylim(error_ylim[0], error_ylim[1])

    #Set limits for horizontal axis
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
    else:
        ax.set_xlim(min(Sim_center_Zbin/Zsun), max(Sim_center_Zbin/Zsun))

    fig.savefig('figures/Z_avg_sfr_allTNGs.pdf',  format="pdf", bbox_inches='tight', dpi=300, transparent=transparent)
    
    if showplot==True:
        plt.show()



if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    path = '/Users/sashalvna/Research/Fit_SFRD_TNG/data/'
    tngs=['TNG50-1', 'TNG50-2', 'TNG50-3', 'TNG100-1', 'TNG100-2', 'TNG300-1'] 
    vers = [1, 2, 3, 1, 2, 1]
    Zsun = 0.014

    cmap_blue = matplotlib.colors.LinearSegmentedColormap.from_list("blue_cmap", ['#40E9E0', '#1C7EB7', '#100045'])
    cmap_pink = matplotlib.colors.LinearSegmentedColormap.from_list("pink_cmap", ['#F76FDD', '#C13277', '#490013'])
    cmap_green = matplotlib.colors.LinearSegmentedColormap.from_list("green_cmap", ['#CCE666', '#79B41C', '#004011'])
    cmap_gray = matplotlib.colors.LinearSegmentedColormap.from_list("gray_cmap", ['#D3D2D2', '#787878', '#222222'])
    #data_colors = ["#0067A6", '#C01874', '#98CB4F', '#D3D2D2']
    data_colors = ["#0067A6", '#C01874', '#98CB4F',"#F6A030"]
    data_colors_all = ['#40E9E0', '#1C7EB7', "#00446E",'#F76FDD', "#900849", '#98CB4F', 'black', 'gray', 'lightgray']
    #data_colors_all = [tol_colors.muted.teal, tol_colors.medium_contrast.light_blue, tol_colors.medium_contrast.dark_blue,tol_colors.muted.wine, tol_colors.muted.purple, tol_colors.muted.olive, 'black', 'gray', 'lightgray']
    #fit_colors = ['#0C0034', '#4B0012', '#005B2F', '#787878']
    fit_colors = ['#0C0034', '#4B0012', '#005B2F', "#D85706"]
    line_styles = ['solid', 'dashed', 'dotted']

    params = [['mu_0', 'sigma_0'], ['mu_z', 'sigma_z'], ['mu_0', 'mu_z'], ['sigma_0', 'sigma_z'], ['sf_a', 'sf_c'], ['sf_b', 'sf_d']]
    #for n in params:
    #compare_params(tngs, vers, params=params, other_sfrd_params=['vanSon23_fid'], showplot=True)

    compare_SFR(path, tngs, vers, plotmodel=False, plotredshift=True, xlim=[0, 20], ylim=[10**-5, 10**-0.8], #, error_ylim = [5e-1, 1e1], 
                 showfractionalerr=False, plotlogscale=True, transparent=True)

    compare_Zdist(path, tngs, vers, xlim=[10**-4, 10], ylim=[1e-3, 4], #error_ylim=[1e-1, 1e3], 
                  plotmodel=False, plotlogscale=True, showfractionalerr=False, transparent=True)

    #Zdist_3panel_plots(path, tng=100, ver=1, showplot=True, transparent=False)s

    #compare_dPdlogZ(path, tngs, vers, ylim=[1e-2, 1e1], levels = [0, 0.55], nlevels=30)