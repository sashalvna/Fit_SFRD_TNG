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
from matplotlib import rc
import matplotlib
plt.rc('font', family='serif', weight='bold')
plt.rc('text', usetex=True)
matplotlib.rcParams['font.weight']= 'bold'
matplotlib.rcParams.update({'font.weight': 'bold'})

def compare_params(tngs=[50, 100, 300], vers=[1, 1, 1], showplot=True):
    
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)


    #ticks =  ["mean metallicity at z=0", "redshift evolution of mean", "variance in metallicity density distribution", 
    #          "redshift evolution of variance", "skewness","SFR scaling", "SFR upward slope", "SFR peak location", "SFR downward slope"]
    ticks = [r'$\mu_0$', r'$\mu_z$', r'$\sigma_0$', r'$\sigma_z$', r'$\alpha$', r'$a$', r'$b$', r'$c$', r'$d$']
    x = [0,1,2,3,4,5,6,7,8] #9 parameters

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    for n, fits in enumerate(fit_params):
        ax[0].plot(x, fits, label="TNG%s-%s"%(tngs[n], vers[n]), lw=3, color=data_colors[n], ls = line_styles[n])

        if n != 1:
            ax[1].plot(x, fit_params[n]/fit_params[1], label='TNG%s-%s/TNG%s-%s'%(tngs[n], vers[n], tngs[1], vers[1]), lw=3, color=data_colors[n])
    
    ax[0].set_xticks(x)
    ax[0].legend(fontsize=15, frameon=True)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].set_xticklabels(ticks, fontsize=20)
    ax[0].tick_params(length=10, width=2, which='major')
    ax[0].tick_params(length=5, width=1, which='minor')
    ax[0].set_ylabel(r'$\mathcal{S}(Z_{\rm{i}},z)$ fit parameter value', fontsize=20)
    ax[0].xaxis.grid(True)

    ax[1].set_xticks(x)
    ax[1].legend(fontsize=15, frameon=True)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].set_xticklabels(ticks, fontsize=20)
    ax[1].tick_params(length=10, width=2, which='major')
    ax[1].tick_params(length=5, width=1, which='minor')
    ax[1].set_ylabel(r'Percent error', fontsize=20)
    ax[1].xaxis.grid(True)

    fig.savefig('figures/fitparams_percenterror.png', bbox_inches='tight', dpi=300)
    
    if showplot==True:
        plt.show()


def compare_SFR(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], error_ylim=[], plotmodel=True, plotredshift=True, show_MD17=True, plotlogscale=False, showplot=True):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)
    
    fig = plt.figure(layout='constrained',figsize=[13, 10])
    ax = fig.add_subplot()
    ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        #Plot the TNG data
        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ, Metals, MetalBins  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=True, metals=True)

        if plotredshift == True:
            xvals = Sim_Redshifts
        else:
            xvals = Sim_Lookbacktimes

        ax.plot(xvals, np.sum(Sim_SFRD, axis=1), lw=8, c=data_colors[n], label=r'TNG%s-%s'%(tng, vers[n]))

        #Plot the TNG model
        if plotmodel==True:
            
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value
            ax.plot(xvals, sfr, lw=3, c=fit_colors[n], ls='--')

        fractionalerr =  sfr/np.sum(Sim_SFRD, axis=1)
        ax_x.plot(xvals, fractionalerr, lw=4, c=data_colors[n])

    x = [-0.0001]
    y1 = [0.0001] 
    y2 = [0.0001]
    plt.plot(x, y1, c='black', ls = '-', lw=6, label=r'$\mathrm{TNG \ simulation}$')
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
        
    ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    ax.legend(fontsize = 25, frameon=False)
    if plotlogscale==True:
        ax.set_yscale('log')
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    ax_x.tick_params(axis='y', which='major', labelsize=20)
    ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
    ax_x.tick_params(length=10, width=2, which='major')
    ax_x.tick_params(length=5, width=1, which='minor')
    ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_x.set_ylabel(r'$\mathcal{S}(z)_\mathrm{fit}/\mathcal{S}(z)_\mathrm{sim}$', fontsize = 20)
    ax_x.set_yscale('log')
    #ax_x.legend(fontsize = 25, frameon=False)
    if len(error_ylim) > 0:
        ax_x.set_ylim(error_ylim[0], error_ylim[1])

    if plotredshift:
        ax.set_xlabel(r"Redshift $z$", fontsize = 35)

        ax2 = ax_x.twiny()
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

        if plotmodel:
            fig.savefig('figures/SFR_redshift_fit.pdf', format="pdf", bbox_inches='tight', dpi=300)
        else:
            fig.savefig('figures/SFR_redshift.pdf', format="pdf",  bbox_inches='tight', dpi=300)
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

        if plotmodel:
            fig.savefig('figures/SFR_lookbackt_fit.pdf',  format="pdf", bbox_inches='tight', dpi=300)
        else:
            fig.savefig('figures/SFR_lookbackt.pdf',  format="pdf", bbox_inches='tight', dpi=300)
    
    if showplot==True:
        plt.show()

def compare_Zdist(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], error_ylim=[], plotmodel=True, plotlogscale=False, showplot=True):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)
    
    fig = plt.figure(layout='constrained',figsize=[13, 10])
    ax = fig.add_subplot()
    ax_x = ax.inset_axes([0, 1.0, 1, 0.30], sharex=ax)

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        #Plot the TNG data
        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=False, metals=False)
        plt.plot(Sim_center_Zbin/Zsun, np.sum(Sim_SFRD, axis=0), lw=8, c=data_colors[n], label=r'TNG%s-%s'%(tng, vers[n]))

        #Plot interpolated TNG data
        #SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = interpolate_TNGdata(Sim_Redshifts, Sim_Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], redshiftlimandstep=[0, 14.1, 0.05], saveplot=False)
        #plt.plot(metals_new/Zsun, np.sum(SFRDnew, axis=1), lw=3, c=fit_colors[n], label=r'interpolated')

        #Plot the TNG model
        if plotmodel==True:
            sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value # Msun year-1 Mpc-3 
            dPdlogZ, metallicities, step_logZ, p_draw_metallicity = Z_SFRD.skew_metallicity_distribution(Sim_Redshifts, metals = Sim_center_Zbin, min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03),
                        mu_0=fit_params[n][0], mu_z=fit_params[n][1], omega_0=fit_params[n][2], omega_z=fit_params[n][3], alpha =-fit_params[n][4])
            model = sfr[:,np.newaxis] * dPdlogZ
            ax.plot(Sim_center_Zbin/Zsun, np.sum(model, axis=0), lw=3, c=fit_colors[n], ls='--') 

        fractionalerr =  np.sum(model, axis=0)/np.sum(Sim_SFRD, axis=0) 
        ax_x.plot(Sim_center_Zbin/Zsun, fractionalerr, lw=4, c=data_colors[n])

    ax.axvline(x=1, linewidth=2, color='gray', zorder=0)
    ax.text(x=0.55, y= 0.002, s='$Z_\odot$', fontsize=30, color='gray')
    ax_x.axvline(x=1, linewidth=2, color='gray', zorder=0)
    ax_x.axhline(y=1, linewidth=1, color='gray', zorder=0)

    x = [-0.0001]
    y1 = [0.0001]
    y2 = [0.0001]
    plt.plot(x, y1, c='black', ls = '-', lw=6, label=r'$\mathrm{TNG \ simulation}$')
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

    ax_x.tick_params(axis='y', which='major', labelsize=20)
    ax_x.tick_params(axis='x', which='both', direction='in', labelbottom=False)
    ax_x.tick_params(length=10, width=2, which='major')
    ax_x.tick_params(length=5, width=1, which='minor')
    ax_x.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_x.set_ylabel(r'$\mathcal{S}(Z)_\mathrm{fit}/\mathcal{S}(Z)_\mathrm{sim}$', fontsize = 20)
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

    if plotmodel:
        fig.savefig('figures/Z_fit_avg_sfr.pdf',  format="pdf", bbox_inches='tight', dpi=300)
    else:
        fig.savefig('figures/Z_avg_sfr.pdf',  format="pdf", bbox_inches='tight', dpi=300)
    
    if showplot==True:
        plt.show()


def SFR_residuals(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], plotredshift=True, show_MD17=True, plotlogscale=True, showplot=True):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)
    
    fig, ax = plt.subplots(figsize=(13,8))

    #gridline for reference
    xs = np.arange(0, 20)
    ys = np.zeros(len(xs))
    plt.plot(xs, ys, c='lightgray', ls='-', lw=3)

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        #Plot the model-data residuals
        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=True, metals=False)

        if plotredshift == True:
            xvals = Sim_Redshifts
        else:
            xvals = Sim_Lookbacktimes

        sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value
        #residuals =  abs(sfr - np.sum(Sim_SFRD, axis=1))/np.sum(Sim_SFRD, axis=1) * 100
        residuals =  abs(np.sum(Sim_SFRD, axis=1) - sfr)/sfr * 100

        plt.plot(xvals, residuals, lw=4, c=data_colors[n], label=r'TNG%s-%s'%(tng, vers[n]))

    if show_MD17 == True:
        #default Madau & Fragos 17
        ax.plot(xvals, Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=0.01, b=2.6, c=3.2, d=6.2), 
            label = r'M\&F 2017', #+'\n'+'$a=%.2f, b=%.2f, c=%.2f, d=%.2f$'% (0.01,2.6,3.2,6.2), 
            c = 'gray', lw=3)

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylabel(r'$\mathcal{S}(z)$ \% error', fontsize = 35)
    #Set redshift ticks; make sure they don't overlap

    if plotlogscale==True:
        ax.set_yscale('log')
    ax.legend(fontsize = 25, frameon=False)
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    if plotredshift:
        ax.set_xlabel(r"Redshift", fontsize = 35)

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

        fig.savefig('figures/SFR_res_redshift_fit.pdf',  format="pdf", bbox_inches='tight', dpi=300)

    else:
        ax.set_xlabel(r"Lookback time [Gyr]", fontsize = 35)
        
        ax2 = ax.twiny()
        redshift_tick_list = [0,0.1, 0.25, 0.5, 0.75, 1.0,1.5, 2, 3, 6, 10]
        ax2.set_xlabel('Redshift', fontsize = 35, labelpad=15)
        ax2.set_xticks([cosmo.lookback_time(z).value for z in redshift_tick_list])
        ax2.set_xticklabels(['${:g}$'.format(z) for z in redshift_tick_list])
        ax2.tick_params(axis='both', which='major', labelsize=23)
        ax2.tick_params(length=10, width=3, which='major')

        #Set limits for horizontal axis
        if len(xlim) > 0:
            ax.set_xlim(xlim[0], xlim[1]) #not always want to set xlimits, so empty if not using any; in lookback time
            ax2.set_xlim(xlim[0], xlim[1]) #in lookback time
        else:
            ax.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))
            ax2.set_xlim(min(Sim_Lookbacktimes), max(Sim_Lookbacktimes))

        fig.savefig('figures/SFR_res_lookbackt_fit.pdf',  format="pdf", bbox_inches='tight', dpi=300)
    
    if showplot==True:
        plt.show()

def Zdist_residuals(path, tngs=[50, 100, 300], vers=[1, 1, 1], xlim=[], ylim=[], plotlogscale=True, showplot=True):

    #Get model fit parameters 
    fit_param_files = []
    for n, tng in enumerate(tngs):
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)
    
    fig, ax = plt.subplots(figsize=(13,8))

    #gridline for reference
    xs = np.arange(0, 20)
    ys = np.zeros(len(xs))
    plt.plot(xs, ys, c='lightgray', ls='-', lw=3)

    #Plot the TNG data
    for n, tng in enumerate(tngs):

        if tng==50:
            rbox=35
        elif tng==100:
            rbox=75
        elif tng==300:
            rbox=205

        #Plot the model-data residuals
        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=False, metals=False)
        sfr = Z_SFRD.Madau_Dickinson2014(Sim_Redshifts, a=fit_params[n][5], b=fit_params[n][6], c=fit_params[n][7], d=fit_params[n][8]).value # Msun year-1 Mpc-3 
        dPdlogZ, metallicities, step_logZ, p_draw_metallicity = Z_SFRD.skew_metallicity_distribution(Sim_Redshifts, metals = Sim_center_Zbin, min_logZ_COMPAS = np.log(1e-4), max_logZ_COMPAS = np.log(0.03),
                        mu_0=fit_params[n][0], mu_z=fit_params[n][1], omega_0=fit_params[n][2], omega_z=fit_params[n][3], alpha =-fit_params[n][4])
        model = sfr[:,np.newaxis] * dPdlogZ

        #residuals =  abs(np.sum(model, axis=0) - np.sum(Sim_SFRD, axis=0))/np.sum(Sim_SFRD, axis=0) * 100
        residuals =  abs(np.sum(Sim_SFRD, axis=0) - np.sum(model, axis=0))/np.sum(model, axis=0) * 100

        plt.plot(Sim_center_Zbin/Zsun, residuals, lw=4, c=data_colors[n], label=r'TNG%s-%s'%(tng, vers[n]))

    ax.axvline(x=1, linewidth=2, color='gray', zorder=0)
    ax.text(x=0.55, y= 0.04, s='$Z_\odot$', fontsize=30, color='gray')

    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(length=15, width=3, which='major')
    ax.tick_params(length=10, width=2, which='minor')
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel(r"Z/Z$_\odot$", fontsize = 35)
    ax.set_ylabel(r'$\mathcal{S}(Z)$ \% error', fontsize = 35)
    #Set redshift ticks; make sure they don't overlap

    if plotlogscale==True:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.legend(fontsize = 25, frameon=False)
    if len(xlim) > 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        ax.set_ylim(ylim[0], ylim[1])

    fig.savefig('figures/Zdist_res.pdf',  format="pdf", bbox_inches='tight', dpi=300)
    
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

        Sim_SFRD, Sim_Lookbacktimes, Sim_Redshifts, Sim_center_Zbin, step_fit_logZ  = readTNGdata(loc = path+'SFRMetallicityFromGasWithMetalsTNG%s-%s.hdf5'%(tng, vers[n]), rbox=rbox, SFR=True, metals=False)
        SFRDnew, redshift_new, Lookbacktimes_new, metals_new, step_fit_logZ_new = interpolate_TNGdata(Sim_Redshifts, Sim_Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, vers[n], saveplot=False)

        dPdlogZ, metallicities, step_logZ, p_draw_metallicity = \
                Z_SFRD.skew_metallicity_distribution(redshift_new , mu_0 = fit_params[n][0], mu_z = fit_params[n][1],
                                                  omega_0= fit_params[n][2] , omega_z=fit_params[n][3] , alpha = fit_params[n][4], 
                                                  metals=metals_new)

        #Model plot
        modelplot = ax.contourf(Lookbacktimes_new, metals_new/Zsun, dPdlogZ.T, levels=levels, cmap=cmap)
        ax.text(0.02, 0.02, "TNG%s-%s"%(tng, vers[n]), transform=ax.transAxes, fontsize=15, color='white')
        
        #Set yscale, TNG version label on each plot, and axis labels
        ax.set_yscale('log')
        fig.supxlabel('Lookback time [Gyr]', y=0.05, fontsize=20)
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
    Zsun = 0.014

    print(1e-4/Zsun)

    #data_colors = [plt.cm.GnBu(0.8), plt.cm.PuRd(0.6), plt.cm.YlGn(0.4), 'tab:red', 'tab:purple', 'tab:brown']
    #fit_colors = ['midnightblue', plt.cm.PuRd(0.9), plt.cm.YlGn(0.8), 'darkred', 'darkpurple', 'darkbrown']
    #line_colors = [plt.cm.GnBu(0.99), plt.cm.PuRd(0.7), plt.cm.YlGn(0.5), 'tab:red', 'tab:purple', 'tab:brown']
    cmap_blue = matplotlib.colors.LinearSegmentedColormap.from_list("blue_cmap", ['#40E9E0', '#1C7EB7', '#100045'])
    cmap_pink = matplotlib.colors.LinearSegmentedColormap.from_list("pink_cmap", ['#F76FDD', '#C13277', '#490013'])
    cmap_green = matplotlib.colors.LinearSegmentedColormap.from_list("green_cmap", ['#CCE666', '#79B41C', '#004011'])
    cmap_gray = matplotlib.colors.LinearSegmentedColormap.from_list("gray_cmap", ['#D3D2D2', '#787878', '#222222'])
    data_colors = ["#0067A6", '#C01874', '#98CB4F', '#D3D2D2']
    fit_colors = ['#0C0034', '#4B0012', '#005B2F', '#787878']
    line_styles = ['solid', 'dashed', 'dotted']

    #compare_params(tngs, vers)
    compare_SFR(path, tngs, vers, plotmodel=True, plotredshift=True, xlim=[0, 14], ylim=[10**-3, 10**-0.8], error_ylim = [5e-1, 1e1], plotlogscale=True)

    compare_Zdist(path, tngs, vers, xlim=[10**-4, 10], ylim=[10**-3, 4], error_ylim=[1e-1, 1e3], plotmodel=True, plotlogscale=True)

    #SFR_residuals(path, tngs, vers, plotredshift=True, xlim=[0, 14], ylim=[1e-2, 1e3], plotlogscale=True, show_MD17=False)
    #Zdist_residuals(path, tngs, vers, xlim=[10**-4, 10], ylim=[1e-2, 1e3], plotlogscale=True)
    #SFR_residuals(path, tngs, vers, plotredshift=False, xlim=[0, 14], ylim=[1e-2, 1e3], plotlogscale=True, show_MD17=False)
    #compare_dPdlogZ(path, tngs, vers, ylim=[1e-2, 1e1], levels = [0, 0.55], nlevels=30)