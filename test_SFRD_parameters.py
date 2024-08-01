import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

############################
# Custom scripts
import get_ZdepSFRD as Z_SFRD
import paths
from Fit_model_TNG_SFRD import readTNGdata, interpolate_TNGdata
from TNG_BBHpop_properties import read_best_fits

def binned_dPdlogZ(tng, ver, redshift_list, center_Zbin, fit_params, plotdefaults=True, plotparamvariations=True, plotsystematicparamvariations=True, showplots=True):

    mu0, muz, omega0, omegaz, alpha0, sf_a, sf_b, sf_c, sf_d = fit_params
    Zsun = 0.014

    #get dP/dlogZ given fiducial (best fit) values
    dPdlogZ, metallicities, step_logZ, p_draw_metallicity = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                mu_0 =mu0, omega_0=omega0,omega_z=omegaz, alpha =alpha0, metals=center_Zbin)

    #create mask of low/middle/high redshift values
    lowz = (redshift_list < 2)
    midz = ((redshift_list >= 2) & (redshift_list < 4))
    highz = ((redshift_list >= 4) & (redshift_list < 6))

    #binned dP/dlogZ by redshift
    lowz_dPdlogZ = dPdlogZ[:,][lowz]
    midz_dPdlogZ = dPdlogZ[:,][midz]
    highz_dPdlogZ = dPdlogZ[:,][highz]

    if plotdefaults==True:
        #Plot mean dP/dlogZ vs Z for the fiducial TNG100 parameters

        plt.plot(metallicities/Zsun, np.mean(lowz_dPdlogZ, axis=0), label="z = 0-2")
        plt.plot(metallicities/Zsun, np.mean(midz_dPdlogZ, axis=0), label="z = 2-4")
        plt.plot(metallicities/Zsun, np.mean(highz_dPdlogZ, axis=0), label="z = 4-6")
        plt.xlabel(r"$Z/Z_\odot$")
        plt.ylabel("Mean " + r"$dP/d\logZ$")
        plt.xscale("log")
        plt.legend()
        plt.title('Default TNG100 parameters:'+'\n'+r'$\mu_O$=%.2f, $\mu_z$=%.2f, $\omega_0$=%.2f, $\omega_z$=%.2f, $\alpha$=%.2f'%(mu0, muz, omega0, omegaz, alpha0));
    
        plt.savefig('figures/checking_fit/parameter_default_TNG%s-%s.png'%(tng, ver), bbox_inches='tight')

        if showplots==True:
            plt.show()

    #Get dP/dlogZ for each variation of the parameters (changing both alpha and omega_z), and bin them

    #low skewness, high redshift evol of variance
    dPdlogZ1, metallicities1, step_logZ1, p_draw_metallicity1 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                        mu_0 =mu0, omega_0=omega0,omega_z=0.1, alpha =0, metals=center_Zbin)

    #high skewness, low redshift evol of variance
    dPdlogZ2, metallicities2, step_logZ2, p_draw_metallicity2 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                        mu_0 =mu0, omega_0=omega0,omega_z=0, alpha =-6, metals=center_Zbin)

    lowz_dPdlogZ1 = dPdlogZ1[:,][lowz]
    midz_dPdlogZ1 = dPdlogZ1[:,][midz]
    highz_dPdlogZ1 = dPdlogZ1[:,][highz]

    lowz_dPdlogZ2 = dPdlogZ2[:,][lowz]
    midz_dPdlogZ2 = dPdlogZ2[:,][midz]
    highz_dPdlogZ2 = dPdlogZ2[:,][highz]

    if plotparamvariations==True:
        #Plot the parameter variations for high (top) and low (bottom) resolution

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ1, axis=0), label="z = 0-2")
        ax[0].plot(metallicities/Zsun, np.mean(midz_dPdlogZ1, axis=0), label="z = 2-4")
        ax[0].plot(metallicities/Zsun, np.mean(highz_dPdlogZ1, axis=0), label="z = 4-6")
        ax[0].set_xlabel(r"$Z/Z_\odot$", fontsize=15)
        ax[0].set_ylabel("Mean " + r"$dP/d\logZ$", fontsize=15)
        ax[0].set_xscale("log")
        ax[0].legend(fontsize=12)
        ax[0].set_title('High redshift evolution of variance/low skewness:'+'\n'+r'$\mu_O$=%.2f, $\mu_z$=%.2f, $\omega_0$=%.2f, $\omega_z$=%.2f, $\alpha$=%.2f'%(mu0, muz, omega0, 0.1, 0), fontsize=15);

        ax[1].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ2, axis=0), label="z = 0-2")
        ax[1].plot(metallicities/Zsun, np.mean(midz_dPdlogZ2, axis=0), label="z = 2-4")
        ax[1].plot(metallicities/Zsun, np.mean(highz_dPdlogZ2, axis=0), label="z = 4-6")
        ax[1].set_xlabel(r"$Z/Z_\odot$", fontsize=15)
        ax[1].set_ylabel("Mean " + r"$dP/d\logZ$", fontsize=15)
        ax[1].set_xscale("log")
        ax[1].legend(fontsize=12)
        ax[1].set_title('Low redshift evolution of variance/high skewness:'+'\n'+r'$\mu_O$=%.2f, $\mu_z$=%.2f, $\omega_0$=%.2f, $\omega_z$=%.2f, $\alpha$=%.2f'%(mu0, muz, omega0, 0,-6), fontsize=15);

        fig.tight_layout()
        fig.savefig('figures/checking_fit/parameter_variations_resolutions_TNG%s-%s.png'%(tng, ver), bbox_inches='tight')

        if showplots==True:
            plt.show()

    if plotsystematicparamvariations==True:
        ##Get dP/dlogZ for each variation of the parameters (one at a time), and bin them
        #high omega_z
        dPdlogZ3, metallicities3, step_logZ3, p_draw_metallicity3 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                                mu_0 =mu0, omega_0=omega0,omega_z=0.1, alpha =alpha0, metals=center_Zbin)
        #low omega_z
        dPdlogZ4, metallicities4, step_logZ4, p_draw_metallicity4 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                                mu_0 =mu0, omega_0=omega0,omega_z=0, alpha =alpha0, metals=center_Zbin)
        #high alpha
        dPdlogZ5, metallicities5, step_logZ5, p_draw_metallicity5 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                                mu_0 =mu0, omega_0=omega0,omega_z=omegaz, alpha =-6, metals=center_Zbin)
        #low alpha
        dPdlogZ6, metallicities6, step_logZ6, p_draw_metallicity6 = Z_SFRD.skew_metallicity_distribution(redshift_list, mu_z =muz, 
                                                                mu_0 =mu0, omega_0=omega0,omega_z=omegaz, alpha =0, metals=center_Zbin)

        lowz_dPdlogZ3 = dPdlogZ3[:,][lowz]
        midz_dPdlogZ3 = dPdlogZ3[:,][midz]
        highz_dPdlogZ3 = dPdlogZ3[:,][highz]

        lowz_dPdlogZ4 = dPdlogZ4[:,][lowz]
        midz_dPdlogZ4 = dPdlogZ4[:,][midz]
        highz_dPdlogZ4 = dPdlogZ4[:,][highz]

        lowz_dPdlogZ5 = dPdlogZ5[:,][lowz]
        midz_dPdlogZ5 = dPdlogZ5[:,][midz]
        highz_dPdlogZ5 = dPdlogZ5[:,][highz]

        lowz_dPdlogZ6 = dPdlogZ6[:,][lowz]
        midz_dPdlogZ6 = dPdlogZ6[:,][midz]
        highz_dPdlogZ6 = dPdlogZ6[:,][highz]

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[0].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ3, axis=0), c='tab:blue', linestyle='-', label=r"z = 0-2, $\omega_z$=%.2f, $\alpha$=%.2f"%(0.1, alpha0))
        ax[0].plot(metallicities/Zsun, np.mean(midz_dPdlogZ3, axis=0), c='tab:blue', linestyle='--', label="z = 2-4")
        ax[0].plot(metallicities/Zsun, np.mean(highz_dPdlogZ3, axis=0), c='tab:blue', linestyle=':', label="z = 4-6")
        ax[0].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ6, axis=0), c='tab:orange', linestyle='-', label=r"z = 0-2, $\omega_z$=%.2f, $\alpha$=%.2f"%(omegaz, 0))
        ax[0].plot(metallicities/Zsun, np.mean(midz_dPdlogZ6, axis=0), c='tab:orange', linestyle='--', label="z = 2-4")
        ax[0].plot(metallicities/Zsun, np.mean(highz_dPdlogZ6, axis=0), c='tab:orange', linestyle=':', label="z = 4-6")
        ax[0].set_xlabel(r"$Z/Z_\odot$", fontsize=15)
        ax[0].set_ylabel("Mean " + r"$dP/d\logZ$", fontsize=15)
        ax[0].set_xscale("log")
        ax[0].legend(fontsize=12)
        ax[0].set_title('Large redshift evolution of variance/low skewness:'+'\n'+r'$\mu_O$=%.2f, $\mu_z$=%.2f, $\omega_0$=%.2f'%(mu0, muz, omega0), fontsize=15);

        ax[1].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ4, axis=0), c='tab:blue', linestyle='-', label=r"z = 0-2, $\omega_z$=%.2f, $\alpha$=%.2f"%(0, alpha0))
        ax[1].plot(metallicities/Zsun, np.mean(midz_dPdlogZ4, axis=0), c='tab:blue', linestyle='--', label="z = 2-4")
        ax[1].plot(metallicities/Zsun, np.mean(highz_dPdlogZ4, axis=0), c='tab:blue', linestyle=':', label="z = 4-6")
        ax[1].plot(metallicities/Zsun, np.mean(lowz_dPdlogZ5, axis=0), c='tab:orange', linestyle='-', label=r"z = 0-2, $\omega_z$=%.2f, $\alpha$=%.2f"%(omegaz, -6))
        ax[1].plot(metallicities/Zsun, np.mean(midz_dPdlogZ5, axis=0), c='tab:orange', linestyle='--', label="z = 2-4")
        ax[1].plot(metallicities/Zsun, np.mean(highz_dPdlogZ5, axis=0), c='tab:orange', linestyle=':', label="z = 4-6")
        ax[1].set_xlabel(r"$Z/Z_\odot$", fontsize=15)
        ax[1].set_ylabel("Mean " + r"$dP/d\logZ$", fontsize=15)
        ax[1].set_xscale("log")
        ax[1].legend(fontsize=12)
        ax[1].set_title('Small redshift evolution of variance/high skewness:'+'\n'+r'$\mu_O$=%.2f, $\mu_z$=%.2f, $\omega_0$=%.2f'%(mu0, muz, omega0), fontsize=15);

        fig.tight_layout()
        fig.savefig('figures/checking_fit/parameter_variations_binned_TNG%s-%s.png'%(tng, ver), bbox_inches='tight')

        if showplots==True:
            plt.show()

def model(theta, redshifts, metals):
    mu0, muz, omega0, omegaz, alpha, sf_a, sf_b, sf_c, sf_d = theta
    
    dPdlogZ, metallicities, step_logZ, p_draw_metallicity = Z_SFRD.skew_metallicity_distribution(redshifts, mu_z =muz, 
                                                        mu_0 =mu0, omega_0=omega0,omega_z=omegaz, alpha =alpha, metals=metals)
    sfr = Z_SFRD.Madau_Dickinson2014(redshifts, a=sf_a, b=sf_b, c=sf_c,  d=sf_d)
    
    return (sfr* (dPdlogZ * step_logZ ).T).value

def lnlike(theta, redshifts, metals, SFRD):
    #first calculate the value of the model at the current parameters
    modelvals = model(theta, redshifts, metals)
    
    #then the (log) likelihood function is a sum of Gaussians with a given variance
    sigma = 0.1 #abs(modelvals-SFRD) #0.1
    sigma2 = sigma**2
    lp = -0.5*np.sum(((SFRD-modelvals)**2)/sigma2 + np.log(2*np.pi*sigma2))
    if np.isnan(lp):
        lp = -np.inf
    return lp
    
def lnprior(theta):
    mu0, muz, omega0, omegaz, alpha, sf_a, sf_b, sf_c, sf_d = theta
    
    if 0.001 < mu0 < 0.1 and -1.0 < muz < 0.0 and 0.01 < omega0 < 5.0 and 0.0 < omegaz < 1.0 and -10.0 < alpha < 0.0: #and 0.01 < sf_a < 0.1: #and 0 < sf_b < 10 and 0 < sf_c < 10 and 0 < sf_d < 10:
        return 0.0
    return -np.inf

def lnprob(theta, redshifts, metals, SFRD):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, redshifts, metals, SFRD)

def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

def mcmc_SFRDparams(tng, ver, redshift_new, metals_new, SFRDnew, fit_params, plotsamples=True, plotcorner=True, plotbestfit=True, showplots=False, nwalkers=100, niter=1000):

    Zsun = 0.014
    data = (redshift_new, metals_new, SFRDnew)
    initial = fit_params
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)] #1e-2 

    sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
    sampler.flatchain[np.argmax(sampler.flatlnprobability)]

    if plotsamples==True:
        labels = ['mu0','muz','omega0','omegaz','alpha','a','b','c','d']
        fig, axes = plt.subplots(9, figsize=(10, 20), sharex=True)
        samples = sampler.get_chain()
        for i in range(9):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        fig.savefig("figures/mcmcparamsamples_TNG%s-%s.png"%(tng, ver))

        if showplots==True:
            plt.show()

    if plotcorner == True:
        fig = corner.corner(sampler.flatchain,show_titles=True, labels=labels, truths=fit_params)
        fig.savefig("figures/cornerplot_TNG%s-%s.png"%(tng, ver))

        if showplots==True:
            plt.show()


    if plotbestfit == True:
        new_theta_max  = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
        new_best_fit_model = model(new_theta_max, redshift_new, metals_new)

        fig, ax = plt.subplots(figsize = (14, 10))

        if tng==50:
            clevels = [1e-10, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05] #tng50
        elif tng==100:
            clevels = [1e-10, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.03, 0.034] #tng100
        else:
            clevels = [1e-10, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.023] #tng300

        dataplot = ax.contourf(Lookbacktimes_new, metals_new/Zsun, SFRDnew)
        modelplot = ax.contour(Lookbacktimes_new, metals_new/Zsun, new_best_fit_model, levels = clevels, colors='white')
        ax.clabel(modelplot, fontsize=10, inline=True)

        ax.set_ylim(10**-1, 10**1)
        ax.set_yscale('log')
        ax.set_xlabel('Lookback time', fontsize=20)
        ax.set_ylabel(r'$\mathrm{Metallicity}, \ Z/Z_{\rm{\odot}}$', fontsize=20)
        cbar = fig.colorbar(dataplot)
        cbar.set_label('SFRD(Z, z)', rotation=270,fontsize=20, labelpad=30);

        print('Theta max: ',new_theta_max)
        fig.savefig("figures/mcmcfit_TNG%s-%s.png"%(tng, ver))

        if showplots==True:
            plt.show()


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    tng= 50
    ver = 1
    Cosmol_sim_location = paths.data / str("SFRMetallicityFromGasTNG%s-%s.hdf5"%(tng,ver))
    fit_filename = ['test_best_fit_parameters_TNG%s-%s.txt'%(tng,ver)]
    if tng==50:
        rbox=35
    elif tng==100:
        rbox=75
    elif tng==300:
        rbox=205

    #Read TNG data and interpolate it
    Sim_SFRD, Lookbacktimes, Redshifts, Sim_center_Zbin, step_fit_logZ = readTNGdata(loc = Cosmol_sim_location, rbox=rbox)
    SFRDnew, redshift_new, Lookbacktimes_new, metals_new = interpolate_TNGdata(Redshifts, Lookbacktimes, Sim_SFRD, Sim_center_Zbin, tng, ver, saveplot=False)
    
    #Get model parameters
    fit_params = read_best_fits(fit_filename)[0]
    
    #dPdlogZ parameter variations
    redshift_list  = np.linspace(0,15, num=100)
    metal_bins = np.logspace(-10, 0, 61) #60 bins
    center_Zbin = (metal_bins[:-1] + metal_bins[1:])/2
    binned_dPdlogZ(tng, ver, redshift_list, center_Zbin, fit_params, plotdefaults=True, plotparamvariations=True, plotsystematicparamvariations=True, showplots=True)

    #Run MCMC
    mcmc_SFRDparams(tng, ver, redshift_new, metals_new, SFRDnew, fit_params, showplots=True, plotsamples=True, plotcorner=True, plotbestfit=True, nwalkers=100, niter=1000)