import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import seaborn as sns
import copy
from astropy.cosmology import Planck15  as cosmo# Planck 2015 since that's what TNG uses

# Custom scripts
sys.path.append('../')
import get_ZdepSFRD as Z_SFRD
import importlib
import paths
from TNG_BBHpop_properties import read_best_fits

import ReadFitData as read
importlib.reload(read)

def compare_params(tngs=[50, 100, 300], vers=[1, 1, 1], showplot=True):
    
    fit_param_files = []
    for n, tng in enumerate(tngs):
        print(tng, vers[n])
        fit_param_files.append('test_best_fit_parameters_TNG%s-%s.txt'%(tng, vers[n]))
    fit_params = read_best_fits(fit_param_files)

    ticks =  ["mean metallicity at z=0", "redshift evolution of mean", "variance in metallicity density distribution", 
              "redshift evolution of variance", "skewness","SFR scaling", "SFR upward slope", "SFR peak location", "SFR downward slope"]
    x = [0,1,2,3,4,5,6,7,8] #9 parameters

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    for n, fits in enumerate(fit_params):
        print(fits)

        ax[0].plot(x, fits, label="TNG%s-%s"%(tngs[n], vers[n]))

        if n != 1:
            ax[1].plot(x, fit_params[n]/fit_params[1], label='TNG%s-%s/TNG%s-%s'%(tngs[n], vers[n], tngs[1], vers[1]))
    
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ticks, rotation=90, fontsize=13)
    ax[0].legend(fontsize=13)
    ax[0].set_title('SFRD fit parameters')

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ticks, rotation=90, fontsize=13)
    ax[1].legend(fontsize=13)
    ax[1].set_title('Percent error in SFRD fit parameters');

    fig.savefig('figures/fitparams_percenterror.png', bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    #Change file names to match TNG version <- turn these into arguments
    tngs=[50, 100, 300, 100] 
    vers = [1, 1, 1, 2]

    compare_params(tngs, vers)