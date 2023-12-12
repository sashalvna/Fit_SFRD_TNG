import sys
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import illustris_python as il

import get_ZdepSFRD as Z_SFRD
import importlib
import paths
import requests

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r


def get_TNGsnapcols(TNGpath, TNG, lvl, snap, nfiles, cols='gas=GFM_Metallicity,StarFormationRate'):
    #downloads a single snapshot subfile; requires path to local TNG data files, which TNG and which level (e.g TNG100-"1"),
    #which snapshot, which subfile in snapshot, and which data columns from snapshot

    #check if snapshot subfile already exists; if not, download
    if not os.path.exists(TNGpath + "/%03d" %(snap)):
        os.chdir(TNGpath)
        os.mkdir("%03d" %(snap))

    os.chdir(TNGpath + "/%03d" %(snap)) #go to correct directory before downloading
    for ifile in range(nfiles):
        fname = TNGpath + "/%03d/snap_%03d.%s.hdf5" %(snap, snap, ifile)
        if not os.path.isfile(fname):
            os.system("wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header='API-Key: e36226423a0cc5e62f2e553f39b44238' 'http://www.tng-project.org/api/TNG%s-%s/files/snapshot-%03d.%s.hdf5?%s'" %(TNG, lvl, snap, ifile, cols))


def delete_snapshotsubfile(TNGpath, snap, ifile):
    #deletes a single snapshot; requires requires path to local TNG data files,
    #which snapshot, which file in snapshot, and which data columns from snapshot

    #check if snapshot subfile exists; if yes, delete
    fname = TNGpath + "/%03d/snap_%03d.%s.hdf5" %(snap, snap, ifile)
    if os.path.exists(fname):
        os.remove(fname)

def get_z_and_lookbacktime(snaps):
    #get the redshifts for each snapshot
    redshifts = np.array([])
    for ind, val in enumerate(snaps):
        redshifts = np.append(redshifts, snaps[ind]['redshift'])
    
    #calculate lookback time for each snapshot
    lookbacktimes = np.array(cosmo.lookback_time(redshifts))
    
    return redshifts, lookbacktimes

#function to get binned SFR for each snapshot
def getSFRMetallicityFromGas(TNGpath, TNG, lvl, snap, nfiles, nBinsSFR=60):
    mbins = np.logspace(-10, 0., nBinsSFR+1)
    sfrs  = np.zeros((nBinsSFR))

    fname =  TNGpath + "/%03d/snap_%03d.%s.hdf5"%(snap, snap, "%d")
    get_TNGsnapcols(TNGpath, TNG, lvl, snap, nfiles)

    for ifile in range(nfiles):

        with h5.File(fname % ifile, "r") as f:

            pStars = f["PartType0"]

            SFR = pStars["StarFormationRate"][:]
            Metals = pStars["GFM_Metallicity"][:]

            data, e = np.histogram(Metals, bins=mbins, weights=SFR)
            sfrs += data

            delete_snapshotsubfile(TNGpath, snap, ifile) #delete subfile after getting data from it

    return sfrs

#function to make the data file; iterates through all snapshots

def getFullSFRMetallicityFromGas(TNGpath, TNG, lvl, snaps, nfiles, nbins=60):
    outfname = TNGpath + "/SFRMetallicityFromGasTNG%d-%d.hdf5" % (TNG,lvl)
    sfrs = np.zeros((len(snaps),nbins))
    redshifts, lookbacktimes = get_z_and_lookbacktime(snaps)

    Count = 0
    for snap in range(2): #range(len(snaps)):
        if sfrs[snap].sum() == 0:
            print( "Doing snap %d." % snap )
            s = getSFRMetallicityFromGas(TNGpath, TNG, lvl, snap, nfiles, nBinsSFR=nbins)
            sfrs[snap,:] = s
            Count += 1

    if Count > 0:
        mbins = np.logspace( -10, 0., nbins+1 )
        with h5.File(outfname, "w") as f:
            f.create_dataset('MetalBins', data=mbins )
            f.create_dataset('Redshifts', data=redshifts )
            f.create_dataset('Lookbacktimes', data=lookbacktimes )
            f.create_dataset('Sfr', data=sfrs )


TNG  = 100
lvl = 1
TNGpath = "/home/sashalvna/research/TNGdata/TNG%s" % (TNG)
baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"e36226423a0cc5e62f2e553f39b44238"}

#get basic snapshots information (redshift, url) for each of the 100 snapshots

r = get(baseUrl) #path to all Illustris TNG simulations
names = [sim['name'] for sim in r['simulations']] #names of all available simulations
i = names.index('TNG%s-%s' % (TNG, lvl)) #index of specific simulation
sim = get( r['simulations'][i]['url'] ) #get path to that simulation
snaps = get( sim['snapshots'] ) #get path to snapshots for that simulation
nfiles = sim['num_files_snapshot'] #number files per snapshot

nfiles_temp = 6
getFullSFRMetallicityFromGas(TNGpath, TNG, lvl, snaps, nfiles_temp, nbins=60)
