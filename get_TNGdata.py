import sys
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import illustris_python as il
import argparse

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


def get_TNGsnapcols(TNGpath, TNG, lvl, snap, nfiles, singlefile=False, cols='gas=GFM_Metallicity,StarFormationRate'):
    #downloads a single snapshot subfile; requires path to local TNG data files, which TNG and which level (e.g TNG100-"1"),
    #which snapshot, which subfile in snapshot, and which data columns from snapshot

    #check if snapshot subfile already exists; if not, download
    if not os.path.exists(TNGpath + "/%03d" %(snap)):
        os.chdir(TNGpath)
        os.mkdir("%03d" %(snap))

    os.chdir(TNGpath + "/%03d" %(snap)) #go to correct directory before downloading

    if singlefile==True: #if only downloading a single file
        fname = TNGpath + "/%03d/snap_%03d.%s.hdf5" %(snap, snap, nfiles)
        os.system("wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header='API-Key: e36226423a0cc5e62f2e553f39b44238' 'http://www.tng-project.org/api/TNG%s-%s/files/snapshot-%03d.%s.hdf5?%s'" %(TNG, lvl, snap, nfiles, cols))

    else:
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

        if os.path.exists(fname % ifile):
            with h5.File(fname % ifile, "r") as f:

                pStars = f["PartType0"]

                SFR = pStars["StarFormationRate"][:]
                Metals = pStars["GFM_Metallicity"][:]

                data, e = np.histogram(Metals, bins=mbins, weights=SFR)
                sfrs += data

                delete_snapshotsubfile(TNGpath, snap, ifile) #delete subfile after getting data from it

        else: #if file is missing, try re downloading that file
            get_TNGsnapcols(TNGpath, TNG, lvl, snap, ifile, singlefile=True)
            
            if os.path.exists(fname % ifile): #if re downloading successful, get the new file
                with h5.File(fname % ifile, "r") as f:
                    pStars = f["PartType0"]
                    SFR = pStars["StarFormationRate"][:]
                    Metals = pStars["GFM_Metallicity"][:]
                    data, e = np.histogram(Metals, bins=mbins, weights=SFR)
                    sfrs += data
                    delete_snapshotsubfile(TNGpath, snap, ifile)
            else: #if file still doesn't exist, just skip it
                continue


    return sfrs

#function to make the data file; iterates through all snapshots

def getFullSFRMetallicityFromGas(TNGpath, TNG, lvl, snaps, nfiles, starting_snap, nbins=60):
    outfname = TNGpath + "/SFRMetallicityFromGasTNG%d-%d.hdf5" % (TNG,lvl)
    sfrs = np.zeros((len(snaps),nbins))
    redshifts, lookbacktimes = get_z_and_lookbacktime(snaps)

    for snap in range(starting_snap, len(snaps)):
        if sfrs[snap].sum() == 0:
            print( "Doing snap %d." % snap )
            s = getSFRMetallicityFromGas(TNGpath, TNG, lvl, snap, nfiles, nBinsSFR=nbins)
            sfrs[snap,:] = s

        if snap==0: #if starting from beginning, assume no file exists to append into
            mbins = np.logspace( -10, 0., nbins+1 )
            if os.path.exists(outfname): #check if file with same name exists; if not append, does not overwrite it, makes a new separate file
                outfname = outfname + "-1"
            with h5.File(outfname, "w") as f:
                f.create_dataset('MetalBins', data=mbins )
                f.create_dataset('Redshifts', data=redshifts )
                f.create_dataset('Lookbacktimes', data=lookbacktimes )
                f.create_dataset('Sfr', data=sfrs)

        else:
            f = h5.File(outfname, 'r+') # open existing file
            data = f['Sfr']       # load existing sfr data; assuming other cols already written
            data[snap] = sfrs[snap] #write new snapshot data to it; overwrites anything already in that column
            f.close()


if __name__ == "__main__":

    # Define command line options for the most commonly varied options
    parser = argparse.ArgumentParser()

    parser.add_argument("--tng", dest= 'TNG',  help="Which TNG simulation (e.g. 50, 100, 300)",type=int, default=100)
    parser.add_argument("--level", dest= 'lvl',  help="Which level simulation (e.g. TNG100-'1', 2, 3)",type=int, default=1)
    parser.add_argument("--bins", dest= 'nbins',  help="Number of metallicity bins for binning data",type=int, default=60)
    parser.add_argument("--starting_snap", dest= 'ssnap',  help="Option to start from a specific snapshot (useful if doing data download in multiple runs, or redoing a specific snapshot)",type=int, default=0)

    args = parser.parse_args()

    TNGpath = str(paths.tng) + "/TNG%s" % args.TNG
    baseUrl = 'http://www.tng-project.org/api/' 
    headers = {"api-key":"e36226423a0cc5e62f2e553f39b44238"} #my api key from TNG website log in

    # Get basic snapshot information (redshift, url) for each of the 100 snapshots

    r = get(baseUrl) #path to all Illustris TNG simulations
    names = [sim['name'] for sim in r['simulations']] #names of all available simulations
    i = names.index('TNG%s-%s' % (args.TNG, args.lvl)) #index of specific simulation
    sim = get( r['simulations'][i]['url'] ) #get path to that simulation
    snaps = get( sim['snapshots'] ) #get path to snapshots for that simulation
    nfiles = sim['num_files_snapshot'] #number files per snapshot

    # Run data download
    #nfiles = 6 #TESTING with less files in snapshot
    getFullSFRMetallicityFromGas(TNGpath, args.TNG, args.lvl, snaps, nfiles, starting_snap=args.ssnap, nbins=args.nbins) #run data download
