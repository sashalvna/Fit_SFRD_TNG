import os
import h5py as h5
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import argparse

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

    """
    Only use if downloading data to your local machine or cluster, not in jupyterlab

    Downloads a single snapshot subfile; requires path to local TNG data files, which TNG and which level (e.g TNG100-"1"),
    Which snapshot, which subfile in snapshot, and which data columns from snapshot
    """

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

    """
    Only use if downloading data to your local machine or cluster, not in jupyterlab

    deletes a single snapshot; requires requires path to local TNG data files,
    which snapshot, which file in snapshot, and which data columns from snapshot
    """

    #check if snapshot subfile exists; if yes, delete
    fname = TNGpath + "/%03d/snap_%03d.%s.hdf5" %(snap, snap, ifile)
    if os.path.exists(fname):
        os.remove(fname)

def get_z_and_lookbacktime(snaps):

    """
    Get the redshifts for each snapshot
    """
    
    redshifts = np.array([])
    for ind, val in enumerate(snaps):
        redshifts = np.append(redshifts, snaps[ind]['redshift'])
    
    #calculate lookback time for each snapshot
    lookbacktimes = np.array(cosmo.lookback_time(redshifts))
    
    return redshifts, lookbacktimes

def getandbinSFRandZdata(filename, ifile, mbins):

    with h5.File(filename % ifile, "r") as f:
        pStars = f["PartType0"]
        SFR = pStars["StarFormationRate"][:]
        Metals = pStars["GFM_Metallicity"][:]
      
        data, e = np.histogram(Metals, bins=mbins, weights=SFR)
        metaldata, e = np.histogram(Metals, bins=mbins)

    return data, metaldata
            
def getSFRMetallicityFromGas(TNGpath, TNG, lvl, snap, nfiles, jupyterlab=True, nBinsSFR=60):

    """
    Get SFR for each snapshot and bin it into metallicity bins
    To do this, reads data from each subfile in the snapshot and appends all the data in the bins
    """

    mbins = np.logspace(-10, 0., nBinsSFR+1)
    sfrs  = np.zeros((nBinsSFR))
    metals = np.zeros((nBinsSFR))

    if jupyterlab == True:
        fname =  TNGpath + "/snapdir_%03d/snap_%03d.%s.hdf5"%(snap, snap, "%d") 
    
        for ifile in range(nfiles):
            data, metaldata = getandbinSFRandZdata(fname, ifile, mbins)
            sfrs += data
            metals += metaldata

    else: #if running on your local machine (or cluster) and having to download files, be more careful to make sure files actually exist
        fname =  TNGpath + "/%03d/snap_%03d.%s.hdf5"%(snap, snap, "%d")
        get_TNGsnapcols(TNGpath, TNG, lvl, snap, nfiles)
        
        for ifile in range(nfiles):
            if os.path.exists(fname % ifile):
                data, metaldata = getandbinSFRandZdata(fname, ifile, mbins)
                sfrs += data
                metals += metaldata
                delete_snapshotsubfile(TNGpath, snap, ifile) #delete subfile after getting data from it

            else: #if file is missing, try re downloading that file
                get_TNGsnapcols(TNGpath, TNG, lvl, snap, ifile, singlefile=True)
            
                if os.path.exists(fname % ifile): #if re downloading successful, read the new file data
                    data, metaldata = getandbinSFRandZdata(fname, ifile, mbins)
                    sfrs += data
                    metals += metaldata
                    delete_snapshotsubfile(TNGpath, snap, ifile) 

                else: #if file still doesn't exist, just skip it
                    continue

    return sfrs, metals

def getFullSFRMetallicityFromGas(TNGpath, outfname, TNG, lvl, snaps, nfiles, starting_snap, jupyterlab=True, nbins=60):

    """
    Make data file and get data from all snapshots
    """
    if jupyterlab==True:
        outfname = outfname % (TNG,lvl)
    else:
        outfname = TNGpath + "/" + outfname % (TNG,lvl)

    print("Output will be stored in file: ", outfname)
        
    sfrs = np.zeros((len(snaps),nbins))
    metals = np.zeros((len(snaps),nbins))
    redshifts, lookbacktimes = get_z_and_lookbacktime(snaps)

    for snap in range(starting_snap, len(snaps)):

        if sfrs[snap].sum() == 0:
            print( "Doing snap %d." % snap )
            s, m = getSFRMetallicityFromGas(TNGpath, TNG, lvl, snap, nfiles, nBinsSFR=nbins, jupyterlab=jupyterlab)
            sfrs[snap,:] = s
            metals[snap,:] = m

        if snap==0: #if starting from beginning, assume no file exists to append into
            mbins = np.logspace( -10, 0., nbins+1 )
            if os.path.exists(outfname): #check if file with same name exists; if not append, does not overwrite it, makes a new separate file
                outfname = outfname + "-1"
            with h5.File(outfname, "w") as f:
                f.create_dataset('MetalBins', data=mbins )
                f.create_dataset('Redshifts', data=redshifts )
                f.create_dataset('Lookbacktimes', data=lookbacktimes )
                f.create_dataset('Sfr', data=sfrs)
                f.create_dataset('Metals', data=metals)

        else:
            f = h5.File(outfname, 'r+') # open existing file
            data = f['Sfr']       # load existing sfr data; assuming other cols already written
            data[snap] = sfrs[snap] #write new snapshot data to it; overwrites anything already in that column
            metal_data = f['Metals']   # load existing metallicity data; assuming other cols already written
            metal_data[snap] = metals[snap] #write new snapshot data to it; overwrites anything already in that column
            f.close()


if __name__ == "__main__":

    # Define command line options for the most commonly varied options
    parser = argparse.ArgumentParser()

    parser.add_argument("--tng", dest= 'TNG',  help="Which TNG simulation (e.g. 50, 100, 300)",type=int, default=100)
    parser.add_argument("--level", dest= 'lvl',  help="Which level simulation (e.g. TNG100-'1', 2, 3)",type=int, default=1)
    parser.add_argument("--jupyterlab", dest= 'jupyterlab',  help="Run on TNG jupyterlab", default=False, action='store_true')
    parser.add_argument("--apikey", dest= 'api_key',  help="TNG api key (if you are not Sasha, change it!!)",type=str, default="e36226423a0cc5e62f2e553f39b44238")
    parser.add_argument("--bins", dest= 'nbins',  help="Number of metallicity bins for binning data",type=int, default=60)
    parser.add_argument("--starting_snap", dest= 'ssnap',  help="Option to start from a specific snapshot (useful if doing data download in multiple runs, or redoing a specific snapshot)",type=int, default=0)
    parser.add_argument("--outfile", dest= 'outfname',  help="Name of output file",type=str, default="SFRMetallicityFromGasWithMetalsTNG%d-%d.hdf5")
    parser.add_argument("--nfiles", dest= 'nfiles',  help="Number of subfiles per snapshot to read; use for testing, otherwise will be missing data",type=int, default=0)
    
    args = parser.parse_args()

    #path to TNG data   
    baseUrl = 'http://www.tng-project.org/api/'
    headers = {"api-key": args.api_key} #my api key from TNG website log in
    if args.jupyterlab==True:
        TNGpath = "/home/tnguser/sims.TNG/TNG%s-%s/output" % (args.TNG, args.lvl)
    else:
        TNGpath = str(paths.tng) + "/TNG%s" % args.TNG

    # Get basic snapshot information (redshift, url) for each of the 100 snapshots
    r = get(baseUrl) #path to all Illustris TNG simulations
    names = [sim['name'] for sim in r['simulations']] #names of all available simulations
    i = names.index('TNG%s-%s' % (args.TNG, args.lvl)) #index of specific simulation
    sim = get( r['simulations'][i]['url'] ) #get path to that simulation
    snaps = get( sim['snapshots'] ) #get path to snapshots for that simulation

    #Number of subfiles per snapshot; usually just leave as the default to get them all (otherwise data WILL be missing)
    #For testing, can set a small number, recommended: 6
    if args.nfiles==0:
        nfiles = sim['num_files_snapshot'] #get number subfiles per snapshot
    else:
        nfiles=args.nfiles

    # Run data download
    getFullSFRMetallicityFromGas(TNGpath, args.outfname, args.TNG, args.lvl, snaps, nfiles, starting_snap=args.ssnap, jupyterlab=args.jupyterlab, nbins=args.nbins)
