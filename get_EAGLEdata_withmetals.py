import os
import h5py as h5
import numpy as np
from astropy.cosmology import Planck13 as cosmo
import argparse
from pathlib import Path
import paths
import requests
import shutil
import tarfile

#

# def read_EAGLEdataset(EAGLEpath, itype, att, zp, nfiles = 16):
#     """ Read a selected dataset, itype is the PartType and att is the attribute name """
#     # Output array
#     data = []
#     # Loop over each file and extract the data
#     for i in range(nfiles):
#         f = h5.File(EAGLEpath + '/snapshot_%03d_%s'%(snap,zp) + '/snap_0%i_%s.0.hdf5'%(snap,zp), 'r')
#         tmp = f['PartType%i/%s'%(itype, att)][...]
#         data.append(tmp)
    
#         # Get conversion factors
#         cgs = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
#         aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
#         hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')
        
#         # Get expansion factor and Hubble parameter from the header
#         a = f['Header'].attrs.get('Time')
#         h = f['Header'].attrs.get('HubbleParam')

#         f.close()

#     # Combine to a single array
#     if len(tmp.shape) > 1:
#         data = np.vstack(data)
#     else:
#         data = np.concatenate(data)

#     # Convert to physical
#     if data.dtype != np.int32 and data.dtype != np.int64:
#         data = np.multiply(data, a**aexp * h**hexp, dtype='f8')
                            
#     return data

def read_dataset(EAGLE, itype, att, nfile, snap, zp):
    """ Read a selected dataset, itype is the PartType and att is the attribute name """

    # Output array
    f = h5.File(EAGLEpath + '/' + EAGLE + '/snapshot_%03d_%s'%(snap,zp) + '/snap_%03d_%s.%s.hdf5'%(snap,zp, nfile), 'r')
    data = f['PartType%i/%s'%(itype, att)][...]

    # Get conversion factors
    aexp = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
    hexp = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')

    # Get expansion factor and Hubble parameter from the header
    a = f['Header'].attrs.get('Time')
    h = f['Header'].attrs.get('HubbleParam')

    f.close()

    # Convert to physical
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, a**aexp * h**hexp, dtype='f8')

    return data


#

def get(EAGLEpath, params=None):

    # make HTTP GET request to path
    r = requests.get(EAGLEpath, params=params)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

#
    
def get_EAGLEsnapcols(EAGLEpath, EAGLE, snap): 
    """
    Only use if downloading data to your local machine or cluster, not in jupyterlab

    Downloads a single snapshot subfile; requires path to local TNG data files, which TNG and which level (e.g TNG100-"1"),
    Which snapshot, which subfile in snapshot, and which data columns from snapshot
    """
    os.chdir(EAGLEpath)
    file_path= Path(EAGLEpath + '/%s_snap_%03d.tar'%(EAGLE,snap))
    #check if snapshot subfile already exists; if not, download
    if not file_path.is_file():
        print('downloading...')
        print(file_path)
        # os.mkdir('/snap_0%i_z000p000'%snap)
        
   # os.chdir(EAGLEpath + '/snapshot_0%i_z000p000'%snap) #go to correct directory before downloading

    #if singlefile==True: #if only downloading a single file
        fname = baseUrl + EAGLE + '&snapnum=%03d'%snap
        os.system("wget --user=%s --password=%s --content-disposition '%s'" %(args.user, args.password, fname)) 
     
    with tarfile.open('%s_snap_%03d.tar'%(EAGLE,snap)) as tar:
        tar.extractall(path=EAGLEpath)

    path = EAGLE
    contents = os.listdir(path)
    zp = contents[0][-8:]
    os.chdir(EAGLEpath +'/'+ EAGLE + '/'+ contents[0])
    file_count = sum(1 for p in Path('.').iterdir() if p.is_file())
    print("Total files in this snapshot:", file_count)
    # else:
    #     for ifile in range(nfiles):
    #         fname = EAGLEpath + '/snap_028_z000p000.%i.hdf5'%i, 'r'(snap, snap, ifile)
    #         if not os.path.isfile(fname):
    #             os.system("wget --user=[username] --password=[password] --content-disposition 'https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots//download?run=RefL0012N0188&snapnum=28'" %(user, password, baseUrl, EAGLE, lvl, snap, ifile))
    return(file_count, zp)
    
#

def delete_snapshotsubfile(EAGLEpath, EAGLE):

    """
    Only use if downloading data to your local machine or cluster, not in jupyterlab

    deletes a single snapshot; requires requires path to local TNG data files,
    which snapshot, which file in snapshot, and which data columns from snapshot
    """

    #check if snapshot subfile exists; if yes, delete
    fname = EAGLEpath + '/' + EAGLE
    if os.path.exists(fname):
       shutil.rmtree(fname)


def getandbinSFRandZdata(EAGLE, nfile, mbins, snap, zp):

    itype = 0 #PartType0 = gas
    Z = read_dataset(EAGLE, itype, att='Metallicity', nfile=nfile, snap=snap, zp=zp) 
    SFR = read_dataset(EAGLE, itype, att='StarFormationRate', nfile=nfile, snap=snap, zp=zp) 

    data, e = np.histogram(Z, bins=mbins, weights=SFR)
    metaldata, e = np.histogram(Z, bins=mbins)

    return data, metaldata

#
    
def getSFRMetallicityFromGas(EAGLEpath, EAGLE, snap, nfiles, jupyterlab=True, nBinsSFR=60):

    """
    Get SFR for each snapshot and bin it into metallicity bins
    To do this, reads data from each subfile in the snapshot and appends all the data in the bins
    """

    mbins = np.logspace(-10, 0., nBinsSFR+1)
    sfrs  = np.zeros((nBinsSFR))
    metals = np.zeros((nBinsSFR))

    z, lookbacktime = 0, 0

    nfiles, zp = get_EAGLEsnapcols(EAGLEpath, EAGLE, snap)
    #fname = EAGLEpath +'/'+ EAGLE + "/snap_%03d_%s/snap_%03d_%s.%s.hdf5"%(snap, zp, snap, zp, "%d")

    for ifile in range(nfiles):
        #if os.path.exists(fname % ifile):
        data, metaldata = getandbinSFRandZdata(EAGLE, ifile, mbins, snap, zp)
        sfrs += data
        metals += metaldata
        z = read_header(EAGLE, snap, zp, ifile)
        lookbacktime = cosmo.lookback_time(z)

    delete_snapshotsubfile(EAGLEpath, EAGLE) #delete subfile after getting data from it
    return sfrs, metals, z, lookbacktime
    


#
    
def getFullSFRMetallicityFromGas(EAGLEpath, outfname, EAGLE, nfiles, nsnaps=28, starting_snap=0, jupyterlab=True, nbins=60):

    """
    Make data file and get data from all snapshots
    """
    outfname = EAGLEpath + "/" + outfname % (EAGLE)

    print("Output will be stored in file: ", outfname)
    
    sfrs = np.zeros((nsnaps,nbins))
    metals = np.zeros((nsnaps,nbins))    
    redshifts = np.zeros(nsnaps)
    lookbacktimes = np.zeros(nsnaps)
    
    for snap in range(starting_snap, nsnaps):

        if sfrs[snap].sum() == 0:
            print( "Doing snap %d." % snap )
            s, m, z, lookbacktime = getSFRMetallicityFromGas(EAGLEpath, EAGLE, snap, nfiles, jupyterlab= False, nBinsSFR= nbins)
            sfrs[snap,:] = s
            metals[snap,:] = m
            redshifts[snap] = z
            print(s)
            print(m)
            lookbacktimes[snap] = lookbacktime.value

        if snap==0: #if starting from beginning, assume no file exists to append into
            mbins = np.logspace( -10, 0., nbins+1 )
            if os.path.exists(outfname): #check if file with same name exists; if not append, does not overwrite it, makes a new separate file
                outfname = outfname + '-1'
            with h5.File(outfname, "w") as f:
                f.create_dataset('MetalBins', data=mbins )
                f.create_dataset('Redshifts', data=redshifts )
                f.create_dataset('Lookbacktimes', data=lookbacktimes)
                f.create_dataset('Sfr', data=sfrs)
                f.create_dataset('Metals', data=metals)

        else:
            f = h5.File(outfname, 'r+') # open existing file
            data = f['Sfr']       # load existing sfr data; assuming other cols already written
            data[snap] = sfrs[snap] #write new snapshot data to it; overwrites anything already in that column
            metal_data = f['Metals']   # load existing metallicity data; assuming other cols already written
            metal_data[snap] = metals[snap] #write new snapshot data to it; overwrites anything already in that column
            z_data = f['Redshifts']   # load existing redshift data; assuming other cols already written
            z_data[snap] = redshifts[snap] #write new snapshot data to it; overwrites anything already in that column
            lookbacktime_data = f['Lookbacktimes']   # load existing lookback time data; assuming other cols already written
            lookbacktime_data[snap] = lookbacktimes[snap] #write new snapshot data to it; overwrites anything already in that column
            f.close()

#

    
def read_header(EAGLE, snapshot, zp, nfile):
    """Read various attributes from the header group """
    f = h5.File(EAGLEpath + '/' + EAGLE + '/snapshot_%03d_%s'%(snapshot,zp) + '/snap_%03d_%s.%s.hdf5'%(snapshot,zp, nfile), 'r')
    z = f['Header'].attrs.get('Redshift') # redshift
    #a = f['Header'].attrs.get('Time')
    #boxsize = f['Header']. attrs.get('BoxSize') # L [cMpc/h]
    f.close()
    
    return z

   # z = read_header(paths, snapshot)
    #lookbacktime = cosmo.lookback_time(z).value #calculate lookback time using appropriate cosmology

    # print("The redshift of this snapshot is z=%.2f"%z)
    # print("The lookback time of this snapshot is %.2f"%lookbacktime, 'Gyr') #0 = present day!
#print("The box size is", boxsize*h, "cMpc") #for this simulation, it should be 12.5 cMpc... not sure why this isn't (unit conversions?)    

#

if __name__ == "__main__":

    # Define command line options for the most commonly varied options
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--eagle", dest= 'EAGLE',  help="Which EAGLE simulation (naming convention",type=str, default='RefL0012N0188')
    parser.add_argument("--jupyterlab", dest= 'jupyterlab',  help="Run on EAGLE jupyterlab", default=False, action='store_true')
    parser.add_argument("--username", dest= 'user',  help="EAGLE username",type=str, default="vrs389")
    parser.add_argument("--password", dest= 'password',  help="EAGLE password",type=str, default="GGP13BW3")
    parser.add_argument("--bins", dest= 'nbins',  help="Number of metallicity bins for binning data",type=int, default=60)
    parser.add_argument("--starting_snap", dest= 'ssnap',  help="Option to start from a specific snapshot (useful if doing data download in multiple runs, or redoing a specific snapshot)",type=int, default=0)
    parser.add_argument("--n_snaps", dest= 'nsnaps',  help="number of snapshots)",type=int, default=28)
    parser.add_argument("--outfile", dest= 'outfname',  help="Name of output file",type=str, default="SFRMetallicityFromGasWithMetalsEAGLE_%s.hdf5")
    parser.add_argument("--nfiles", dest= 'nfiles',  help="Number of subfiles per snapshot to read; use for testing, otherwise will be missing data",type=int, default=0)
    
    args = parser.parse_args()

    #path to EAGLE data 
    baseUrl = 'https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots//download?run=' #correct link?
    #headers = {"api-key": args.api_key} #my api key from EAGLE website log in
    if args.jupyterlab==True:
        EAGLEpath = "insert correct link" % (args.EAGLE)
    else:
        EAGLEpath = str(paths.eagle) + '/'+ args.EAGLE


    # Get basic snapshot information (redshift, url) for each of the 100 snapshots
    #r = get(baseUrl) #path to all Illustris TNG simulations
    # names = [sim['name'] for sim in r['simulations']] #names of all available simulations
    # i = names.index('EAGLE%s-%s' % (args.EAGLE, args.lvl)) #index of specific simulation
    # sim = get( r['simulations'][i]['url'] ) #get path to that simulation
    # snaps = get( sim['snapshots'] ) #get path to snapshots for that simulation

    #Number of subfiles per snapshot; usually just leave as the default to get them all (otherwise data WILL be missing)
    #For testing, can set a small number, recommended: 6
    nfiles=args.nfiles

    # Run data download
    getFullSFRMetallicityFromGas(EAGLEpath, args.outfname, args.EAGLE, nfiles, starting_snap=args.ssnap, jupyterlab=args.jupyterlab, nbins=args.nbins, nsnaps=args.nsnaps)
