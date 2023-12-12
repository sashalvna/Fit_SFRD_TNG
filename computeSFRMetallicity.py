import numpy as np
from gadget import *

TNG               = 50
lvl               = 1
TNGpath           = "/virgo/simulations/IllustrisTNG/TNG%d-%d/output/" % (TNG,lvl)
TNGVariationsPath = "/isaac/ptmp/gc/apillepi/sims.TNG_method/"

def getSFRMetallicityFromStarsVariations():
  import glob
  runs = glob.glob( TNGVariationsPath + "L25n512_????" )
  runs.sort()
  
  snap = 4
  
  nBinsMass = 60
  nBinsAge  = 13*6
  
  for run in runs:
    print( run, os.path.basename(run) )
    mbins = np.logspace( -10, 0., nBinsMass+1 )
    abins = np.linspace( 0, 13., nBinsAge+1 )
    mass  = np.zeros( (nBinsMass,nBinsAge) )
    
    s = gadget_readsnap( snap, snappath=run + "/output/", snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
  
    fname = "%s/output/snapdir_%03d/snap_%03d.%s.hdf5" % (run, snap, snap, "%d")
    for ifile in range( s.num_files ):
      with h5py.File(fname % ifile, "r") as f:
        print( "Reading file %d/%d." % (ifile,s.num_files) )
        
        if not "PartType4" in f:
          continue
        
        pStars = f["PartType4"]
      
        Ages   = pStars["GFM_StellarFormationTime"][:]
        Masses = pStars["GFM_InitialMass"][:]
        Metals = pStars["GFM_Metallicity"][:]
      
        i,         = np.where( Ages > 0 )
        AgesInGyrs = s.cosmology_get_lookback_time_from_a( Ages[i], is_flat=True )
      
        data, ex, ey = np.histogram2d( Metals[i], AgesInGyrs, bins=[mbins,abins], weights=Masses[i] )
        mass += data    
    
    if mass.sum() > 0:    
      dset = os.path.basename(run) + "/"
      with h5py.File("SFRMetallicityFromStarsVariations.hdf5", "a") as f:
        f.create_dataset(dset+'MetalBins', data=mbins )
        f.create_dataset(dset+'AgeBins', data=abins )
        f.create_dataset(dset+'Mass', data=mass )

def getSFRMetallicityFromStars():
  snap = 99
  
  nBinsMass = 60
  nBinsAge  = 13*6
  
  mbins = np.logspace( -10, 0., nBinsMass+1 )
  abins = np.linspace( 0, 13., nBinsAge+1 )
  mass  = np.zeros( (nBinsMass,nBinsAge) )
  
  s = gadget_readsnap( snap, snappath=TNGpath, snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
  
  fname = "%s/snapdir_%03d/snap_%03d.%s.hdf5" % (TNGpath, snap, snap, "%d")
  for ifile in range( s.num_files ):
    with h5py.File(fname % ifile, "r") as f:
      print( "Reading file %d/%d." % (ifile,s.num_files) )
      
      pStars = f["PartType4"]
      
      Ages   = pStars["GFM_StellarFormationTime"][:]
      Masses = pStars["GFM_InitialMass"][:]
      Metals = pStars["GFM_Metallicity"][:]
      
      i,         = np.where( Ages > 0 )
      AgesInGyrs = s.cosmology_get_lookback_time_from_a( Ages[i], is_flat=True )
      
      data, ex, ey = np.histogram2d( Metals[i], AgesInGyrs, bins=[mbins,abins], weights=Masses[i] )
      mass += data
  
  with h5py.File("SFRMetallicityFromStarsTNG%d-%d.hdf5" % (TNG,lvl), "w") as f:
    f.create_dataset('MetalBins', data=mbins )
    f.create_dataset('AgeBins', data=abins )
    f.create_dataset('Mass', data=mass )

def getSFRMetallicityFromGas( snap, nBinsMass=60 ):
  mbins = np.logspace( -10, 0., nBinsMass+1 )
  mass  = np.zeros( (nBinsMass) )
  
  s = gadget_readsnap( snap, snappath=TNGpath, snapbase='snap_', loadonlytype=[5], loadonly=['mass'], chunk=0 )
  
  fname = "%s/snapdir_%03d/snap_%03d.%s.hdf5" % (TNGpath, snap, snap, "%d")
  for ifile in range( s.num_files ):
    with h5py.File(fname % ifile, "r") as f:
      print( "Reading file %d/%d." % (ifile,s.num_files) )
      
      pStars = f["PartType0"]
      
      Masses = pStars["StarFormationRate"][:]
      Metals = pStars["GFM_Metallicity"][:]
      
      data, e = np.histogram( Metals, bins=mbins, weights=Masses )
      mass += data
  
  return mass, s.redshift, s.cosmology_get_lookback_time_from_a( s.time, is_flat=True )
  
  with h5py.File("SFRMetallicityFromStarsTNG%d-%d.hdf5" % (TNG,lvl), "w") as f:
    f.create_dataset('MetalBins', data=mbins )
    f.create_dataset('AgeBins', data=abins )
    f.create_dataset('Mass', data=mass )

def getFullSFRMetallicityFromGas():
  if os.path.exists("SFRMetallicityFromGasTNG%d-%d.hdf5" % (TNG,lvl)):
    with h5py.File("SFRMetallicityFromGasTNG%d-%d.hdf5" % (TNG,lvl), "r") as f:
      ages      = f["Lookbacktimes"][:]
      redshifts = f["Redshifts"][:]
      masses    = f["Sfr"][:]
  else:
    redshifts = np.zeros( 100 )
    ages      = np.zeros( 100 )
    masses    = np.zeros( (100,60) )
  
  Count = 0
  for snap in range( 0, 100 ):
    if masses[snap].sum() == 0:
      print( "Doing snap %d." % snap )
      m, z, t = getSFRMetallicityFromGas( snap )
    
      masses[snap,:]  = m
      redshifts[snap] = z
      ages[snap]      = t
      
      Count += 1

  if Count > 0:
    mbins = np.logspace( -10, 0., 60+1 )
    with h5py.File("SFRMetallicityFromGasTNG%d-%d.hdf5" % (TNG,lvl), "w") as f:
      f.create_dataset('MetalBins', data=mbins )
      f.create_dataset('Redshifts', data=redshifts )
      f.create_dataset('Lookbacktimes', data=ages )
      f.create_dataset('Sfr', data=masses )

getSFRMetallicityFromStars()
getFullSFRMetallicityFromGas()
#getSFRMetallicityFromStarsVariations()

