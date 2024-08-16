"""
This script makes a slurm job to run FastCosmicIntegration with a specified set of parameters 
"""
import numpy as np
import os
from subprocess import Popen, PIPE, call
import subprocess
import sys
import time
from fnmatch import fnmatch
import h5py
sys.path.append( str(os.getcwd()) + '/scripts/')
import paths
import init_values as In

##################################################################
# This is the slurm script youre using
#SBATCH --partition=%s              # Partition to submit to
##################################################################
# note indentation needs to be like this
SlurmJobString="""#!/bin/bash
#SBATCH --job-name=%s          #job name
#SBATCH --nodes=%s             # Number of nodes
#SBATCH --ntasks=%s            # Number of cores
#SBATCH --output=%s            # output storage file
#SBATCH --error=%s             # error storage file
#SBATCH --time=%s              # Runtime in minutes
#SBATCH --mem=%s               # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -p %s
#SBATCH --mail-user=%s         # Send email to user
#SBATCH --mail-type=FAIL       #
#
#Print some stuff on screen
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export QT_QPA_PLATFORM=offscreen # To avoid the X Display error
#
cd %s
#
# Run your job
%s
"""

###############################################
###
###############################################
def RunSlurmBatch(run_dir = None, job_name = "cosmic_integration", dependency = False, dependent_ID = None):
    if not dependency:
        sbatchArrayCommand = 'sbatch ' + os.path.join(run_dir+job_name+'.sbatch') 
    else:
        sbatchArrayCommand = 'sbatch --dependency=afterok:' + str(int(dependent_ID)) + ' ' + os.path.join(run_dir+job_name+'.sbatch') 

    # Open a pipe to the sbatch command.
    proc = Popen(sbatchArrayCommand, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

    # Send job_string to sbatch
    if (sys.version_info > (3, 0)):
        proc.stdin.write(sbatchArrayCommand.encode('utf-8'))
    else:
        proc.stdin.write(sbatchArrayCommand)

    print('sbatchArrayCommand:', sbatchArrayCommand)
    out, err = proc.communicate()
    print("out = ", out)
    job_id = out.split()[-1]
    print("job_id", job_id)
    return job_id


#########################################################
# Assuming you have The COMPAS simulation data output
# Make a slurm job to call 
def Call_Cosmic_Integration(root_out_dir, COMPASfilename, rate_file_name, data_file_name, user_email, jname = 'CI',
                            ZdepSFRD_param_sets = [[],[]], #[[fid_dpdZ_parameters, fid_sfr_parameters]],
                           partitions = 'defq', Wtime = "5:00:00", mem = "150000",
                           number_of_nodes = 1, number_of_cores = 1):
    """
    Call slurm batch 

    Args:
        root_out_dir           --> [string] Path to the COMPAS file that contains the simulation data
        COMPASfilename         --> [string] Name of the COMPAS file
        rate_file_name         --> [string] Name of the output file containing the rates
        ZdepSFRD_param_sets    --> [array of float arrays] 
        # consists of: 
            dco_type               --> [string] Which DCO type you used to calculate rates 
            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)

        maxzdet                   --> [float] max z for detection rate, will be used as maxz: Maximum redshhift up to where we would like to store the data

    """
    if not os.path.isdir(root_out_dir+'/RateData/'):
        print('make output dir for rate data: ', root_out_dir+'/RateData/')
        os.mkdir(root_out_dir+'/RateData/')

    # Index to name your job
    n_CI = 1 
    check_job_completionID = []

    # Run over each complete variation of the metallicity dependent SFRD
    for SFRD_zZ in ZdepSFRD_param_sets:  
        job_name = jname+str(n_CI)
        
        print(10* "*" + ' You are Going to Run FastCosmicIntegration.py')
        print('dPdZ data = ', SFRD_zZ[0])
        print('SFR(z) data = ', SFRD_zZ[1])
        
        mu0, muz, sigma0, sigmaz, alpha0 = SFRD_zZ[0]
        sf_a, sf_b, sf_c, sf_d           = SFRD_zZ[1]
        DEPEND, append_job_id = False, 0

        # Flag to pass to FasCosmicIntegrator   #'/RateData/'+str(n_CI)+'_'+rate_file_name+\
        Flags = " --path "+root_out_dir + " --filename "+ COMPASfilename+" --outfname " +'/home/alevina1/data/RateData/'+str(n_CI)+'_'+rate_file_name+\
        " --mu0 " +str(mu0)+" --muz "+str(muz)+" --sigma0 "+str(sigma0)+" --sigmaz "+str(sigmaz)+" --alpha "+str(alpha0)+\
        " --aSF " +str(sf_a)+" --bSF "+str(sf_b)+" --cSF "+str(sf_c)+" --dSF "+str(sf_d)+\
        " --weight "+"mixture_weight"+ " --zstep "+"0.05"+" --sens "+"O3"+ " --m1min "+"5."+ " --dco_type BBH"+ " --BinAppend"+ \
        " --redshiftBinSize "+"0.05" + ' --maxzdet ' + "14" + ' --maxz ' + "14" + " --cosmology " + "Planck15 " + " --zSF " + "14" \
        + " --datafname " + data_file_name

        run_dir = In.script_dir +'/CosmicIntegration/'

        # Make and safe a slurm command
        job_line = "python FastCosmicIntegration-withdata.py "+Flags+" > "+ root_out_dir + "/slurm_out/"+job_name+".log"

        # Make slurm script string
        interface_job_string = SlurmJobString % (job_name, number_of_nodes, number_of_cores, \
        root_out_dir+'/slurm_out/'+job_name+'.out', root_out_dir+'/slurm_out/'+job_name+'.err', Wtime, mem, partitions, user_email, run_dir, job_line)
        
        # Write your bash file
        sbatchFile = open(run_dir+job_name+'.sbatch','w')
        print('writing ', run_dir+job_name+'.sbatch')
        sbatchFile.write(interface_job_string)
        sbatchFile.close()
  
        # Submit the job to sbatch! 
        CIjob_id = RunSlurmBatch(run_dir = run_dir, job_name = job_name ,\
        dependency = DEPEND, dependent_ID = append_job_id)

        check_job_completionID.append(CIjob_id.decode("utf-8"))

        n_CI += 1
        DEPEND, append_job_id = False, CIjob_id # no need for it to be dependent
    
    
    np.savetxt(root_out_dir+'/RateData/CI_job_IDs.txt', np.c_[check_job_completionID],header = "# job IDs ", delimiter=',', fmt="%s")

if __name__ == "__main__": 
    # MAIN

    # Initialize values
    In.init()

    # Best fit parameters
    mu0_best, muz_best, sigma0_best, sigmaz_best, alpha0_best,sf_a_best, sf_b_best, sf_c_best, sf_d_best = In.mu0_best, In.muz_best, In.sigma0_best, In.sigmaz_best, In.alpha0_best,In.sf_a_best, In.sf_b_best, In.sf_c_best, In.sf_d_best
    fid_dpdZ_parameters = [mu0_best, muz_best, sigma0_best, sigmaz_best, alpha0_best]
    fid_sfr_parameters  = [sf_a_best, sf_b_best, sf_c_best, sf_d_best]

    ################################################################
    # All at once
    ################################################################
    Call_Cosmic_Integration(In.data_dir, In.COMPASfilename, In.rate_file_name, In.data_file_name, In.user_email, jname = 'D_CI',
                           ZdepSFRD_param_sets =[[In.fid_dpdZ_parameters, fid_sfr_parameters]],
                           partitions = 'parallel', Wtime = "2:00:00", mem = "120000") #conroy,hernquist,shared

