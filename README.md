Model the binary black hole (BBH) merger population by obtaining the metallicity-dependent cosmic star formation rate density SFRD(Z, z) from an IllustrisTNG (https://www.tng-project.org/) cosmological simulation. The "cosmic integration" step is done using the COMPAS cosmic integration post-processing tool (https://github.com/TeamCOMPAS/COMPAS, following the framework from https://arxiv.org/pdf/1906.08136 and https://arxiv.org/pdf/2209.03385).
The SFRD(Z, z) can be modeled using a 2D array (obtained from a TNG simulation) or an analytical fit to this 2D array. 

Required inputs: 
- COMPAS output file, download from: https://zenodo.org/records/7612755 
- GWTC-4 BBH B-spline model file (BBHMassSpinRedshift_BSplineIID.h5) from: https://zenodo.org/records/16911563 

To obtain the SFRD(Z, z) from a TNG simulation:
- Run get_TNGdata_withmetals.py
- Returns SFRMetallicityFromGasWithMetalsTNG[version].hdf5

To fit the analytical model to a TNG simulation SFRD(Z, z):
- Run Fit_model_TNG_SFRD.py
- Returns test_best_fit_parameters_TNG[version].txt

To model BBH merger population using TNG simulation or analytical fit SFRD(Z, z):
- Run CosmicIntegration/FastCosmicIntegration-withdata.py or CosmicIntegration/FastCosmicIntegration.py
- If using slurm:
    - set up init_values.py
    - TNG simulation SFRD(Z, z) (as a 2D array): CallCosmicIntegration_data.py
    - Fitted SFRD(Z, z): CallCosmicIntegration_full.py
- Returns Rate_info_TNG[version].h5

To reproduce the figures:
Fig 1-2: compare_SFRDparams.py compare_SFR (Fig 1), compare_Zdist (Fig 2)
Fig 3, 10: SFRD_2Dplots.py
Fig 4-9, 11: TNG_BBHpop_properties.py
