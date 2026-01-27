## BBH merger populations using cosmic star formation histories of cosmological simulations

This is the code used in the paper: 

**From cosmological simulations to binary black hole mergers: The impact of using analytical star formation history models on gravitational-wave source populations**

Model the binary black hole (BBH) merger population by obtaining the metallicity-dependent cosmic star formation rate density SFRD(Z, z) from an [IllustrisTNG](https://www.tng-project.org/) cosmological simulation. The "cosmic integration" step is done using the [COMPAS](https://github.com/TeamCOMPAS/COMPAS) cosmic integration post-processing tool.
The SFRD(Z, z) can be modeled using a 2D array (obtained from a TNG simulation) or an analytical fit to this 2D array following the methods described in [Neijssel et al. 2019](https://arxiv.org/pdf/1906.08136) and [van Son et al. 2023](https://arxiv.org/pdf/2209.03385). 

Required inputs: 
- COMPAS output file, download [here](https://zenodo.org/records/7612755 )
- GWTC-4 BBH B-spline model file (BBHMassSpinRedshift_BSplineIID.h5), download [here](https://zenodo.org/records/16911563 )

To obtain the SFRD(Z, z) from a TNG simulation:
- Run get_TNGdata_withmetals.py
- Returns SFRMetallicityFromGasWithMetalsTNG[version].hdf5

To fit the analytical model to a TNG simulation SFRD(Z, z):
- Run Fit_model_TNG_SFRD.py
- Returns test_best_fit_parameters_TNG[version].txt

To model BBH merger population using TNG simulation or analytical fit SFRD(Z, z)
- See cosmic_int_guide.txt inside the CosmicIntegration directory for additional detail 
- Run CosmicIntegration/FastCosmicIntegration-withdata.py or CosmicIntegration/FastCosmicIntegration.py
- If using slurm:
    - Set up init_values.py
    - TNG simulation SFRD(Z, z) (as a 2D array): CallCosmicIntegration_data.py 
    - Fitted SFRD(Z, z): CallCosmicIntegration_full.py OR CallCosmicIntegration_data.py without specifying a file containing the SFRD (remove " --datafname " + data_file_name)
- Returns Rate_info_TNG[version].h5

To reproduce the figures:
- Fig 1-2, 11: compare_SFRDparams.py
    - compare_SFR (Fig 1), compare_Zdist (Fig 2)
    - Zdist_3panel_plots (Fig 11)
- Fig 3, 10: SFRD_2Dplots.py
    - SFRD_2Dplot_sidepanels with plotregions=False (Fig 3) and plotregions=True (Fig 10)
- Fig 4-9, 12: TNG_BBHpop_properties.py
    - compare_BBH_data_and_model_rates (Fig 4)
    - plot_BBH_mass_dist_formation_channels (Fig 5)
    - compare_BBH_data_and_model_mass_dist_over_z with plotdiff=False (Fig 6) and plotdiff=True (Fig 12)
    - residuals_BBH_data_and_model_mass_dist (Fig 7)
    - plot_BBH_mass_Z_z with fractionalerror=False (Fig 8) and fractionalerror=True (Fig 9)











