#! /bin/bash

################################
## Simulate the data

#rm -f simulated_data.hdf simulated_params.bson
#julia --project=. scripts/julia/simulate_matfac.jl temp/real_data_runs/omic_data/heldout_ctypes\=OV\:GBM\:LAML__kept_ctypes\=HNSC:CESC:ESCA:STAD.hdf data/reactome/top_pwys\=1000.json simulated_params.bson simulated_data.hdf K=5 use_batch=true omic_types=mrnaseq:methylation:cna:mutation var_filter=0.01 configuration=fsard A_density=0.025 missingness=0.125
#
###############################
## Visualize data and parameters
#python scripts/python/vis_dataset.py simulated_data.hdf simulated_data.png --title "Simulated dataset"
#
#xdg-open simulated_data.png &
#
#julia --project=. scripts/julia/bson_to_hdf.jl simulated_params.bson simulated_params.hdf
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_col_params.png --param col_params
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_batch_params.png --param batch_params
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_Y.png --param Y 
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_X.png --param X 
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_A.png --param A 
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_S.png --param S 
#python scripts/python/vis_model_params.py simulated_params.hdf simulated_tau.png --param tau

################################
## Fit model to the data

#rm -f simulated_params_fit.hdf simulated_data_transformed.hdf
#julia --project=. scripts/julia/fit_matfac.jl simulated_data.hdf data/reactome/top_pwys\=1000.json simulated_params_fit.bson simulated_data_transformed.hdf K=5 use_batch=true omic_types=mrnaseq:methylation:cna:mutation var_filter=1.0 fsard_A_prior_frac=0.1 fsard_frac_atol=0.05 configuration=fsard max_epochs=3000 lr_theta=0.5 lr=0.5 lr_ard=0.001 
#
#
#################################
### Visualize fit 
#julia --project=. scripts/julia/bson_to_hdf.jl simulated_params_fit.bson simulated_params_fit.hdf

python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_col_params_fit.png --param col_params
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_batch_params_fit.png --param batch_params
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_Y_fit.png --param Y 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_X_fit.png --param X 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_A_fit.png --param A 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_S_fit.png --param S 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_tau_fit.png --param tau
