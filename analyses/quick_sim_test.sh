#! /bin/bash

###############################
# Simulate the data

rm -f simulated_data.hdf simulated_params.bson
julia --project=. scripts/julia/simulate_matfac.jl temp/real_data_runs/omic_data/heldout_ctypes\=OV\:GBM\:LAML__kept_ctypes\=HNSC:CESC:ESCA:STAD.hdf data/msigdb/hallmark-pwys.json simulated_params.bson simulated_data.hdf K=5 use_batch=true omic_types=mrnaseq:methylation:mutation var_filter=0.05 configuration=fsard A_density=0.1 missingness=0.125

#############################
 Visualize data and parameters
python scripts/python/vis_dataset.py simulated_data.hdf simulated_data.png --title "Simulated dataset"

xdg-open simulated_data.png &

julia --project=. scripts/julia/bson_to_hdf.jl simulated_params.bson simulated_params.hdf
python scripts/python/vis_model_params.py simulated_params.hdf simulated_col_params.png --param col_params
python scripts/python/vis_model_params.py simulated_params.hdf simulated_batch_params.png --param batch_params
python scripts/python/vis_model_params.py simulated_params.hdf simulated_Y.png --param Y 
python scripts/python/vis_model_params.py simulated_params.hdf simulated_X.png --param X 
python scripts/python/vis_model_params.py simulated_params.hdf simulated_A.png --param A 
python scripts/python/vis_model_params.py simulated_params.hdf simulated_S.png --param S 
python scripts/python/vis_model_params.py simulated_params.hdf simulated_beta.png --param beta 

##############################
 Fit model to the data

rm -f simulated_params_fit.hdf simulated_data_transformed.hdf
julia --threads 4 --project=. scripts/julia/fit_matfac.jl simulated_data.hdf data/msigdb/hallmark-pwys.json simulated_params_fit.bson simulated_data_transformed.hdf K=10 use_batch=true omic_types=mrnaseq:methylation:mutation var_filter=1.0 fsard_A_prior_frac=0.7 fsard_frac_atol=0.05 configuration=fsard max_epochs=6000 lr_theta=1.0 lr_regress=1.0 lr=1.0  

################################
## Visualize fit 
julia --project=. scripts/julia/bson_to_hdf.jl simulated_params_fit.bson simulated_params_fit.hdf

julia --project=. scripts/julia/vis_model.jl simulated_params_fit.bson simulated_params_fit.html

python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_col_params_fit.png --param col_params
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_batch_params_fit.png --param batch_params
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_Y_fit.png --param Y 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_X_fit.png --param X 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_A_fit.png --param A 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_S_fit.png --param S 
python scripts/python/vis_model_params.py simulated_params_fit.hdf simulated_beta_fit.png --param beta

python scripts/python/score_matfac.py simulated_params.hdf simulated_params_fit.hdf sim_scores.json
