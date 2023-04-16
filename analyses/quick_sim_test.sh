#! /bin/bash

rm simulated_data.hdf simulated_params.bson

julia --project=. scripts/julia/simulate_matfac.jl temp/real_data_runs/omic_data/heldout_ctypes\=OV\:GBM\:LAML__kept_ctypes\=HNSC:CESC:ESCA:STAD.hdf data/reactome/top_pwys\=500.json simulated_params.bson simulated_data.hdf K=5 use_batch=true omic_types=mrnaseq:methylation:cna:mutation var_filter=0.05 configuration=ard missingness=0.5

python scripts/python/vis_dataset.py simulated_data.hdf simulated_data.png

xdg-open simulated_data.png &

julia --project=. scripts/julia/bson_to_hdf.jl simulated_params.bson simulated_params.hdf
python scripts/python/vis_model_params.py simulated_params.hdf simulated_col_params.png --param col_params
python scripts/python/vis_model_params.py simulated_params.hdf simulated_Y.png --param Y 
python scripts/python/vis_model_params.py simulated_params.hdf simulated_X.png --param X 

