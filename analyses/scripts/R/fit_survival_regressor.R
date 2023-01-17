

library(optparse)
library(randomForestSRC)
library(rhdf5)

include("script_util.R")


##############################################
# LOAD DATA
##############################################

load_training_features <- function(hdf_path){
    data <- h5read(hdf_path, "X")
    rows <- h5read(hdf_path, "instances")
    
    rownames(data) <- rows
    return data
}


load_survival_data <- function(hdf_path, training_instances){

    clinical_matrix <- load_clinical_hdf(hdf_path)

}


##############################################
# RUN SCRIPT
##############################################

# Build argument parser
option_list <- list(
    #make_option("--design", type="character", default="traditional"),
    #make_option("--blockraropt_db", type="character", default=""),
    #make_option("--n_blocks", type="integer", default=1),
    #make_option("--target_t1", default=0.05),
    #make_option("--target_power", default=0.8)
)

parser <- OptionParser(usage="Rscript fit_survival_regressor.R  train_features.hdf clinical.hdf fitted_model.RData", option_list=option_list)
arguments <- parse_args(parser, positional_arguments=3)
#opt <- arguments$options
pargs <- arguments$args

train_hdf <- pargs[1]
fitted_rd <- pargs[2]
#true_p_A <- as.numeric(pargs[3])
#true_p_B <- as.numeric(pargs[4])


