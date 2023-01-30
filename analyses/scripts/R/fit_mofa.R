# fit_mofa.R
#
# 


library("rhdf5")
library("MOFA2")
library("jsonlite")
library("optparse")
source("scripts/R/script_util.R")


###################################################
# HELPER FUNCTIONS
###################################################


###################################################
# PARSE ARGUMENTS
###################################################
option_list <- list(
    make_option("--output_dim", type="integer", default=10, help="Number of dimensions the output should take"),
    make_option("--omic_types", type="character", default="mrnaseq,methylation", help="List of omic assays to use, separated by commas (no spaces). Default 'mrnaseq,methylation,mutation'."),
    make_option("--is_grouped", action="store_true", default=FALSE, help="Toggles sample grouping by cancer type.")
    )

parser <- OptionParser(usage="fit_mofa.R DATA_HDF FITTED_RDS OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
omic_types <- opts$omic_types
omic_types <- strsplit(omic_types, ",")[[1]]
output_dim <- opts$output_dim
is_grouped <- opts$is_grouped

pargs <- arguments$args
data_hdf <- pargs[1]
fitted_rds <- pargs[2]
output_hdf <- pargs[3]

###################################################
# LOAD DATA
###################################################

omic_data <- h5read(data_hdf, "omic_data/data")
feature_genes <- h5read(data_hdf, "omic_data/feature_genes")
feature_assays <- h5read(data_hdf, "omic_data/feature_assays")
instances <- h5read(data_hdf, "omic_data/instances")
instance_groups <- h5read(data_hdf, "omic_data/instance_groups")
target <- h5read(data_hdf, "target")

rownames(omic_data) <- instances

#print(omic_data)
print("OMIC DATA")
print(dim(omic_data))

assay_dists <- list("mrnaseq"="gaussian",
                    "methylation"="gaussian",
                    "mutation"="bernoulli",
                    "rppa"="gaussian",
                    "cna"="gaussian")

####################################################
# PREPARE DATA 
####################################################

data_matrices <- list()
for(ot in omic_types){
    relevant_cols <- (feature_assays == ot)
    relevant_data <- omic_data[,relevant_cols]
    colnames(relevant_data) <- feature_genes[relevant_cols]

    relevant_data <- relevant_data[,colSums(is.nan(relevant_data)) < 0.05*nrow(relevant_data)]
 
    data_matrices[[ot]] <- t(relevant_data) 
}

print("DATA")
print(lapply(data_matrices, dim))

####################################################
# PREPARE MOFA OBJECT
####################################################

# MOFA object
mofa_object <- NULL
if(is_grouped){
    mofa_object <- create_mofa(data_matrices, groups=instance_groups)
}else{
    mofa_object <- create_mofa(data_matrices)
}

###########################
# data options
data_opts <- get_default_data_options(mofa_object)
data_opts$scale_views <- TRUE

###########################
# model options
model_opts <- get_default_model_options(mofa_object)
model_opts$num_factors <- output_dim

model_opts$likelihoods <- sapply(omic_types, function(n){ return(assay_dists[[n]]) }, USE.NAMES=TRUE)

print("VIEW LIKELIHOODS")
print(model_opts$likelihoods)


###########################
# training options 
train_opts <- get_default_training_options(mofa_object)
train_opts$convergence_mode <- "medium"

##########################
# Prepare MOFA
mofa_object <-prepare_mofa(object=mofa_object,
                           data_options=data_opts,
                           model_options=model_opts,
                           training_options=train_opts) 

####################################################
# FIT MOFA MODEL
####################################################

outfile <- system.file("extdata","test_data.RData", package="MOFA2")
trained_mofa <- run_mofa(mofa_object, outfile=outfile, save_data=FALSE,
                                                       use_basilisk=TRUE) 


####################################################
# SAVE MODEL & TRANSFORMED DATA
####################################################



