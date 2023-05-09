# fit_mofa.R
#
# 

library("reticulate")

library("rhdf5")
library("MOFA2")
library("optparse")
source("scripts/R/script_util.R")


###################################################
# HELPER FUNCTIONS
###################################################

reconstruct_matrix <- function(matrix_ls, original_rows){

    M <- length(original_rows)
    N <- ncol(matrix_ls[[1]])
    full_matrix <- matrix(NaN, M, N)
    rownames(full_matrix) <- original_rows

    gp_names <- names(matrix_ls) 
    for(gp in gp_names){
        gp_rows <- rownames(matrix_ls[[gp]])
        full_matrix[gp_rows,] <- matrix_ls[[gp]][gp_rows,] 
    }

    return(full_matrix)
}

###################################################
# PARSE ARGUMENTS
###################################################
option_list <- list(
    make_option("--output_dim", type="integer", default=10, help="Number of dimensions the output should take"),
    make_option("--omic_types", type="character", default="mutation:methylation:mrnaseq:cna", help="List of omic assays to use, separated by colons (no spaces). Default 'mrnaseq:methylation:mutation:cna'."),
    make_option("--is_grouped", action="store_true", default=TRUE, help="Toggles sample grouping by cancer type."),
    make_option("--var_filter", type="numeric", default=0.5, help="fraction of most-variable features to keep within each view"),
    make_option("--mofa_python", help="path to python that has mofapy2 installed")
    )

parser <- OptionParser(usage="fit_mofa.R DATA_HDF FITTED_RDS OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
omic_types <- opts$omic_types
omic_types <- strsplit(omic_types, ":")[[1]]
output_dim <- opts$output_dim
is_grouped <- opts$is_grouped
var_filter <- opts$var_filter
mofa_python <- opts$mofa_python

reticulate::use_python(mofa_python) 

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

assay_dists <- list("mrnaseq"="gaussian",
                    "methylation"="gaussian",
                    "mutation"="bernoulli",
                    "rppa"="gaussian",
                    "cna"="gaussian")

####################################################
# PREPARE DATA 
####################################################

data_matrices <- list() # List of omic views
all_features <- list()  # Will be a vector of all feature names
mu <- list() # list of views' means
sigma <- list() # list of views' standard deviations

for(ot in omic_types){
    relevant_cols <- (feature_assays == ot)
    relevant_data <- omic_data[,relevant_cols]
    relevant_genes <- feature_genes[relevant_cols]
    colnames(relevant_data) <- sapply(relevant_genes, function(g) paste(g, "_", ot, sep=""))

    # Filter the data by NaNs and variance
    relevant_data <- relevant_data[,colSums(is.nan(relevant_data)) < 0.05*nrow(relevant_data)]
    feature_vars <- apply(relevant_data, 2, function(v) var(v, na.rm=TRUE))
    min_var <- quantile(feature_vars, 1-var_filter)
    relevant_data <- relevant_data[,feature_vars >= min_var]

    all_features[[ot]] <- colnames(relevant_data)
    data_matrices[[ot]] <- t(relevant_data) 
    mu[[ot]] <- rowMeans(omic_data, na.rm=TRUE)                     
    sigma[[ot]] <- apply(omic_data, 1, function(v) sd(v, na.rm=TRUE))

    data_matrices[[ot]] <- (data_matrices[[ot]] - mu[[ot]])/sigma[[ot]] 
}
all_features <- unlist(all_features)
mu <- unlist(mu)
sigma <- unlist(sigma)

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
data_opts$scale_views <- FALSE

###########################
# model options
model_opts <- get_default_model_options(mofa_object)
model_opts$num_factors <- output_dim

model_opts$likelihoods <- sapply(omic_types, function(n){ return(assay_dists[[n]]) }, USE.NAMES=TRUE)

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

temp_file <- paste(output_hdf, "_temp", sep="")

trained_mofa <- run_mofa(mofa_object, temp_file,
                                      save_data=FALSE,
                                      use_basilisk=FALSE) 
file.remove(temp_file)
X_ls <- get_expectations(trained_mofa, "Z")
Y_ls <- get_expectations(trained_mofa, "W")

X <- reconstruct_matrix(X_ls, instances)
Y <- reconstruct_matrix(Y_ls, all_features)  

####################################################
# SAVE MODEL & TRANSFORMED DATA
####################################################

h5write(X, output_hdf, "X")
h5write(instances, output_hdf, "instances")
h5write(instance_groups, output_hdf, "instance_groups")
h5write(target, output_hdf, "target")

fitted_model <- list()
fitted_model[["Y"]] <- Y
fitted_model[["mu"]] <- mu
fitted_model[["sigma"]] <- sigma

saveRDS(fitted_model, fitted_rds)


