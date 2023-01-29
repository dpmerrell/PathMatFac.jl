
library("rhdf5")
library("PLIER")
library("jsonlite")
library("optparse")

source("scripts/R/script_util.R")


#########################################################
# PARSE ARGUMENTS
#########################################################

option_list <- list(
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'.")
    )

parser <- OptionParser(usage="transform_plier.R DATA_HDF FITTED_RDS OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
omic_type <- opts$omic_type

pargs <- arguments$args
data_hdf <- pargs[1]
fitted_rds <- pargs[2]
output_hdf <- pargs[3]


#########################################################
# LOAD DATA
#########################################################

omic_data <- h5read(data_hdf, "omic_data/data")
feature_assays <- h5read(data_hdf, "omic_data/feature_assays")
feature_genes <- h5read(data_hdf, "omic_data/feature_genes")
instances <- h5read(data_hdf, "omic_data/instances")
instance_groups <- h5read(data_hdf, "omic_data/instance_groups")
target <- h5read(data_hdf, "target")

mrnaseq_cols <- (feature_assays == omic_type)
omic_data <- omic_data[,mrnaseq_cols]
feature_genes <- feature_genes[mrnaseq_cols]
rownames(omic_data) <- instances
colnames(omic_data) <- feature_genes


#########################################################
# LOAD MODEL
#########################################################

fitted_params <- readRDS(fitted_rds)
mu <- fitted_params[["mu"]]
sigma <- fitted_params[["sigma"]]
fitted_plier <- fitted_params[["plier"]]
factors <- fitted_plier$Z
print("FACTORS:")
print(factors[1:3,1:3])
print(dim(factors))
factor_norms <- apply(factors, 2, function(v) norm(v, "2")) 
factors <- t(t(factors) / factor_norms)


#########################################################
# TRANSFORM DATA
#########################################################

used_cols <- names(mu) 
omic_data <- omic_data[,used_cols]

omic_data <- t((t(omic_data) - mu)/sigma)
omic_data[is.nan(omic_data)] <- 0.0

X <- omic_data %*% factors

########################################################
# SAVE RESULTS
########################################################

h5write(X, output_hdf, "X")
h5write(instances, output_hdf, "instances")
h5write(instance_groups, output_hdf, "instance_groups")
h5write(target, output_hdf, "target")

