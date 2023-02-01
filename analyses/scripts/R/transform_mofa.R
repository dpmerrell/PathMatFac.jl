

library(rhdf5)
library(optparse)

source("scripts/R/script_util.R")


###################################################
# PARSE ARGUMENTS
###################################################
option_list <- list(
    #make_option("--variance_filter", type="numeric", default=0.5, help="fraction of most-variable features to keep within each view") 
    )

parser <- OptionParser(usage="transform_mofa.R DATA_HDF FITTED_RDS OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options

pargs <- arguments$args
data_hdf <- pargs[1]
fitted_rds <- pargs[2]
output_hdf <- pargs[3]


######################################################
# LOAD DATA
######################################################

omic_data <- h5read(data_hdf, "omic_data/data")
feature_genes <- h5read(data_hdf, "omic_data/feature_genes")
feature_assays <- h5read(data_hdf, "omic_data/feature_assays")
instances <- h5read(data_hdf, "omic_data/instances")
instance_groups <- h5read(data_hdf, "omic_data/instance_groups")
target <- h5read(data_hdf, "target")
features <- apply(cbind(feature_genes, feature_assays), 1, function(v) paste(v[1],v[2],sep="_"))

rownames(omic_data) <- instances
colnames(omic_data) <- features

######################################################
# LOAD FITTED MODEL
######################################################

fitted_params = readRDS(fitted_rds)
Y <- t(fitted_params[["Y"]])
mu <- fitted_params[["mu"]]
sigma <- fitted_params[["sigma"]]


######################################################
# TRANSFORM DATA
######################################################

omic_data <- omic_data[,colnames(Y)] # Restrict to the modeled features
omic_data <- t((t(omic_data) - mu)/sigma)
X <- linear_transform(omic_data, Y)

######################################################
# SAVE TRANSFORMED DATA
######################################################

h5write(X, output_hdf, "X")
h5write(instances, output_hdf, "instances")
h5write(instance_groups, output_hdf, "instance_groups")
h5write(target, output_hdf, "target")

