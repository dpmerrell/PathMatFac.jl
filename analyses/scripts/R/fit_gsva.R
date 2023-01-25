
library("GSVA")
library("jsonlite")
library("rhdf5")
library("optparse")
library("nipals")
source("scripts/R/script_util.R")


###########################
# PARSE ARGUMENTS
option_list <- list(
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'."),
    make_option("--kcdf", type="character", default="Gaussian", help="'Gaussian' or 'Poisson'."),
    make_option("--minsize", default=10, help="Minimum gene set size. Default 10."),
    make_option("--maxsize", default=1000, help="Maximum gene set size. Default 100."),
    make_option("--threads", default=1, help="number of CPU threads to use. Default 1."),
    make_option("--output_dim", default=10, help="Dimension of the output. We use PCA to transform the GSVA output to this dimension")
    )

parser <- OptionParser(usage="fit_gsva.R DATA_HDF PATHWAY_JSON TRANSFORMED_HDF FITTED_MODEL_RDS [OPTS]",
                       option_list=option_list)

print("PARSING ARGS")

arguments <- parse_args(parser, positional_arguments=4)

opts <- arguments$options
omic_type <- opts$omic_type
kcdf <- opts$kcdf
minsize <- opts$minsize
maxsize <- opts$maxsize
threads <- opts$threads
output_dim <- opts$output_dim

pargs <- arguments$args
data_hdf <- pargs[1]
pwy_json <- pargs[2]
transformed_hdf <- pargs[3]
fitted_model_rds <- pargs[4]

####################################
# LOAD PATHWAYS

print("LOADING PATHWAYS")
pwy_dict <- read_json(pwy_json)
pwys <- pwy_dict$pathways
pwy_names <- pwy_dict$names

print("TRANSLATING TO GENESETS")
# translate to genesets
genesets <- pwys_to_genesets(pwys, pwy_names)


###################################
# LOAD RNASEQ DATA

print("LOADING OMIC DATA")
omic_data <- h5read(data_hdf, "omic_data/data")
feature_genes <- h5read(data_hdf, "omic_data/feature_genes")
feature_assays <- h5read(data_hdf, "omic_data/feature_assays")
instances <- h5read(data_hdf, "omic_data/instances")
instance_groups <- h5read(data_hdf, "omic_data/instance_groups")

mrnaseq_cols <- (feature_assays == omic_type)
omic_data <- omic_data[,mrnaseq_cols]
feature_genes <- feature_genes[mrnaseq_cols]
rownames(omic_data) <- instances
colnames(omic_data) <- feature_genes


#######################################
# HANDLE MISSING VALUES

# Only keep genes that are observed for most samples
omic_data <- omic_data[,colSums(is.nan(omic_data)) < 0.1*dim(omic_data)[1]]
print(dim(omic_data))

# Stick to inexpensive median imputation for now.
# (Reduces impact on enrichment??)
omic_data <- median_impute(omic_data)

############################################
# RUN GSVA 

# Call GSVA with the given parameter settings.
curried_gsva <- function(omic_data, genesets){
    cat("Running GSVA...\n")
    omic_data <- t(omic_data)
    results <- gsva(omic_data, genesets, min.sz=minsize,
                    max.sz=maxsize, kcdf=kcdf, parallel.sz=1)
    return(t(results))
}

fitted_model <- list()

gsva_results <- curried_gsva(omic_data, genesets)
used_pwys <- colnames(gsva_results)

fitted_model[["used_genes"]] <- colnames(omic_data)
fitted_model[["used_pathways"]] <- used_pwys


###########################################
# RUN PCA

pca_results <- nipals(gsva_results, ncomp=output_dim)

X <- pca_result$scores

fitted_model[["mu"]] <- pca_results$center
fitted_model[["sigma"]] <- pca_results$scale
fitted_model[["Y"]] <- t(pca_results$loadings)
fitted_model[["R2"]] <- pca_results$R2

###########################################
# SAVE FITTED MODEL AND TRANSFORMED DATA

saveRDS(fitted_model, fitted_model_rds)

# Need to get the target data
target <- h5read(data_hdf, "target")

h5write(X, transformed_hdf, "X")
h5write(rownames(gsva_results), transformed_hdf, "instances")
h5write(instance_groups, transformed_hdf, "instance_groups")
h5write(target, transformed_hdf, "target") 


