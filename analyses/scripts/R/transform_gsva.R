
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
    make_option("--threads", default=1, help="number of threads to use. Default 1.")
    )

parser <- OptionParser(usage="fit_gsva.R TRAIN_HDF TEST_HDF FITTED_RDS PATHWAY_JSON TRANSFORMED_HDF [OPTS]",
                       option_list=option_list)

print("PARSING ARGS")

arguments <- parse_args(parser, positional_arguments=5)

opts <- arguments$options
omic_type <- opts$omic_type
kcdf <- opts$kcdf
threads <- opts$threads

pargs <- arguments$args
train_hdf <- pargs[1]
test_hdf <- pargs[2]
fitted_rds <- pargs[3]
pwy_json <- pargs[4]
transformed_hdf <- pargs[5]

#################################
# Load the fitted model

print("LOADING FITTED MODEL")
fitted_params <- readRDS(fitted_rds)
used_genes <- fitted_params[["used_genes"]]
used_pathways <- fitted_params[["used_pathways"]] 

###############################
# LOAD OMIC DATA

# Need to use the training set again during test; 
# have to run GSVA on the combined (train, test) data 
# for better consistency in the gene-wise KCDF estimates.
# Even this is imperfect, though. It would be better
# if GSVA could store gene-wise KCDFs from the 
# train set and use them to score the test set.
# Implementing that seems out-of-scope for this project.

print("LOADING OMIC DATA")

train_omic <- h5read(train_hdf, "omic_data/data")
train_genes <- h5read(train_hdf, "omic_data/feature_genes")
train_assays <- h5read(train_hdf, "omic_data/feature_assays")
train_omic <- train_omic[,train_assays == omic_type]
train_genes <- train_genes[train_assays == omic_type]
colnames(train_omic) <- train_genes
train_omic <- train_omic[,used_genes]
train_instances <- h5read(train_hdf, "omic_data/instances")
train_groups <- h5read(train_hdf, "omic_data/instance_groups")
rownames(train_omic) <- train_instances

test_omic <- h5read(test_hdf, "omic_data/data")
test_genes <- h5read(test_hdf, "omic_data/feature_genes")
test_assays <- h5read(test_hdf, "omic_data/feature_assays")
test_omic <- test_omic[,test_assays == omic_type]
test_genes <- test_genes[test_assays == omic_type]
colnames(test_omic) <- test_genes
test_omic <- test_omic[,used_genes]
test_instances <- h5read(test_hdf, "omic_data/instances")
test_groups <- h5read(test_hdf, "omic_data/instance_groups")
rownames(test_omic) <- test_instances


################################
# LOAD PATHWAYS

print("LOADING PATHWAYS")
pwy_dict <- read_json(pwy_json)
pwys <- pwy_dict$pathways
pwy_names <- pwy_dict$names

print("TRANSLATING TO GENESETS")
# translate to genesets
genesets <- pwys_to_genesets(pwys, pwy_names)

# Restrict to the genesets used during training
genesets <- genesets[used_pathways]


##################################
# RUN GSVA ON COMBINED DATA

print("IMPUTING MISSING VALUES")
combined_omic <- rbind(train_omic, test_omic)
combined_omic <- median_impute(combined_omic)

print("RUNNING GSVA")
gsva_results <- t(gsva(t(combined_omic), genesets, min.sz=1,
                       max.sz=Inf, kcdf=kcdf, parallel.sz=threads))

gsva_results <- gsva_results[test_instances,]


##################################
# TRANSFORM TEST DATA
#mu <- fitted_params[["mu"]]
#sigma <- fitted_params[["sigma"]]

#print("TRANSFORMING VIA PCA")
#gsva_results <- t((t(gsva_results) - mu)/sigma)

# Fortunately, we know gsva_results contains no NaNs.
# So the PCA transform is just matrix multiplication!
#print("GSVA RESULT NANS:")
#print(sum(is.nan(gsva_results)))
#Y <- fitted_params[["Y"]]
#gsva_results <- gsva_results %*% t(Y)


####################################
# SAVE RESULTS
target <- h5read(test_hdf, "target")

h5write(gsva_results, transformed_hdf, "X")
h5write(rownames(gsva_results), transformed_hdf, "instances")
h5write(test_groups, transformed_hdf, "instance_groups")
h5write(target, transformed_hdf, "target")

 
