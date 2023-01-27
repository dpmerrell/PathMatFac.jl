
library("rhdf5")
library("PLIER")
library("jsonlite")
library("optparse")

source("scripts/R/script_util.R")

###################################################
# SOME HELPER FUNCTIONS
###################################################


geneset_to_binvec <- function(geneset, feat_to_idx){
    
    N_genes <- length(feat_to_idx)
    binvec <- rep(0.0, N_genes)

    for(gene in geneset){
        binvec[feat_to_idx[[gene]]] <- 1.0 
    }

    return(binvec)
}

genesets_to_priormat <- function(genesets, omic_features){

    feat_to_idx <- list()
    for(i in 1:length(omic_features)){
        feat <- omic_features[[i]]
        feat_to_idx[[feat]] <- i
    }
    
    binvecs <- list()
    for(gs_name in names(genesets)){
        binvecs[[gs_name]] <- geneset_to_binvec(genesets[[gs_name]], feat_to_idx)
    }

    priormat <- as.matrix(data.frame(binvecs))
    rownames(priormat) <- omic_features
    return(priormat)
}

###################################################
# PARSE ARGUMENTS
###################################################
option_list <- list(
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'."),
    make_option("--output_dim", type="integer", default=10, help="Number of dimensions the output should take")
    )

parser <- OptionParser(usage="run_plier.R DATA_HDF PATHWAY_JSON FITTED_RDS OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=4)

opts <- arguments$options
omic_type <- opts$omic_type
output_dim <- opts$output_dim

pargs <- arguments$args
data_hdf <- pargs[1]
pwy_json <- pargs[2]
fitted_rds <- pargs[3]
output_hdf <- pargs[4]


###################################################
# LOAD & PREP DATA
###################################################

omic_data <- h5read(data_hdf, "omic_data/data")
feature_genes <- h5read(data_hdf, "omic_data/feature_genes")
feature_assays <- h5read(data_hdf, "omic_data/feature_assays")
instances <- h5read(data_hdf, "omic_data/instances")
instance_groups <- h5read(data_hdf, "omic_data/instance_groups")
target <- h5read(data_hdf, "target")

mrnaseq_cols <- (feature_assays == omic_type)
omic_data <- omic_data[,mrnaseq_cols]
feature_genes <- feature_genes[mrnaseq_cols]
rownames(omic_data) <- instances
colnames(omic_data) <- feature_genes

# Filter out the columns with too many NaNs
omic_data <- omic_data[,(colSums(is.nan(omic_data)) < 0.05*nrow(omic_data))]

# Mean imputation for the remaining NaNs
cmeans <- colMeans(omic_data, na.rm=TRUE)
nan_idx <- is.nan(omic_data)
omic_data[nan_idx] <- 0.0
omic_data <- omic_data + (nan_idx*cmeans)

#print(omic_data)
print("OMIC DATA")
print(dim(omic_data))

####################################################
# LOAD & PREP PATHWAYS
####################################################

pwy_dict <- read_json(pwy_json)
pwys <- pwy_dict$pathways
pwy_names <- pwy_dict$names
genesets <- pwys_to_genesets(pwys, pwy_names)
priormat <- genesets_to_priormat(genesets, colnames(omic_data)) 

#print(priormat)
print("PRIOR MAT")
print(dim(priormat))

# Only keep the features that are present in the pathways
omic_data <- omic_data[, rownames(priormat)]

print("OMIC DATA")
print(dim(omic_data))

##################################################
# RUN PLIER
##################################################

# Call PLIER with the given parameter settings.
plier_result <- PLIER(t(omic_data), priormat, trace=TRUE)

X <- t(plier_result$B)
X[,c(1:output_dim)]  # Keep the top-`output_dim` dimensions


#################################################
# SAVE RESULTS
#################################################

saveRDS(plier_result, fitted_rds)

h5write(X, output_hdf, "X")
h5write(instances, output_hdf, "instances")
h5write(instance_groups, output_hdf, "instance_groups")
h5write(target, output_df, "target")


