
library("GSVA")
library("jsonlite")
library("rhdf5")
library("optparse")
library("missForest")
source("scripts/R/script_util.R")


###########################
# PARSE ARGUMENTS
option_list <- list(
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'."),
    make_option("--kcdf", type="character", default="Gaussian", help="'Gaussian' or 'Poisson'."),
    make_option("--minsize", default=10, help="Minimum gene set size. Default 10."),
    make_option("--maxsize", default=500, help="Maximum gene set size. Default 100."),
    make_option("--cores", default=1, help="number of CPU cores to use. Default 1.")
    )

parser <- OptionParser(usage="fit_gsva.R DATA_HDF PATHWAY_JSON OUTPUT_HDF [OPTS]",
                       option_list=option_list)

print("PARSING ARGS")

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
omic_type <- opts$omic_type
kcdf <- opts$kcdf
minsize <- opts$minsize
maxsize <- opts$maxsize
cores <- opts$cores

pargs <- arguments$args
data_hdf <- pargs[1]
pwy_json <- pargs[2]
output_hdf <- pargs[3]


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

mrnaseq_cols <- (feature_assays == "mrnaseq")
omic_data <- omic_data[,mrnaseq_cols]
feature_genes <- feature_genes[mrnaseq_cols]
rownames(omic_data) <- instances
colnames(omic_data) <- feature_genes

#######################################
# HANDLE MISSING VALUES

# Only keep genes that are observed for >95% of samples
omic_data <- omic_data[,colSums(is.nan(omic_data)) < 0.05*dim(omic_data)[1]]
print(dim(omic_data))

## Then impute missing data with random forests
#imp_obj <- missForest(omic_data)
#omic_data <- imp_obj$ximp

# Never mind: missForest is too costly, we'll
# stick to mean imputation for now.
cm <- colMeans(omic_data, na.rm=TRUE)
for(i in 1:length(cm)){
    omic_data[is.nan(omic_data[,i]),i] <- cm[i]
} 

############################################
# RUN GSVA 

# Call GSVA with the given parameter settings.
curried_gsva <- function(omic_data, genesets){
    cat("Running GSVA...\n")
    omic_data <- t(omic_data)
    results <- gsva(omic_data, genesets, min.sz=minsize,
                    max.sz=maxsize, kcdf=kcdf, parallel.sz=cores)
    return(t(results))
}


results <- curried_gsva(omic_data, genesets)

###########################
# SAVE OUTPUTS TO HDF5

save_to_hdf(results, output_hdf)


