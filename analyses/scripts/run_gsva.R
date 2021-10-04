
library("GSVA")
library("jsonlite")
library("rhdf5")
library("optparse")

source("script_util.R")


###########################
# PARSE ARGUMENTS
option_list <- list(
    make_option("--mode", type="character", default="groupwise", help="either 'groupwise' or 'combined'. Default 'groupwise'."),
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'."),
    make_option("--kcdf", type="character", default="Gaussian", help="'Gaussian' or 'Poisson'."),
    make_option("--minsize", default=10, help="Minimum gene set size. Default 10."),
    make_option("--maxsize", default=100, help="Maximum gene set size. Default 100."),
    make_option("--cores", default=1, help="number of CPU cores to use. Default 1.")
    )

parser <- OptionParser(usage="run_gsva.R DATA_HDF PATHWAY_JSON OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
mode <- opts$mode
omic_type <- opts$omic_type
kcdf <- opts$kcdf
minsize <- opts$minsize
maxsize <- opts$maxsize
cores <- opts$cores

pargs <- arguments$args
data_hdf <- pargs[1]
pwy_json <- pargs[2]
output_hdf <- pargs[3]


# Make a "mode" option with two possible settings:
# (1) "combined" or (2) "groupwise".
# Default to "combined".

###########################
# LOAD PATHWAYS

pwy_dict <- read_json(pwy_json)
pwys <- pwy_dict$pathways
pwy_names <- pwy_dict$names

genesets <- pwys_to_genesets(pwys, pwy_names)

###########################
# LOAD OMIC DATA
#  - instance IDs
#  - feature IDs
#    * isolate the RNAseq features 
#  - omic matrix
#    * keep only the RNAseq features

all_data <- h5read(data_hdf, "data")
featurenames <- h5read(data_hdf, "features")
instancenames <- h5read(data_hdf, "instances")
instance_groups <- h5read(data_hdf, "groups")

omic_data <- get_omic_data(all_data, featurenames, omic_type)
rownames(omic_data) <- instancenames

############################################
# RUN GSVA 

# Call GSVA with the given parameter settings.
curried_gsva <- function(omic_data, genesets){
    cat("Running GSVA...\n")
    results <- gsva(t(omic_data), genesets, min.sz=minsize,
                    max.sz=maxsize, kcdf=kcdf, parallel.sz=cores)
    return(t(results))
}

# Call GSVA separately on each *group* of instances.
# Then combine the results into one matrix.
groupwise_gsva <- function(omic_data, genesets, groups){

   unq_groups <- unique(groups)

   result_ls <- list()

   for(gp in unq_groups){
       cat("Group: ")
       cat(gp)
       cat("\n")

       gp_data <- omic_data[groups == gp,]
       gp_results <- curried_gsva(gp_data, genesets)

       cat("\tgroup results: ")
       cat(dim(gp_results))
       cat("\n")
       result_ls[[gp]] <- gp_results 
   }

   result <- do.call(rbind, result_ls)

   return(result)
}


if(mode == "combined"){
    results <- curried_gsva(omic_data, genesets)
}else if(mode == "groupwise"){
    results <- groupwise_gsva(omic_data, genesets, instance_groups)
}

###########################
# SAVE OUTPUTS TO HDF5

save_to_hdf(results, output_hdf)


