
library("rhdf5")
library("PLIER")
library("jsonlite")
library("optparse")

source("script_util.R")

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

keep_pwy_features <- function(omic_data, pwy_features){
    return(omic_data[,pwy_features])
}

###########################
# PARSE ARGUMENTS
option_list <- list(
    make_option("--mode", type="character", default="groupwise", help="either 'groupwise' or 'combined'. Default 'groupwise'."),
    make_option("--omic_type", type="character", default="mrnaseq", help="The type of omic data to use. Default 'mrnaseq'.")
    )


parser <- OptionParser(usage="run_plier.R DATA_HDF PATHWAY_JSON OUTPUT_HDF [--mode MODE]",
                       option_list=option_list)

arguments <- parse_args(parser, positional_arguments=3)

opts <- arguments$options
mode <- opts$mode
omic_type <- opts$omic_type

pargs <- arguments$args
data_hdf <- pargs[1]
pwy_json <- pargs[2]
output_hdf <- pargs[3]


all_data <- h5read(data_hdf, "data")
featurenames <- h5read(data_hdf, "features")
instancenames <- h5read(data_hdf, "instances")
instance_groups <- h5read(data_hdf, "groups")

omic_data <- get_omic_data(all_data, featurenames, omic_type)
rownames(omic_data) <- instancenames

pwy_dict <- read_json(pwy_json)
pwys <- pwy_dict$pathways
pwy_names <- pwy_dict$names
genesets <- pwys_to_genesets(pwys, pwy_names)
priormat <- genesets_to_priormat(genesets, colnames(omic_data)) 

# Call PLIER with the given parameter settings.
curried_plier <- function(omic_data, priormat){
    results <- PLIER(t(omic_data), priormat)
    scores <- results[["U"]] %*% results[["B"]]
    return(t(scores))
}

# Call PLIER separately on each *group* of instances.
# Then combine the results into one matrix.
groupwise_plier <- function(omic_data, genesets, groups){

   unq_groups <- unique(groups)

   result_ls <- list()

   for(gp in unq_groups){
       cat("Group: ")
       cat(gp)
       cat("\n")

       gp_data <- omic_data[groups == gp,]
       gp_results <- curried_plier(gp_data, genesets)

       cat("\tgroup results: ")
       cat(dim(gp_results))
       cat("\n")
       result_ls[[gp]] <- gp_results 
   }

   result <- do.call(rbind, result_ls)

   return(result)
}


if(mode == "combined"){

    scores <- curried_plier(omic_data, priormat)

}else if(mode == "groupwise"){

    scores <- groupwise_plier(omic_data, priormat, instance_groups)

}

save_to_hdf(scores, output_hdf)


