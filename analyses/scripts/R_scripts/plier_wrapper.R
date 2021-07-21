
library("rhdf5")
library("PLIER")
library("jsonlite")
library("optparse")


get_suffix <- function(split_name){
    return(split_name[[length(split_name)]])
}


get_mrnaseq_idx <- function(feature_vec){
    spl <- strsplit(feature_vec, "_")
    suff <- lapply(spl, get_suffix)
    mrnaseq_idx <- which(suff == "mrnaseq")
    return(mrnaseq_idx)
}


get_gene <- function(split_name){
    return(split_name[[1]])
}


get_prot_idx <- function(genes, proteins){
    #spl <- strsplit(feats, "_")
    #gene <- lapply(spl, get_gene)
    prot_idx <- which(genes %in% proteins)
    return(prot_idx)
}


#get_train_idx <- function(data_patients, train_patients){
#
#    pat_to_idx <- list()
#    for(i in 1:length(data_patients)){
#        pat_to_idx[[data_patients[i]]] <- i
#    }
#    
#    train_idx <- list()
#    for(i in 1:length(train_patients)){
#        train_idx[i] <- pat_to_idx[[train_patients[i]]]
#    }
#
#    return(train_idx)
#}


get_mrnaseq_data <- function(hdf_file, train_idx, proteins){

    feature_names <- h5read(hdf_file, "features")
    data_genes <- lapply(strsplit(feature_names, "_"), get_gene)
    instance_names <- h5read(hdf_file, "instances")
    train_instances <- instance_names[train_idx]

    #train_idx <- get_train_idx(instance_names, train_patients)

    mrnaseq_idx <- get_mrnaseq_idx(feature_names)
    mrnaseq_feat <- data_genes[mrnaseq_idx]

    #print("GOT THIS FAR")
    prot_idx <- get_prot_idx(mrnaseq_feat, proteins)
    prot_mrnaseq_idx <- mrnaseq_idx[prot_idx]
    prot_mrnaseq_feat <- mrnaseq_feat[prot_idx]
    #print("PROT MRNASEQ IDX:")
    #print(prot_mrnaseq_idx)

    myh5 <- H5Fopen(hdf_file)
    dataset <- myh5$"/data"
    #print("DATASET (ROWS, COLS)")
    print(nrow(dataset))
    print(ncol(dataset))

    #print("GOT A LITTLE FARTHER")
    mrnaseq_data <- dataset[prot_mrnaseq_idx, train_idx]
    print(nrow(mrnaseq_data))
    print(ncol(mrnaseq_data))

    #print("GOT A LITTLE FARTHER")
    mrnaseq_df <- as.data.frame(mrnaseq_data, row.names=prot_mrnaseq_feat)
    #print("BUILT THE DF")
    colnames(mrnaseq_df) <- train_instances
    #print("ADDED COLUMN NAMES")
    return(mrnaseq_df)

}


get_proteins <- function(pwy_ls){
    all_proteins = list()
    idx <- 1
    for(pwy in pwy_ls){
        for(edge in pwy){
            u <- edge[[2]]
            if(substr(u, 1, 1) == "p"){
                all_proteins[[idx]] <- edge[[1]]
                idx <- idx + 1
            }
            if(substr(u, 2, 2) == "p"){
                all_proteins[[idx]] <- edge[[3]]
                idx <- idx + 1
            }
        }
    }
    all_proteins <- unique(all_proteins)
    return(all_proteins)
}


pwy_to_geneset <- function(pwy, proteins){

    result <- rep(0,length(proteins))
    prot_to_idx <- list()
    for(i in 1:length(proteins)){
        prot_to_idx[[proteins[[i]]]] <- i
    }

    this_pwy_proteins <- get_proteins(list(pwy))
    
    # for each protein in this pathway,
    # set the corresponding entry to 1 
    for(prot in this_pwy_proteins){
        result[prot_to_idx[[prot]]] <- 1
    } 

    return(result)
}


pwys_to_genesets <- function(pwys, pwy_names){

    # Get the full set of proteins
    proteins <- get_proteins(pwys)
   
    # Initialize the result dataframe
    result <- data.frame(row.names=proteins) 

    # Add each pathway to the dataframe
    for(i in 1:length(pwys)){
        pwy <- pwys[[i]]
        name <- pwy_names[[i]]
        result[[name]] <- pwy_to_geneset(pwy, proteins)
    }
    return(result)
}


# INPUTS:
# * Pathways
# * Omic data HDF file
# 
main <- function(){
    #opts <- list(
    #             make_option("--alpha_a", default=0.5),
    #             make_option("--beta_a", default=0.5),
    #             make_option("--alpha_b", default=0.5),
    #             make_option("--beta_b", default=0.5),
    #             make_option("--transition_dist", type="character", default="beta_binom"),
    #             make_option("--test_statistic", type="character", default="wald"),
    #             make_option("--act_l", type="numeric", default=0.2)
    #             )

    parser <- OptionParser(usage="plier_wrapper.R OMIC_HDF SPLIT_JSON PWY_JSON OUTPUT_FILE") # [options]", option_list=option_list)

    # Parse the arguments
    arguments <- parse_args(parser, positional_arguments=4)
    opt <- arguments$options
    args <- arguments$args

    omic_hdf <- args[1]
    split_json <- args[2]
    pwy_json <- args[3]
    out_file <- args[4]
    print(omic_hdf)
    print(pwy_json)

    pwy_info <- read_json(pwy_json)
    pwy_names <- pwy_info[["names"]]
    pwys <- pwy_info[["pathways"]]

    # Convert the list of pathways into 
    # a dataframe of genesets
    genesets <- pwys_to_genesets(pwys, pwy_names)
    print("GENE SETS (ROWS, COLS)")
    print(nrow(genesets)) 
    print(ncol(genesets))

    # Load the training split
    split_info = read_json(split_json, simplifyVector=TRUE)# auto_unbox=TRUE)
    train_idx = split_info[["train"]]
    print(train_idx)
 
    # Get all of the RNA seq data
    # corresponding to those proteins
    mrnaseq_df <- get_mrnaseq_data(omic_hdf, train_idx, row.names(genesets))
    #print("MRNASEQ_DF (ROWS, COLS)")
    #print(row.names(mrnaseq_df))
    #print(nrow(mrnaseq_df))
    #print(ncol(mrnaseq_df))

    print("ABOUT TO RUN PLIER")
    plier_results <- PLIER(as.matrix(mrnaseq_df), as.matrix(genesets), scale=T, trace=T)

    saveRDS(plier_results, file=out_file)

}


main()
