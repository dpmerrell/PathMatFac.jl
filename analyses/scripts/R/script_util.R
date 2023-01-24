

load_survival_data <- function(hdf_path){
    # Load features as dataframe
    data <- h5read(hdf_path, "X")
    rows <- h5read(hdf_path, "instances")
    rownames(data) <- rows
    data <- data.frame(data)

    # Load labels
    labels <- h5read(hdf_path, "target")
    dtd <- as.numeric(labels[,1])
    dtlf <- as.numeric(labels[,2])

    # Store "is_dead" and "t_final" in dataframe
    is_alive <- is.nan(dtd)
    is_dead <- !is_alive

    data[["is_dead"]] <- is_dead
    data[["t_final"]] <- -1.0
    data[["t_final"]][is_alive] <- dtlf[is_alive]
    data[["t_final"]][is_dead] <- dtd[is_dead]

    return(data)
}


pwys_to_genesets <- function(pwys, pwy_names){

    result <- list()

    for(i in 1:length(pwys)){

        pwy <- pwys[[i]]
        pwy_name <- pwy_names[[i]]

        a_vec <- sapply(pwy, function(v) v[[1]])
        b_vec <- sapply(pwy, function(v) v[[3]])
        flag_vec <- sapply(pwy, function(v) substr(v[[2]],1,1) == "a")

        a_vec <- a_vec[flag_vec]
        b_vec <- b_vec[flag_vec]
        all_genes <- c(a_vec, b_vec)
        unq_genes <- unique(all_genes)

        result[[as.character(i)]] <- unq_genes 
    }
    return(result)
}


#get_omic_data <- function(all_data, all_features, omic_type){
#
#    splt_features <- strsplit(all_features, "_")
#    feature_names <- sapply(splt_features, function(s) s[[1]])
#    feature_types <- sapply(splt_features, function(s) s[[length(s)]])
#  
#    relevant_features <- feature_names[feature_types == omic_type]
#    relevant_data <- all_data[,feature_types == omic_type] 
#
#    colnames(relevant_data) <- relevant_features
#
#    return(relevant_data) 
#}
#
#
#load_clinical_hdf <- function(path){
#    data <- h5read(path, "data")
#    cols <- h5read(path, "index")   # Rows and columns got mixed up
#    rows <- h5read(path, "columns") # during preprocessing...
#    
#    colnames(data) <- cols
#    rownames(data) <- rows
#
#    return data
#}
#
#
#save_to_hdf <- function(results_matrix, output_hdf){
#
#    h5write(results_matrix, output_hdf, "scores")
#    h5write(colnames(results_matrix), output_hdf, "instances")
#    h5write(rownames(results_matrix), output_hdf, "pathways")
#
#}


