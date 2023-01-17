

pwys_to_genesets <- function(pwys, pwy_names){

    result <- list()

    for(i in 1:length(pwys)){

        pwy <- pwys[[i]]
        pwy_name <- pwy_names[[i]]
        gs <- list()

        for(edge in pwy){
            tag <- edge[[2]]
            if(substr(tag,1,1) == "p"){
                gs <- append(gs, edge[[1]])
            }else if(substr(tag,2,2) == "p"){
                gs <- append(gs, edge[[3]])
            }
        }

        gs <- unique(gs)
        result[[pwy_name]] <- unlist(gs)
    }
    return(result)
}


get_omic_data <- function(all_data, all_features, omic_type){

    splt_features <- strsplit(all_features, "_")
    feature_names <- sapply(splt_features, function(s) s[[1]])
    feature_types <- sapply(splt_features, function(s) s[[length(s)]])
  
    relevant_features <- feature_names[feature_types == omic_type]
    relevant_data <- all_data[,feature_types == omic_type] 

    colnames(relevant_data) <- relevant_features

    return(relevant_data) 
}


load_clinical_hdf <- function(path){
    data <- h5read(path, "data")
    cols <- h5read(path, "index")   # Rows and columns got mixed up
    rows <- h5read(path, "columns") # during preprocessing...
    
    colnames(data) <- cols
    rownames(data) <- rows

    return data
}


save_to_hdf <- function(results_matrix, output_hdf){

    h5write(results_matrix, output_hdf, "scores")
    h5write(colnames(results_matrix), output_hdf, "instances")
    h5write(rownames(results_matrix), output_hdf, "pathways")

}


