

load_survival_data <- function(hdf_path){
    # Load features as dataframe
    mydata <- h5read(hdf_path, "X")
    myrows <- h5read(hdf_path, "instances")
    rownames(mydata) <- myrows
    mydata <- data.frame(mydata)
    # Load labels
    labels <- h5read(hdf_path, "target")
    dtd <- as.numeric(labels[,1])
    dtlf <- as.numeric(labels[,2])

    # Store "is_dead" and "t_final" in dataframe
    is_alive <- is.nan(dtd)
    is_dead <- !is_alive

    mydata[["is_dead"]] <- is_dead
    mydata[["t_final"]] <- -1.0
    mydata[["t_final"]][is_alive] <- dtlf[is_alive]
    mydata[["t_final"]][is_dead] <- dtd[is_dead]

    return(mydata)
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


median_impute <- function(omic_data){
    cmeds <- apply(omic_data, 2, function(v) median(v, na.rm=TRUE))
    for(i in 1:length(cmeds)){
        omic_data[is.nan(omic_data[,i]),i] <- cmeds[i]
    } 
    return(omic_data)
}


linear_transform <- function(Z, Y){

    K <- nrow(Y)
    N <- ncol(y)
    M <- nrow(Z)

    X <- matrix(0, K, M)
    grad_X <- matrix(0, K, M) 
    grad_ssq = matrix(0, K, M) + 1e-8

    nan_idx <- is.nan(Z) 
    lss <- Inf
    i <- 0

    # Apply Adagrad updates until convergence...
    while(i < max_iter){
        new_lss <- 0.0
            
        # Compute the gradient of squared loss w.r.t. X
        delta <- (X.transpose() %*% Y) - Z
        delta[nan_idx] <- 0.0
        grad_X <- Y %*% delta.transpose()
  
        # Update the sum of squared gradients
        grad_ssq <- grad_ssq + grad_X*grad_X

        # Apply the update
        X <- lr*(grad_X / sqrt(grad_ssq))
      
        # Compute the loss 
        delta <- delta*delta 
        new_lss <- new_lss + np.sum(delta)

        # Check termination criterion
        if( (lss - new_lss)/lss < rel_tol ){
            print(paste("Loss decrease < rel tol (",rel_tol,"). Terminating", sep=""))
            break
        }

        # Update loop variables
        lss <- new_lss
        i <- i + 1
        print(paste("Iteration: ", i, "; Loss: ", lss, sep=""))
    }
    if(i >= max_iter){
        print(paste("Reached max iter (",max_iter,"). Terminating", sep=""))
    }

    return(X)
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


