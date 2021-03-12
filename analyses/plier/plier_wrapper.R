
library("PLIER")
library("jsonlite")
library("optparse")


load_sifs <- function(sif_files){
  results = list()
  for(f in sif_files){
     results[[f]] <- read.table(sif_files, sep="\t")
  }
  return(results)
}


get_proteins <- function(sif_data){
    proteins <- c()
    for(table in sif_data){
        l_proteins <- as.character(table[startsWith(as.character(table$V2), "protein"),1])
        r_
        
        proteins <- c(proteins, l_proteins)
  }
  return(unique(proteins))
}