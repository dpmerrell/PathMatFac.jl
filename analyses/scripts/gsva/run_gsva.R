
library("GSVA")
library("jsonlite")
library("rhdf5")
library("optparse")


###########################
# PARSE ARGUMENTS
parser <- OptionParser(usage="run_gsva.R DATA_HDF GENESET_JSON OUTPUT_HDF")
arguments <- parse_args(parser, positional_arguments=3)
pargs <- arguments$args

data_hdf <- pargs[1]
geneset_json <- pargs[2]
output_hdf <- pargs[3]


###########################
# LOAD GENESETS

geneset_dict <- read_json(geneset_json)
genesets <- geneset_dict$genesets
names(genesets) <- geneset_dict$names

###########################
# LOAD OMIC DATA
#  - instance IDs
#  - feature IDs
#    * isolate the RNAseq features 
#  - omic matrix
#    * keep only the RNAseq features


###########################
# RUN GSVA ON EACH CANCER TYPE

###########################
# SAVE OUTPUTS TO HDF5

