

library(optparse)
library(survival)
library(jsonlite)
library(rhdf5)
library(randomForestSRC)

source("scripts/R/script_util.R")

################################
# Parse arguments
################################

parser <- OptionParser(usage="Rscript score_survival_regressor.R fitted_model.RData test_data.hdf scores.json") #, option_list=option_list)
arguments <- parse_args(parser, positional_arguments=3)

#opt <- arguments$options

pargs <- arguments$args
model_rd <- pargs[1]
test_hdf <- pargs[2]
scores_json <- pargs[3]

#################################
# Load data and fitted model
#################################
test_data <- load_survival_data(test_hdf)
fitted_model <- readRDS(model_rd)

####################################
# Compute concordance on test set
####################################
pred_obj <- predict.rfsrc(fitted_model, test_data)
c_index <- 1 - pred_obj$err.rate[pred_obj$ntree]

####################################
# Save results to JSON
####################################
results <- list()
results[["concordance"]] <- c_index
 
write_json(results, scores_json)
 
