

library(optparse)
library(randomForestSRC)
library(rhdf5)

source("scripts/R/script_util.R")

##############################################
# Fit model 
##############################################

fit_model <- function(data, ntree){
    model <- rfsrc(Surv(t_final, is_dead)~., data,
                   n_tree=ntree)
    return(model)
}


##############################################
# RUN SCRIPT
##############################################

# Build argument parser
parser <- OptionParser(usage="Rscript fit_survival_regressor.R  train_data.hdf fitted_model.RData") 
arguments <- parse_args(parser, positional_arguments=2)

pargs <- arguments$args
train_hdf <- pargs[1]
fitted_rd <- pargs[2]

training_data <- load_survival_data(train_hdf)

ntree <- 500

fitted_model <- fit_model(training_data, ntree)

saveRDS(fitted_model, fitted_rd)



