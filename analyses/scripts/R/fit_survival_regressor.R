

library(optparse)
library(randomForestSRC)
library(rhdf5)

source("scripts/R/script_util.R")

##############################################
# Fit model 
##############################################

fit_model <- function(data){

    model <- rfsrc(Surv(t_final, is_dead)~., data,
                   n_tree=100)

    return(model)
}


##############################################
# RUN SCRIPT
##############################################

# Build argument parser
#option_list <- list(
#    make_option("--design", type="character", default="traditional"),
#    make_option("--blockraropt_db", type="character", default=""),
#    make_option("--n_blocks", type="integer", default=1),
#    make_option("--target_t1", default=0.05),
#    make_option("--target_power", default=0.8)
#)

parser <- OptionParser(usage="Rscript fit_survival_regressor.R  train_data.hdf fitted_model.RData") #, option_list=option_list)
arguments <- parse_args(parser, positional_arguments=2)

#opt <- arguments$options

pargs <- arguments$args
train_hdf <- pargs[1]
fitted_rd <- pargs[2]

training_data <- load_survival_data(train_hdf)

fitted_model <- fit_model(training_data)

saveRDS(fitted_model, fitted_rd)



