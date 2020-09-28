
args = commandArgs(trailingOnly=TRUE)

data = readRDS(args[1])

write.csv(data, args[2], quote=FALSE)

