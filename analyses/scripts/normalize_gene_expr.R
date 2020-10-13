

library("DESeq2")

args = commandArgs(trailingOnly=TRUE)

counts = t(read.csv(args[1], row.names=1))
mode(counts) <- "integer"
#print(counts)

print(ncol(as.integer(counts)))
conditions = factor(rep("A", ncol(counts)))
print(conditions)

dds = DESeqDataSetFromMatrix(counts, DataFrame(conditions), ~conditions)
print(dds)
