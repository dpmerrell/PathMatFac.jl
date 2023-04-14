
module PathwayMultiomics

using MatFac, CUDA, SparseArrays, SuiteSparse, Statistics, StatsBase, Distributions, 
      HDF5, CSV, JSON, DataFrames, BSON, ChainRules, ChainRulesCore,
      Zygote, Flux, Functors, Adapt, LinearAlgebra, Krylov, DataStructures

MF = MatFac
CuSparseMatrixCSC = CUDA.CUSPARSE.CuSparseMatrixCSC

include("util.jl")
include("batch_array.jl")
include("layers.jl")
include("prep_pathways.jl")
include("regularizers.jl")
include("optimizers.jl")
include("featureset_ard.jl")
include("model.jl")
include("model_io.jl")
include("scores.jl")
include("callbacks.jl")
include("fit_lbfgs.jl")
include("fit.jl")
include("transform.jl")
include("simulate_params.jl")
include("remove_batch_effect.jl")
include("impute.jl")

end
