
module PathwayMultiomics

using MatFac, CUDA, SparseArrays, Statistics, 
      HDF5, CSV, JSON, DataFrames, BSON, ChainRules, ChainRulesCore,
      Zygote, Flux, Functors, LinearAlgebra

MF = MatFac
CuSparseMatrixCSC = CUDA.CUSPARSE.CuSparseMatrixCSC

include("util.jl")
include("batch_array.jl")
include("layers.jl")
include("prep_pathways.jl")
include("assemble_model.jl")
include("regularizers.jl")
include("model.jl")
include("model_io.jl")
include("postprocess.jl")
include("scores.jl")
include("callbacks.jl")
include("fit.jl")
include("transform.jl")
include("simulate.jl")

end
