
module PathwayMultiomics

using BatchMatFac, CUDA, SparseArrays, Statistics, 
      HDF5, CSV, DataFrames, BSON, ChainRules, ChainRulesCore,
      Flux, Functors

BMF = BatchMatFac

include("typedefs.jl")
include("util.jl")
include("prep_pathways.jl")
include("assemble_model.jl")
include("network_regularizer.jl")
include("model.jl")
include("model_io.jl")
include("postprocess.jl")
include("fit.jl")
include("simulate.jl")

end
