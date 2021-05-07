
module PathwayMultiomics


include("model.jl")
include("losses.jl")
include("regularizer.jl")
include("graph_util.jl")
include("line_search.jl")

if CUDA.has_cuda()
    include("fit_cuda.jl")
    include("transform_cuda.jl")
else
    include("fit.jl")
    include("transform.jl")
end


BLAS.set_num_threads(1)


end # module
