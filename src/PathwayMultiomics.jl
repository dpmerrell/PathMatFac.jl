
module PathwayMultiomics

println("LOADING PATHWAYMULTIOMICS PACKAGE")

include("preprocess.jl")
include("rrglrm.jl")
include("row_regularizer.jl")

include("rr_objective.jl")

include("graph_util.jl")

if Threads.nthreads() > 1
  include("rr_proxgrad_multithread.jl")
else
  include("rr_proxgrad.jl")
end

end # module
