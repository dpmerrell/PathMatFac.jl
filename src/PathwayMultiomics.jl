
module PathwayMultiomics

println("LOADING PATHWAYMULTIOMICS PACKAGE")

include("graph_util.jl")
include("preprocess.jl")
include("rrglrm.jl")
include("rr_regularizers.jl")
include("rr_objective.jl")


if Threads.nthreads() > 1
  println("USING MULTITHREAD VERSION")
  include("rr_proxgrad_multithread.jl")
else
  println("USING SINGLE THREAD VERSION")
  include("rr_proxgrad.jl")
end

end # module
