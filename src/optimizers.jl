

####################################################
# An optimizer that performs _truncated_ updates on 
# a model's L1-regularized parameters.
# 
# Whenever the "ordinary" update rule would change the sign of
# an L1-regularized parameter, the truncated rule 
# instead sets that parameter to zero.
#
# The update rule is a simpler version of the scheme described by 
# Langford, Li, and Zhang (2009).


mutable struct TruncatedOptimiser <: Flux.Optimise.AbstractOptimiser
    opt::Flux.Optimise.AbstractOptimiser
    l1_params::IdDict
end


function TruncatedOptimiser(pairs::Vector; inner_opt=nothing)
    if inner_opt == nothing
        inner_opt = Flux.Optimise.AdaGrad()
    end
    l1_params = IdDict()
    for pair in pairs
        l1_params[pair[1]] = pair[2]
    end
    return TruncatedOptimiser(inner_opt, l1_params)
end


function Flux.Optimise.apply!(trunc_opt::TruncatedOptimiser, p, g)

    # `d` is the _negative_ update. I.e., a vector in the 
    # direction of the gradient.
    delta = Flux.Optimise.apply!(trunc_opt.opt, p, g)
 
    if p in keys(trunc_opt.l1_params)
        trunc_idx = similar(trunc_opt.l1_params[p])
        trunc_idx .= true 
        trunc_idx .*= trunc_opt.l1_params[p]
        trunc_idx .*= (sign.(p) .== sign.(delta))
        trunc_idx .*= (abs.(p) .< abs.(delta))
        
        delta[trunc_idx] .= p[trunc_idx] 
    end

    return delta
end

