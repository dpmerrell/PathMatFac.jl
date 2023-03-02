


# The `apply!` functions for Flux optimisers don't 
# handle *views* of parameters unless we make them do so.
function Flux.Optimise.apply!(o::Flux.Optimise.AdaGrad, p::SubArray, g)
    η = o.eta
    acc_full = get!(() -> fill!(similar(p.parent), o.epsilon), o.acc, p.parent)::typeof(p.parent)
    acc_view = view(acc_full, p.indices...)
    @. acc_view += g * conj(g)
    @. g *= η / (√acc_view + o.epsilon)
    return g
end


##########################################################
# A projected AdaGrad optimizer wrapped in an Iterated 
# Shrinkage-Thresholding Algorithm (ISTA) rule. 
# 
# This imposes nonnegativity constraints on the parameters
# and applies L1-regularization.
#
# We designed this for a specific case where we're
# optimizing a function of a single array. 

mutable struct ISTAOptimiser <: Flux.Optimise.AbstractOptimiser
    lr::Float32
    ssq_grad::AbstractArray
    lambda::AbstractArray
end

function ISTAOptimiser(target::AbstractArray, lr::Number, l1_lambda)
    ssq_grad = zero(target) .+ 1e-8
    return ISTAOptimiser(lr, ssq_grad, l1_lambda)
end

# ISTA projection rule for L1 regularization. 
function ist_proj!(X_new, alpha)
    X_new .= max.(abs.(X_new) .- alpha, 0.0)
end

# It makes most sense to define ISTA's behavior in 
# `update!`, rather than `apply!`.
function Flux.Optimise.update!(ist::ISTAOptimiser, p, g)

    # Compute the (current) step size for each
    # parameter
    ist.ssq_grad .+= (g.*g)
    eta = ist.lr ./ sqrt.(ist.ssq_grad)

    # Apply the gradient update.
    # Impose the nonnegativity constraint.
    p .-= (eta.*g)
    p .= max.(p, 0.0)
        
    # Apply the ISTA projection.
    # The threshold for each parameter
    # is the product (L1 lambda)*(step size)
    ist_proj!(p, ist.lambda .* eta)
end


