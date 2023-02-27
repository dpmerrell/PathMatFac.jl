# Regularize a matrix Y under these
# probabilistic assumptions:
#
# tau_kj ~ Gamma(alpha, beta .+ A_k^T S_j)
# Y_kj ~ Normal(0, 1 / tau_kj)
# 
# I.e., we assume that the variances of the
# entries of `Y` are controlled by linear combinations
# of feature sets `S`, with weights given by `A`.
#
# The idea is to use this in an 
# Expectation-Maximization procedure.
# I.e.:  alternate between: 
#     * updating `A` given fixed tau ("maximization"); 
#     * updating `tau` given fixed `A` and `Y` ("expectation");
#     * updating `Y` given fixed tau ("maximization")
 
mutable struct FeatureSetARDReg
    alpha0::Float32
    beta0::Float32
    tau::AbstractMatrix
    S::AbstractMatrix
    A::AbstractMatrix
    opt::Flux.Optimise.AbstractOptimiser
end

@functor FeatureSetARDReg


function FeatureSetARDReg(K::Integer, S::AbstractMatrix;
                          alpha0=0.5, beta0=0.5, lr=0.1)
    tau = zeros(K,N)
    n_sets, N = size(S)
    A = zeros(n_sets, K)
    l1_lambda = sqrt.(sum(S; dims=2)./N)
    opt = ISTAOptimiser(A, lr, l1_lambda)
 
    return FeatureSetARDReg(alpha0, beta0, zeros(K,N),
                            S, A, opt) 
end


function construct_featureset_ard(K, feature_ids, feature_sets;
                                     alpha0=0.5, beta0=0.5,
                                     lr=0.1, l1_lambda=nothing)
    L = length(feature_sets)
    N = length(featureset_ids)
    f_to_j = value_to_idx(feature_ids)

    # Construct the sparse feature set matrix
    nnz = sum(map(length, feature_sets))
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Bool}(undef, nnz)
    idx = 1
    for (i, fs) in enumerate(feature_sets)
        for f in fs
            I[idx] = i
            J[idx] = f_to_j[f]
            V[idx] = true
            idx += 1
        end
    end
    S = sparse(I, J, V, L, N)

    return FeatureSetARDReg(K, S; alpha0=alpha0, beta0=beta0,
                                  lr=lr, l1_lambda=l1_lambda)
end 


# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(Y::AbstractMatrix)
    return 0.5*sum(reg.tau .* (Y.*Y))
end

function ChainRulesCore.rrule(reg::FeatureSetARDReg, Y::AbstractMatrix)

    g = 0.5.* (reg.tau .* Y)

    function featureset_ard_pullback(loss_bar)
        return NoTangent(), loss_bar.*g
    end

    return sum(g.*Y), featureset_ard_pullback
end


# Update tau via posterior expectation.
function update_tau!(reg::FeatureSetARDReg, Y::AbstractMatrix)
    reg.tau .= (reg.alpha0 + 0.5) ./(reg.beta0 .+ transpose(reg.S)*reg.A .+ 0.5.*(Y.*Y))
end


function gamma_loss(A, S, tau, alpha0, beta0)
    AtS = beta0 .+ transpose(A)*S
    return sum(-alpha0.*log.(AtS) .+ (AtS).*tau)
end


# Update the matrix of "activations" via 
# nonnegative projected ISTA
function update_A_inner!(reg::FeatureSetARDReg; max_epochs=1000, 
                                                atol=1e-5, rtol=1e-5)

    A_loss = A -> gamma_loss(A, reg.S, reg.tau, 
                                reg.alpha0, reg.beta0)
    total_lss = Inf
    for epoch in max_epochs
        new_lss, A_grads = Zygote.withgradient(A_loss, reg.A)
        update!(reg.opt, reg.A, A_grads[1]) 
        
        new_lss += sum(reg.lambda .* abs.(reg.A))

        loss_diff = abs(new_lss - total_lss)
        if (loss_diff < atol) | (loss_diff/new_lss < rtol)
            break
        end
        total_lss = new_lss
    end
end


function update_A!(reg::FeatureSetARDReg; max_epochs=1000,
                                          atol=1e-5, rtol=1e-5,
                                          n_lambda=10, lambda_min_ratio=1e-3)

    # Set lambda_max 
    # Compute the sequence of lambdas
    # Keep track of the best A and best score
    # For each lambda: 
        # Set reg.lambda
        # fit reg.A via `update_A_inner!`
        # Score A
        # If score is best, update best A and best score
    
    # Set reg.A to be the best A 

end

