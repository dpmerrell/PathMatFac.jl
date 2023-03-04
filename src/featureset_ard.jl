# Regularize a matrix Y under these
# probabilistic assumptions:
#
# tau_kj ~ Gamma(alpha, scale*(beta_0 .+ A_k^T S_j))
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
#     * updating `Y` given fixed `tau` ("maximization")
# 
# We also allow `alpha` and `scale` to be updated by 
# Method of Moments estimates (from tau).
# We allow the `alpha` and `scale` to vary between columns
# of tau, but our updates assume they're constant
# within feature views. 
 
mutable struct FeatureSetARDReg
    feature_view_ids::AbstractVector
    feature_views::Tuple
    alpha::AbstractVector
    scale::AbstractVector
    beta0::Float32
    tau::AbstractMatrix
    tau_max::Float32
    S::AbstractMatrix
    A::AbstractMatrix
    A_opt::Flux.Optimise.AbstractOptimiser
end

@functor FeatureSetARDReg
Flux.trainable(r::FeatureSetARDReg) = ()

function FeatureSetARDReg(K::Integer, S::AbstractMatrix,
                          feature_views::AbstractVector; 
                          beta0=1e-6, lr=0.1, tau_max=1e6)

    n_sets, N = size(S)
    feature_view_ids = unique(feature_views)
    feature_views = Tuple(ids_to_ranges(feature_views))
    alpha = fill(beta0, N)
    scale = ones(N)
    tau = zeros(K,N)
    A = zeros(n_sets, K)
    l1_lambda = sqrt.(sum(S; dims=2)./N)
    A_opt = ISTAOptimiser(A, lr, l1_lambda)
 
    return FeatureSetARDReg(feature_view_ids,
                            feature_views,
                            alpha, scale, beta0, 
                            tau, tau_max,
                            S, A, A_opt) 
end


function construct_featureset_ard(K, feature_ids, feature_views, feature_sets;
                                  beta0=1e-6, lr=0.1, tau_max=1e6)
    L = length(feature_sets)
    N = length(feature_ids)
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

    return FeatureSetARDReg(K, S, feature_views; 
                            beta0=beta0, lr=lr, tau_max=tau_max)
end 


# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(Y::AbstractMatrix)
    return 0.5*sum(reg.tau .* (Y.*Y))
end


function ChainRulesCore.rrule(reg::FeatureSetARDReg, Y::AbstractMatrix)

    g = reg.tau .* Y

    function featureset_ard_pullback(loss_bar)
        return NoTangent(), loss_bar.*g
    end

    return 0.5*sum(g.*Y), featureset_ard_pullback
end


# Update `alpha` via Method of Moments (on tau);
# and update `scale` to be consistent with alpha and tau.
function update_alpha_scale!(reg::FeatureSetARDReg; q=0.95)

    # Compute view-wise means for tau
    view_means = map(r->mean(reg.tau[:,r]), reg.feature_views)
    view_means_sq = view_means.*view_means

    # Compute view-wise variances for tau
    viewsq_means = map(r->mean(reg.tau[:,r].^2), reg.feature_views) 
    view_var = viewsq_means .- view_means_sq

    # Compute new view-wise alpha; assign to model
    view_alpha = view_means_sq./view_var
    for (i,r) in enumerate(reg.feature_views)
        reg.alpha[r] .= view_alpha[i]
    end

    # Compute view-wise maxes for tau 
    # (By default we actually use 95%-ile for improved stability)
    view_maxes = map(r->quantile(vec(reg.tau[:,r]), q), reg.feature_views) 

    # Compute view-wise scale; assign to model
    view_scale = view_alpha ./ (reg.beta0 .* view_maxes)
    for (i,r) in enumerate(reg.feature_views)
        reg.scale[r] .=  view_scale[i]
    end
end


# Update tau via posterior expectation.
function update_tau!(reg::FeatureSetARDReg, Y::AbstractMatrix)
    reg.tau .= (transpose(reg.alpha) .+ 0.5) ./(transpose(reg.scale).*(reg.beta0 .+ transpose(reg.A)*reg.S) .+ 0.5.*(Y.*Y))
    map!(x->min(x, reg.tau_max), reg.tau, reg.tau)
end


function gamma_loss(A, S, tau, alpha, beta0, scale)
    beta = transpose(scale).*(beta0 .+ transpose(A)*S)
    return sum(-transpose(alpha).*log.(beta) .+ (beta).*tau)
end


# Update the matrix of "activations" via 
# nonnegative projected ISTA
function update_A_inner!(reg::FeatureSetARDReg; max_epochs=1000, 
                                                atol=1e-5, rtol=1e-5)

    A_loss = A -> gamma_loss(A, reg.S, reg.tau, 
                                reg.alpha, reg.beta0, reg.scale)
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


function set_lambda_max(reg::FeatureSetARDReg)
    A = reg.A
    A .= 0

    A_loss = Z -> gamma_loss(Z, reg.S, reg.tau, 
                                reg.alpha, reg.beta0, reg.scale)

    origin_grad = Zygote.gradient(A_loss, A)
    lambda_max = maximum(abs.(origin_grad))

    return lambda_max
end

# Score A by the fraction of its columns 
# containing a nonzero entry.
function score_A(A)
    K = size(A,2)
    return sum(sum(A .!= 0, dims=1) .> 0, dims=2)/K
end

function update_A!(reg::FeatureSetARDReg; max_epochs=1000,
                                          atol=1e-5, rtol=1e-5,
                                          n_lambda=10, lambda_min_ratio=1e-3,
                                          score_threshold=0.7)

    # Compute the sequence of lambdas
    lambda_max = set_lambda_max(reg)
    lambda_min = lambda_max*lambda_min_ratio
    lambdas = exp.(collect(LinRange(log(lambda_max), 
                                    log(lambda_min), 
                                    n_lambda)))
 
    # For each lambda:
    for lambda in lambdas

        reg.opt.lambda .= lambda 
        update_A_inner!(reg; max_epochs=max_epochs,
                             atol=atol, rtol=rtol)
    
        # If we pass the threshold, then
        # we're finished!    
        if score_A(reg.A) >= score_threshold
            break
        end
    end    
    
end

