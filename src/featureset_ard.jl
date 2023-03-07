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
# This can be incorporated into a block-coordinate descent
# procedure that alternates between (1) regularizing Y and
# (2) fitting the matrix A on the matrix Y.
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
    beta::AbstractMatrix
    S::AbstractMatrix
    A::AbstractMatrix
    A_opt::ISTAOptimiser
end

@functor FeatureSetARDReg
Flux.trainable(r::FeatureSetARDReg) = ()

function FeatureSetARDReg(K::Integer, S::AbstractMatrix,
                          feature_views::AbstractVector; 
                          beta0=1e-6, lr=0.05)

    n_sets, N = size(S)
    feature_view_ids = unique(feature_views)
    feature_views = Tuple(ids_to_ranges(feature_views))
    alpha = fill(beta0, N)
    scale = ones(N)
    beta = fill(beta0, K,N) 
    A = zeros(n_sets, K)
    l1_lambda = ones(n_sets)  #TODO: revisit lambda
    A_opt = ISTAOptimiser(A, lr, l1_lambda)
 
    return FeatureSetARDReg(feature_view_ids,
                            feature_views,
                            alpha, 
                            scale, beta0, beta,
                            S, A, A_opt) 
end


function construct_featureset_ard(K, feature_ids, feature_views, feature_sets;
                                  beta0=1e-6, lr=0.05)
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
                            beta0=beta0, lr=lr)
end 


# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(Y::AbstractMatrix)
    b = 1 .+ (0.5 ./ reg.beta).*(Y.*Y)
    return sum(transpose(0.5 .+ reg.alpha) .* sum(log.(b), dims=1))
end


function ChainRulesCore.rrule(reg::FeatureSetARDReg, Y::AbstractMatrix)

    b = 1 .+ (0.5 ./ reg.beta).*(Y.*Y)
    
    function featureset_ard_pullback(loss_bar)
        return NoTangent(), transpose((loss_bar .* reg.alpha) .+ 0.5) .* Y ./ (b .* reg.beta)
    end

    return sum(transpose(0.5 .+ reg.alpha) .* sum(log.(b), dims=1)), featureset_ard_pullback
end


# Update `alpha` via Method of Moments (on the posterior means of tau, given Y);
# and update `scale` to be consistent with alpha.
# TODO replace this with a more direct estimate from Y.
function update_alpha_scale!(reg::FeatureSetARDReg, Y::AbstractMatrix)

    tau = (reg.beta0 .+ 0.5) ./ (reg.beta0 .+ (0.5.*Y.*Y))

    # Compute view-wise means for tau
    view_means = map(r->mean(tau[:,r]), reg.feature_views)
    view_means_sq = view_means.*view_means

    # Compute view-wise variances for tau
    viewsq_means = map(r->mean(tau[:,r].^2), reg.feature_views) 
    view_var = viewsq_means .- view_means_sq

    # Compute new view-wise alpha; assign to model
    view_alpha = cpu(view_means_sq./view_var)
    for (i,r) in enumerate(reg.feature_views)
        reg.alpha[r] .= view_alpha[i]
    end

    # For now, set feature-wise `scale` uniformly to one.
    # TODO replace this with something more principled
    reg.scale .= 1
end


function gamma_normal_loss(A, S, alpha, beta0, scale, Y)
    beta = transpose(scale).*(beta0 .+ transpose(A)*S)
    return -sum(transpose(alpha).*sum(log.(beta), dims=1)) .+ sum(transpose(alpha.+0.5).*sum(log.(beta .+ 0.5.*(Y.*Y)), dims=1))
end

function ChainRulesCore.rrule(::typeof(gamma_normal_loss), A, S, alpha, beta0, scale, Y)

    beta = beta0 .+ transpose(A)*S
    Y2 = Y.*Y

    function gamma_normal_loss_pullback(loss_bar)

        grad_AtS = (-transpose(alpha) ./ beta) .+ transpose(alpha .+ 0.5)./(beta .+ Y2./(2 .* transpose(scale)))
        grad_A = S*transpose(grad_AtS)
        return NoTangent(), loss_bar.*grad_A,
                            NoTangent(), NoTangent(),
                            NoTangent(), NoTangent(), 
                            NoTangent()
    end

    lss = -sum(transpose(alpha).*sum(log.(transpose(scale).*beta), dims=1)) .+ sum(transpose(alpha .+ 0.5).*sum(log.(transpose(scale).*beta .+ 0.5.*Y2), dims=1))

    return lss, gamma_normal_loss_pullback
end 


# Update the matrix of "activations" via 
# nonnegative projected ISTA
function update_A_inner!(reg::FeatureSetARDReg, Y::AbstractMatrix; 
                         max_epochs=1000, atol=1e-5, rtol=1e-5,
                         verbosity=1, print_prefix="", print_iter=100,
                         term_iter=3)

    A_loss = A -> gamma_normal_loss(A, reg.S, reg.alpha, reg.beta0, reg.scale, Y)
    cur_loss = Inf
    term_count = 0

    for epoch=1:max_epochs
        new_loss, A_grads = Zygote.withgradient(A_loss, reg.A)
        update!(reg.A_opt, reg.A, A_grads[1]) 
        
        new_loss += sum(reg.A_opt.lambda .* abs.(reg.A))

        loss_diff = abs(new_loss - cur_loss)

        if epoch % print_iter == 0
            v_println("Iteration ", epoch, ":\t Loss=", new_loss; verbosity=verbosity,
                                                                  prefix=print_prefix)
        end

        # If progress is too small (or in the wrong direction), then 
        # increment the termination counter
        if (loss_diff < atol) | (abs(loss_diff/new_loss) < rtol)
            term_count += 1
            v_println("Small loss diff: ", loss_diff, ". termination counter=", term_count; verbosity=verbosity,
                                                                                            prefix=print_prefix)
        else # Reset the counter if we start making more progress
            term_count = 0
        end

        if term_count >= term_iter
            v_println("Convergence reached -- terminating.", verbosity=verbosity,
                                                             prefix=print_prefix)
            break
        end
        cur_loss = new_loss
    end
end


function set_lambda_max(reg::FeatureSetARDReg, Y::AbstractMatrix)
    
    A_loss = Z -> gamma_normal_loss(Z, reg.S, reg.alpha, reg.beta0, reg.scale, Y)

    A = zero(reg.A)
    origin_grad = Zygote.gradient(A_loss, A)[1]
    lambda_max = quantile(vec((abs.(origin_grad))), 0.95)

    return lambda_max
end


# Score A by the fraction of its columns 
# containing a nonzero entry.
function score_A(A; threshold=1e-6)
    K = size(A,2)
    return sum(sum(A .> threshold, dims=1) .> 0)/K
end


function update_A!(reg::FeatureSetARDReg, Y::AbstractMatrix; 
                   max_epochs=1000, atol=1e-5, rtol=1e-5,
                   n_lambda=10, lambda_min_ratio=1e-3,
                   score_threshold=0.7,
                   print_iter=100,
                   verbosity=1, print_prefix="")

    n_pref = string(print_prefix, "    ")

    # Compute the sequence of lambdas
    lambda_max = set_lambda_max(reg, Y)
    lambda_min = lambda_max*lambda_min_ratio
    lambdas = exp.(collect(LinRange(log(lambda_max), 
                                    log(lambda_min), 
                                    n_lambda)))
 
    # For each lambda:
    for lambda in lambdas

        v_println("Updating A with λ_A = ", lambda; verbosity=verbosity, prefix=print_prefix)
        reg.A_opt.lambda .= lambda 
        update_A_inner!(reg, Y; max_epochs=max_epochs,
                                atol=atol, rtol=rtol,
                                verbosity=verbosity,
                                print_iter=print_iter, 
                                print_prefix=n_pref)
    
        # If we pass the threshold, then
        # we're finished!    
        if score_A(reg.A) >= score_threshold
            v_println("Finished updating A; selected λ_A = ", lambda; verbosity=verbosity, prefix=print_prefix)
            break
        end
    end    
   
    reg.beta = transpose(reg.scale) .* (reg.beta0 .+ transpose(reg.A)*reg.S)

end


