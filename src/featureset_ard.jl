# Regularize a matrix Y under these
# probabilistic assumptions:
#
# tau_kj ~ Gamma(alpha, beta0*(1 .+ A_k^T S_j))
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
# We also allow `alpha` to be updated by 
# Method of Moments estimates (from tau).
# We allow the `alpha` to vary between columns
# of tau, but our updates assume they're constant
# within feature views. 
 
mutable struct FeatureSetARDReg
    featureset_ids::AbstractVector
    feature_view_ids::AbstractVector
    feature_views::Tuple
    alpha::AbstractVector
    beta0::Float32
    beta::AbstractMatrix
    S::AbstractMatrix
    A::AbstractMatrix
    A_opt::ISTAOptimiser
end

#@functor FeatureSetARDReg
Flux.trainable(r::FeatureSetARDReg) = ()

function FeatureSetARDReg(K::Integer, S::AbstractMatrix,
                          feature_views::AbstractVector,
                          featureset_ids::AbstractVector; 
                          beta0=1e-2, lr=0.05)

    n_sets, N = size(S)
    feature_view_ids = unique(feature_views)
    feature_views = Tuple(ids_to_ranges(feature_views))
    alpha = fill(beta0, N)
    beta = fill(beta0, K,N) 
    A = zeros(n_sets, K)
    l1_lambda = ones(n_sets)  #TODO: revisit lambda
    A_opt = ISTAOptimiser(A, lr, l1_lambda)
 
    return FeatureSetARDReg(featureset_ids, 
                            feature_view_ids,
                            feature_views,
                            alpha, 
                            beta0, beta,
                            S, A, A_opt) 
end


function reorder_reg!(reg::FeatureSetARDReg, p)
    reg.beta .= reg.beta[p,:]
    reg.A .= reg.A[:,p]
    reg.A_opt.ssq_grad .= reg.A_opt.ssq_grad[:,p]
    return
end


function Adapt.adapt_storage(::Flux.FluxCUDAAdaptor, r::FeatureSetARDReg)
    return FeatureSetARDReg(r.featureset_ids, 
                            r.feature_view_ids, 
                            r.feature_views,
                            gpu(r.alpha),
                            r.beta0,
                            gpu(r.beta),
                            gpu(SparseMatrixCSC{Float32,Int32}(r.S)),
                            gpu(r.A),
                            gpu(r.A_opt)
                            )
end

function Adapt.adapt_storage(::Flux.FluxCPUAdaptor, r::FeatureSetARDReg)
    return FeatureSetARDReg(r.featureset_ids,
                            r.feature_view_ids, 
                            r.feature_views,
                            cpu(r.alpha),
                            r.beta0,
                            cpu(r.beta),
                            SparseMatrixCSC{Float32,Int}(cpu(r.S)),
                            cpu(r.A),
                            cpu(r.A_opt)
                            )
end

function construct_featureset_ard(K, feature_ids, feature_views, feature_sets;
                                  featureset_names=nothing, beta0=Float32(1e-2), 
                                  lr=Float32(0.05))
    L = length(feature_sets)
    N = length(feature_ids)
    f_to_j = value_to_idx(feature_ids)

    # Construct the sparse feature set matrix
    nnz = sum(map(length, feature_sets))
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float32}(undef, nnz)
    idx = 1
    for (i, fs) in enumerate(feature_sets)
        fs_scale = 1/sqrt(length(fs))
        for f in fs
            I[idx] = i
            J[idx] = f_to_j[f]
            V[idx] = fs_scale
            idx += 1
        end
    end
    S = sparse(I, J, V, L, N)

    if featureset_names == nothing
        featureset_names = collect(1:L)
    end

    return FeatureSetARDReg(K, S, feature_views, 
                            featureset_names; 
                            beta0=beta0, lr=lr)
end 


# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(Y::AbstractMatrix)
    b = 1 .+ (Float32(0.5) ./ reg.beta).*(Y.*Y)
    return sum(transpose(0.5 .+ reg.alpha) .* sum(log.(b), dims=1))
end


function ChainRulesCore.rrule(reg::FeatureSetARDReg, Y::AbstractMatrix)

    b = 1 .+ (Float32(0.5) ./ reg.beta).*(Y.*Y)
    
    function featureset_ard_pullback(loss_bar)
        return NoTangent(), transpose((loss_bar .* reg.alpha) .+ Float32(0.5)) .* Y ./ (b .* reg.beta)
    end

    return sum(transpose(Float32(0.5) .+ reg.alpha) .* sum(log.(b), dims=1)), featureset_ard_pullback
end


# Update `alpha` via the closed form estimator of Ye and Chen (2017)
# (https://doi.org/10.1080%2F00031305.2016.1209129) 
# **on the posterior means of tau**
# TODO replace this with a more direct estimate from Y.
function update_alpha!(reg::FeatureSetARDReg, Y::AbstractMatrix; tau_smoothing=Float32(1e-3))

    tau = (tau_smoothing .+ Float32(0.5)) ./ (tau_smoothing .+ (Float32(0.5).*Y.*Y))

    # Compute view-wise quantities from tau
    view_mean_t = map(r->mean(tau[:,r]), reg.feature_views)
    println("MEAN TAU")
    println(view_mean_t)
    view_var_t = map(r->var(tau[:,r]), reg.feature_views)
    println("VAR TAU")
    println(view_var_t)

    view_mean_lt = map(r->mean(log.(tau[:,r] .+ Float32(1e-9))), reg.feature_views)
    view_mean_tlt = map(r->mean(tau[:,r].*log.(tau[:,r] .+ Float32(1e-9))), reg.feature_views)

    # Compute new view-wise alphas; assign to model
    view_alpha = view_mean_t ./(view_mean_tlt .- view_mean_t .* view_mean_lt)
    for (i,r) in enumerate(reg.feature_views)
        reg.alpha[r] .= view_alpha[i]
    end
    println("FSARD: alpha")
    println(reg.alpha)
 
    beta0 = reg.beta0
    println("FSARD: beta0")
    println(beta0)

    # Replace NaNs with beta0
    nan_idx = (!isfinite).(reg.alpha)
    reg.alpha[nan_idx] .= beta0
    
    println("FSARD: finite alpha")
    println(reg.alpha)
 
    # Require alpha to be at least as large as beta0
    reg.alpha = map(x->max(x,beta0), reg.alpha)
    println("FSARD: rounded alpha")
    println(reg.alpha)
 
    # For features that do not appear in any feature sets,
    # set alpha = beta0 for an uninformative ARD prior on
    # that column of Y.
    L = size(reg.S, 1)
    feature_appearances = vec(ones_like(Y,1,L) * reg.S)
    noprior_features = (feature_appearances .== 0)
    reg.alpha[noprior_features] .= beta0
    println("FSARD: noprior-corrected alpha")
    println(reg.alpha)
end


function gamma_normal_loss(A, S, alpha, beta0, Y)
    beta = beta0 .* (1 .+ transpose(A)*S)
    alpha_p_5 = alpha .+ Float32(0.5)
    lss = -sum(transpose(alpha).*sum(log.(beta), dims=1)) .+ sum(transpose(alpha_p_5).*sum(log.(beta .+ Float32(0.5).*(Y.*Y)), dims=1))
    # Calibration term
    lss -= sum(transpose((alpha_p_5).*log.(alpha_p_5) .- (alpha).*log.(alpha)) .+ sum(log.(abs.(Y) .+ Float32(1e-9)), dims=1))
    return lss
end

function ChainRulesCore.rrule(::typeof(gamma_normal_loss), A, S, alpha, beta0, Y)

    beta = beta0.*(1 .+ transpose(A)*S)
    Y2 = Y.*Y
    alpha_p_5 = alpha .+ Float32(0.5)

    function gamma_normal_loss_pullback(loss_bar)

        grad_AtS = beta0.*((-transpose(alpha) ./ beta) .+ transpose(alpha_p_5)./(beta .+ Float32(0.5).*Y2))
        grad_A = S*transpose(grad_AtS)
        return NoTangent(), loss_bar.*grad_A,
                            NoTangent(), NoTangent(),
                            NoTangent(), NoTangent()
    end

    lss = -sum(transpose(alpha).*sum(beta, dims=1)) .+ sum(transpose(alpha_p_5).*sum(log.(beta .+ Float32(0.5).*Y2), dims=1))
    
    # Calibration term
    lss -= sum(transpose((alpha_p_5).*log.(alpha_p_5) .- (alpha).*log.(alpha)) .+ sum(log.(abs.(Y) .+ Float32(1e-9)), dims=1))

    return lss, gamma_normal_loss_pullback
end 


# Update the matrix of "activations" via 
# nonnegative projected ISTA
function update_A_inner!(reg::FeatureSetARDReg, Y::AbstractMatrix; 
                         max_epochs=1000, term_iter=50, atol=1e-5,
                         verbosity=1, print_prefix="", print_iter=100)

    A_loss = A -> gamma_normal_loss(A, reg.S, reg.alpha, reg.beta0, Y)
    reg_loss = A -> sum(reg.A_opt.lambda .* abs.(A))
    term_count = 0

    best_loss = A_loss(reg.A) + reg_loss(reg.A)
    A_best = deepcopy(reg.A)
    v_println("Iteration 0:\t Loss=", best_loss; verbosity=verbosity,
                                                 prefix=print_prefix)

    for epoch=1:max_epochs

        A_grads = Zygote.gradient(A_loss, reg.A)
        update!(reg.A_opt, reg.A, A_grads[1]) 
        new_loss = A_loss(reg.A) + reg_loss(reg.A)

        # Track the best we've seen thus far.
        # If we don't make any progress, then increment
        # the termination counter.
        if new_loss < best_loss
            loss_diff = best_loss - new_loss

            best_loss = new_loss
            A_best .= reg.A
            if loss_diff > atol 
                term_count = 0
            else
                term_count += 1
            end
        else
            term_count += 1
            v_println("Δ_loss: ", (new_loss - best_loss), ". termination counter=", term_count; verbosity=verbosity-1,
                                                                                                 prefix=print_prefix)
        end

        # Print information for this iteration
        if epoch % print_iter == 0
            v_println("Iteration ", epoch, ":\t Loss=", new_loss; verbosity=verbosity-1,
                                                                  prefix=print_prefix)
        end

        # Terminate if progress has halted 
        if term_count >= term_iter
            v_println("Criterion satisfied -- terminating.", verbosity=verbosity-1,
                                                             prefix=print_prefix)
            break
        end
    end
    if term_count < term_iter 
        v_println("Max iterations reached -- terminating.", verbosity=verbosity,
                                                            prefix=print_prefix)

    end

    reg.A .= A_best
    v_println("Final loss: ", best_loss; verbosity=verbosity, prefix=print_prefix)

    return best_loss
end


function set_lambda_max(reg::FeatureSetARDReg, Y::AbstractMatrix)
    
    A_loss = Z -> gamma_normal_loss(Z, reg.S, reg.alpha, reg.beta0, Y)

    A = zero(reg.A)
    origin_grad = Zygote.gradient(A_loss, A)[1]
    lambda_max = maximum((abs.(origin_grad)))

    return lambda_max
end


# Score A by the fraction of its columns 
# containing a nonzero entry.
function score_A(A; threshold=1e-3)
    L,K = size(A)
    #return sum(sum(A .> threshold, dims=1) .> 0)/K
    return sum(A .> threshold)/(L*K)
end


function update_A!(reg::FeatureSetARDReg, Y::AbstractMatrix; 
                   max_epochs=500, term_iter=50,
                   target_frac=0.7,
                   print_iter=100,
                   bin_search_max_iter=20,
                   bin_search_frac_atol=0.1,
                   bin_search_lambda_atol=1e-2,
                   verbosity=1, print_prefix="",
                   history=nothing)

    n_pref = string(print_prefix, "    ")

    v_println("Updating A and λ_A..."; verbosity=verbosity, prefix=print_prefix)
    
    i = 1
    function eval_func(lambda)
        v_println("(", i,") Updating A with λ_A=", lambda; verbosity=verbosity-1, prefix=print_prefix)
        reg.A_opt.lambda .= lambda
        reg.A .= 0 # Ditch the warm-start -- it seems to get stuck! 
        update_A_inner!(reg, Y; max_epochs=max_epochs,
                                term_iter=term_iter,
                                verbosity=verbosity-1,
                                print_iter=print_iter, 
                                print_prefix=n_pref)
        score = -score_A(reg.A)
        v_println("Fraction=", -score; verbosity=verbosity-1, prefix=print_prefix)
        i += 1 # Some statefulness... `reg`, and the number of iterations 
        return score 
    end

    # Search for lambda that yields the desired score.
    # This updates `reg` along the way.
    lambda_start = mean(reg.A_opt.lambda)
    lambda, score = func_binary_search(lambda_start, -target_frac, eval_func; max_iter=bin_search_max_iter,
                                                                              z_atol=bin_search_frac_atol,
                                                                              x_atol=bin_search_lambda_atol)
    score *= -1
    history!(history; name="update_A", lambda_A=lambda, A_frac=score)

    v_println("Finished updating A; selected λ_A=", lambda, "; A_fraction=", score; verbosity=verbosity, prefix=print_prefix)
 
    reg.beta =  reg.beta0 .* (1 .+ transpose(reg.A)*reg.S)
end


