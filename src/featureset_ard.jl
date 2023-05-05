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

mutable struct FeatureSetARDReg
    col_ranges::Tuple  # Tuple of UnitRanges -- encodes views
    A::Tuple           # Tuple of L_v x K matrices
    S::Tuple           # Tuple of L_v x N_v sparse matrices
    alpha0::Float32    # Default 1.01
    featureset_ids::Tuple # Tuple of L_v-dim vectors containing the names of feature sets
    alpha::AbstractVector  # Holds intermediate result for fast computation
    beta::AbstractMatrix   # Holds intermediate result for fast computation
    A_opts::Tuple          # Tuple of ISTAOptimisers
end

Flux.trainable(r::FeatureSetARDReg) = ()

function FeatureSetARDReg(K::Integer, feature_views::AbstractVector, S_vec, featureset_ids_vec; 
                          alpha0=1.01, lr=0.05)

    N = length(feature_views)
    col_ranges = Tuple(ids_to_ranges(feature_views))
    for (cr, S, fid) in zip(col_ranges, S_vec, featureset_ids_vec)
        @assert length(cr) == size(S, 2) "The number of columns in each `feature_view` must match size(S, 2) for corresponding S in `S_vec`"
        @assert length(fid) == size(S, 1) "Number of featureset_ids incompatible with size(S, 2). Check `featureset_ids_vec` and `S_vec` for compatibility."
    end

    A = []
    lambda = []
    for S in S_vec
        L_v = size(S, 1)
        push!(A, zeros(Float32, L_v, K))
        push!(lambda, ones(Float32, K))
    end
    A = Tuple(A)
    lambda = Tuple(lambda)

    alpha = fill(alpha0, N)
    beta = fill(alpha0-1, K, N)

    A_opts = [ISTAOptimiser(A_mat, lr, lambda_vec) for (A_mat, lambda_vec) in zip(A, lambda)]

    return FeatureSetARDReg(col_ranges,
                            A, 
                            Tuple(S_vec),
                            alpha0, 
                            Tuple(featureset_ids_vec),
                            alpha, beta,
                            Tuple(A_opts)) 
end


function reorder_reg!(reg::FeatureSetARDReg, p)
    reg.beta .= reg.beta[p,:]
    
    for A_mat in reg.A
        reg.A .= reg.A[:,p]
    end
    
    for A_opt in reg.A_opts
        A_opt.ssq_grad .= A_opt.ssq_grad[:,p]
        A_opt.lambda .= A_opt.lambda[p]
    end

    for lambda_vec in reg.lambda
        lambda_vec .= lambda_vec[p]
    end

    return
end


function Adapt.adapt_storage(::Flux.FluxCUDAAdaptor, r::FeatureSetARDReg)
    return FeatureSetARDReg(r.col_ranges,
                            map(gpu, r.A),
                            map(S_mat -> gpu(SparseMatrixCSC{Float32,Int32}(S_mat)), r.S),
                            r.alpha0,
                            r.featureset_ids, 
                            gpu(r.alpha),
                            gpu(r.beta),
                            gpu(r.A_opts)
                            )
end

function Adapt.adapt_storage(::Flux.FluxCPUAdaptor, r::FeatureSetARDReg)
    return FeatureSetARDReg(r.col_ranges,
                            map(cpu, r.A),
                            map(S_mat -> SparseMatrixCSC{Float32,Int32}(cpu(S_mat)), r.S),
                            r.alpha0,
                            r.featureset_ids, 
                            cpu(r.alpha),
                            cpu(r.beta),
                            cpu(r.A_opts)
                            )
end



function construct_featureset_ard(K, feature_ids, feature_views, feature_sets_vec;
                                  featureset_ids=nothing, alpha0=Float32(1.01), 
                                  lr=Float32(0.05))
    col_ranges = ids_to_ranges(feature_views)    
    S_vec = []
    for (cr, feature_sets) in zip(col_ranges, feature_sets_vec) 
        push!(S_vec, featuresets_to_csc(feature_ids[cr], feature_sets))
    end
    if featureset_ids == nothing
        featureset_ids = [collect(1:length(fs)) for fs in feature_sets_vec]
    end
  
    return FeatureSetARDReg(K, feature_views, S_vec, featureset_ids; 
                            alpha0=alpha0, lr=lr)
end 


# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(Y::AbstractMatrix)
    b = 1 .+ (Float32(0.5) ./ reg.beta).*(Y.*Y)
    return sum(transpose(Float32(0.5) .+ reg.alpha) .* sum(log.(b), dims=1))
end


function ChainRulesCore.rrule(reg::FeatureSetARDReg, Y::AbstractMatrix)

    b = 1 .+ (Float32(0.5) ./ reg.beta).*(Y.*Y)
    
    function featureset_ard_pullback(loss_bar)
        return NoTangent(), transpose((loss_bar .* reg.alpha) .+ Float32(0.5)) .* Y ./ (b .* reg.beta)
    end

    return sum(transpose(Float32(0.5) .+ reg.alpha) .* sum(log.(b), dims=1)), featureset_ard_pullback
end


#function update_alpha!(reg::FeatureSetARDReg, Y::AbstractMatrix; tau_smoothing=Float32(1e-3))
#
#    tau = (tau_smoothing .+ Float32(0.5)) ./ (tau_smoothing .+ (Float32(0.5).*Y.*Y))
#
#    # Compute view-wise quantities from tau
#    view_mean_t = map(r->mean(tau[:,r]), reg.feature_views)
#    println(view_mean_t)
#    view_var_t = map(r->var(tau[:,r]), reg.feature_views)
#    println(view_var_t)
#
#    view_mean_lt = map(r->mean(log.(tau[:,r] .+ Float32(1e-9))), reg.feature_views)
#    view_mean_tlt = map(r->mean(tau[:,r].*log.(tau[:,r] .+ Float32(1e-9))), reg.feature_views)
#
#    # Compute new view-wise alphas; assign to model
#    view_alpha = view_mean_t ./(view_mean_tlt .- view_mean_t .* view_mean_lt)
#    for (i,r) in enumerate(reg.feature_views)
#        reg.alpha[r] .= view_alpha[i]
#    end
#    println(reg.alpha)
# 
#    beta0 = reg.beta0
#
#    # Replace NaNs with beta0
#    nan_idx = (!isfinite).(reg.alpha)
#    reg.alpha[nan_idx] .= beta0
#    
#    # Require alpha to be at least as large as beta0
#    reg.alpha = map(x->max(x,beta0), reg.alpha)
# 
#    # For features that do not appear in any feature sets,
#    # set alpha = beta0 for an uninformative ARD prior on
#    # that column of Y.
#    L = size(reg.S, 1)
#    feature_appearances = vec(ones_like(Y,1,L) * reg.S)
#    noprior_features = (feature_appearances .== 0)
#    reg.alpha[noprior_features] .= beta0
#end


function gamma_normal_loss(A, S, alpha, alpha0, Y)
    beta0 = alpha0 - 1
    beta = beta0 .* (1 .+ transpose(A)*S)
    alpha_p_5 = alpha .+ Float32(0.5)
    lss = -sum(transpose(alpha).*sum(log.(beta), dims=1)) .+ sum(transpose(alpha_p_5).*sum(log.(beta .+ Float32(0.5).*(Y.*Y)), dims=1))
    # Calibration term
    lss -= sum(transpose((alpha_p_5).*log.(alpha_p_5) .- (alpha).*log.(alpha)) .+ sum(log.(abs.(Y) .+ Float32(1e-9)), dims=1))
    return lss
end

function ChainRulesCore.rrule(::typeof(gamma_normal_loss), A, S, alpha, alpha0, Y)

    beta0 = alpha0 - 1
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


function update_lambda!(reg::FeatureSetARDReg, Y::AbstractMatrix)

    # For each view
    for (cr, S, A, opt) in zip(reg.col_ranges, reg.S, reg.A, reg.A_opts)

        # Compute row-wise mean-squared of Y
        Y_view = view(Y, :, cr)
        Y_ms = vec(mean(Y_view .* Y_view, dims=2))

        den = map(yv -> max(yv, Float32(1e-2)), Y_ms)
        # Compute mean of S for this view
        S_mean = mean(S)

        opt.lambda .= S_mean ./ den 
    end

end


# Update a matrix of "assignments" via 
# nonnegative projected ISTA
function update_A_inner!(A::AbstractMatrix, S::AbstractMatrix, Y::AbstractMatrix,
                         alpha::AbstractVector, alpha0, A_opt; 
                         max_epochs=1000, term_iter=20, atol=1e-5,
                         verbosity=1, print_prefix="", print_iter=100)

    A_loss = A -> gamma_normal_loss(A, S, alpha, alpha0, Y)
    reg_loss = A -> sum(transpose(A_opt.lambda) .* abs.(A))
    term_count = 0

    best_loss = A_loss(A) + reg_loss(A)
    A_best = deepcopy(A)
    v_println("Iteration 0:\t Loss=", best_loss; verbosity=verbosity,
                                                 prefix=print_prefix)

    for epoch=1:max_epochs

        A_grads = Zygote.gradient(A_loss, A)
        update!(A_opt, A, A_grads[1]) 
        new_loss = A_loss(A) + reg_loss(A)

        # Track the best we've seen thus far.
        # If we don't make any progress, then increment
        # the termination counter.
        if new_loss < best_loss
            loss_diff = best_loss - new_loss

            best_loss = new_loss
            A_best .= A
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

    A .= A_best
    v_println("Final loss: ", best_loss; verbosity=verbosity, prefix=print_prefix)

    return best_loss
end

function update_A!(reg::FeatureSetARDReg, Y::AbstractMatrix;
                   max_epochs=1000, term_iter=20, atol=1e-5,
                   verbosity=1, print_prefix="", print_iter=100)

    for (cr, A, S, opt) in zip(reg.col_ranges, reg.A, reg.S, reg.A_opts)
        Y_view = view(Y, :, cr)
        A .= 0
        update_A_inner!(A, S, Y_view, reg.alpha[cr], reg.alpha0, opt;
                        max_epochs=max_epochs, term_iter=term_iter, atol=atol,
                        verbosity=verbosity, print_prefix=print_prefix, print_iter=print_iter)

        reg.beta[:,cr] .= (reg.alpha0 - 1).*(1 .+ transpose(A)*S)
    end
end


#function set_lambda_max(reg::FeatureSetARDReg, Y::AbstractMatrix)
#    
#    A_loss = Z -> gamma_normal_loss(Z, reg.S, reg.alpha, reg.beta0, Y)
#
#    A = zero(reg.A)
#    origin_grad = Zygote.gradient(A_loss, A)[1]
#    lambda_max = maximum((abs.(origin_grad)))
#
#    return lambda_max
#end


## Score A by the fraction of its columns 
## containing a nonzero entry.
#function score_A(A; threshold=1e-3)
#    L,K = size(A)
#    #return sum(sum(A .> threshold, dims=1) .> 0)/K
#    return sum(A .> threshold)/(L*K)
#end


#function update_A!(reg::FeatureSetARDReg, Y::AbstractMatrix; 
#                   max_epochs=500, term_iter=50,
#                   target_frac=0.7,
#                   print_iter=100,
#                   bin_search_max_iter=20,
#                   bin_search_frac_atol=0.1,
#                   bin_search_lambda_atol=1e-2,
#                   verbosity=1, print_prefix="",
#                   history=nothing)
#
#    n_pref = string(print_prefix, "    ")
#
#    v_println("Updating A and λ_A..."; verbosity=verbosity, prefix=print_prefix)
#    
#    i = 1
#    function eval_func(lambda)
#        v_println("(", i,") Updating A with λ_A=", lambda; verbosity=verbosity-1, prefix=print_prefix)
#        reg.A_opt.lambda .= lambda
#        reg.A .= 0 # Ditch the warm-start -- it seems to get stuck! 
#        update_A_inner!(reg, Y; max_epochs=max_epochs,
#                                term_iter=term_iter,
#                                verbosity=verbosity-1,
#                                print_iter=print_iter, 
#                                print_prefix=n_pref)
#        score = -score_A(reg.A)
#        v_println("Fraction=", -score; verbosity=verbosity-1, prefix=print_prefix)
#        i += 1 # Some statefulness... `reg`, and the number of iterations 
#        return score 
#    end
#
#    # Search for lambda that yields the desired score.
#    # This updates `reg` along the way.
#    lambda_start = mean(reg.A_opt.lambda)
#    lambda, score = func_binary_search(lambda_start, -target_frac, eval_func; max_iter=bin_search_max_iter,
#                                                                              z_atol=bin_search_frac_atol,
#                                                                              x_atol=bin_search_lambda_atol)
#    score *= -1
#    history!(history; name="update_A", lambda_A=lambda, A_frac=score)
#
#    v_println("Finished updating A; selected λ_A=", lambda, "; A_fraction=", score; verbosity=verbosity, prefix=print_prefix)
# 
#    reg.beta =  reg.beta0 .* (1 .+ transpose(reg.A)*reg.S)
#end


 

