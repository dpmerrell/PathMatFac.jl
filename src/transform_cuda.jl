
import ScikitLearnBase: transform

export transform


function transform(model::MatFacModel, A::AbstractMatrix;
                   inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
                   max_iter::Integer=100, 
                   lr::Real=0.001, momentum::Real=0.5,
                   abs_tol::Real=1e-3, rel_tol::Real=1e-7,
                   loss_iter::Integer=10, 
                   X_new::Union{Nothing,AbstractMatrix}=nothing,
                   new_inst_reg_mats::Union{Nothing,AbstractVector}=nothing,
                   K_opt_X::Union{Nothing,Integer}=nothing) 

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    K = size(model.X, 1)
    M_old = size(model.X, 2)
    M_new = size(A,1)
    M_comb = M_old + M_new
    N = size(A,2)

    if X_new == nothing
        X_new = 0.001*randn(K, M_new)
    end
    X_comb = hcat(model.X, X_new) 
   
    println("X_COMB: ", size(X_comb))

    @assert size(X_comb,1) == size(model.Y,1)
    @assert size(X_new, 2) == size(A,1)

    # Distinguish between (1) K, the hidden dimension;
    #                     (2) the subsets 1:K_opt_X, 1:K_opt_Y of 1:K that is optimized;
    #                     (3) the subset of optimized dims that are regularized.
    # The regularized dim is determined by the lists of matrices
    # provided during MatFacModel construction.
    # K_opt_X, K_opt_Y have default value K.
    if K_opt_X == nothing
        K_opt_X = K
    end
    @assert K_opt_X <= K

    lr = Float32(lr)

    # Move data and model to the GPU.
    # Float16 is sufficiently precise for the data.
    X_comb_d = CuArray{Float32}(X_comb)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float16}(A)
    XY_d = CUDA.zeros(Float16, M_new, N)

    println("X_COMB_D: ", size(X_comb_d))
    println("Y_D: ", size(Y_d))
    println("A_D: ", size(A_d))
    println("XY_D: ", size(XY_d))

    # Some frequently-used views of the X and Y arrays
    X_new_view = view(X_comb_d, :, (M_old+1):(M_old+M_new))
    X_opt_view = view(X_comb_d, 1:K_opt_X, (M_old+1):(M_old+M_new))
    Y_grad_view = view(Y_d, 1:K_opt_X, :)

    println("X_NEW_VIEW: ", size(X_new_view))
    println("X_OPT_VIEW: ", size(X_opt_view))
    println("Y_GRAD_VIEW: ", size(Y_grad_view))

    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float16(0.5))
    # Convert NaNs in the data --> something benign so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float16(0.5) : Float16(x)
    map!(mask_func, A_d, A_d)

    # Scaling factors for the columns
    scales = [loss.scale for loss in model.losses]
    feature_scales = CuArray{Float32}(transpose(scales))
    
    # Bookkeeping for the loss functions
    ql_idx = CuVector{Int64}(findall(typeof.(model.losses) .== QuadLoss))
    ll_idx = CuVector{Int64}(findall(typeof.(model.losses) .== LogisticLoss))
    pl_idx = CuVector{Int64}(findall(typeof.(model.losses) .== PoissonLoss))

    XY_ql_view = view(XY_d, :, ql_idx)
    XY_ll_view = view(XY_d, :, ll_idx)
    XY_pl_view = view(XY_d, :, pl_idx)
    A_ql_view = view(A_d, :, ql_idx)
    A_ll_view = view(A_d, :, ll_idx)
    A_pl_view = view(A_d, :, pl_idx)

    # If regularizer matrices are not provided,
    # then let them be the identity
    if new_inst_reg_mats == nothing
        new_inst_reg_mats = [sparse(I, M_comb, M_comb) for i=1:K_opt_X]

    # Otherwise, make sure they're the right size
    # for the combined (old, new) X matrix.
    else
        @assert typeof(new_inst_reg_mats) <: AbstractVector
        for mat in new_inst_reg_mats
            @assert typeof(mat) <: AbstractMatrix
            @assert size(mat) == (M_comb, M_comb)
        end
    end
    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in new_inst_reg_mats]

    # Arrays for holding gradients and velocities
    grad_X_temp = CUDA.zeros(Float32, (K_opt_X, M_comb))
    grad_X = CUDA.zeros(Float32, (K_opt_X, M_new))
    vel_X = CUDA.zeros(Float32, (K_opt_X, M_new))

    println("GRAD_X_TEMP: ", size(grad_X_temp))
    println("GRAD_X: ", size(grad_X))
    println("VEL_X: ", size(vel_X))

    while iter < max_iter 

        ############################
        # Update X 

        # take momentum "half-step"
        X_opt_view .+= (momentum.*vel_X)

        # compute gradient at half-step
        XY_d .= transpose(X_new_view)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask 
        compute_quadloss_delta!(XY_ql_view, A_ql_view)
        compute_logloss_delta!(XY_ll_view, A_ll_view)
        compute_poissonloss_delta!(XY_pl_view, A_pl_view)

        XY_d .*= feature_scales
        grad_X .= (Y_grad_view * transpose(XY_d)) ./ N

        grad_X_temp .= 0.0
        add_reg_grad!(grad_X_temp, X_comb_d, inst_reg_mats_d) 
        grad_X .+= grad_X_temp[:,(M_old+1):(M_old+M_new)]

        # scale gradient by learning rate
        grad_X .*= lr

        # update velocity
        vel_X .*= momentum
        vel_X .-= grad_X

        # complete X update
        X_opt_view .-= grad_X

        iter += 1
        print_str = "Iteration: $iter"

        ############################
        # Every so often, compute the loss
        if (iter % loss_iter == 0)
            XY_d .= transpose(X_new_view)*Y_d
            XY_d .*= obs_mask
            
            new_loss = compute_quadloss!(XY_ql_view, A_ql_view)
            new_loss += compute_logloss!(XY_ll_view, A_ll_view)
            new_loss += compute_poissonloss!(XY_pl_view, A_pl_view)
            new_loss += compute_reg_loss(X_comb_d, inst_reg_mats_d)
            print_str = string(print_str, "\tLoss: $new_loss")
            println(print_str)

            if abs(new_loss - cur_loss) < abs_tol
                println(string("Absolute change <", abs_tol, ". Terminating."))
                break
            end
            if abs((new_loss - cur_loss)/cur_loss) < rel_tol
                println(string("Relative change <", rel_tol, ". Terminating."))
                break
            end
            cur_loss = new_loss
        end

    end # while

    # Move transformed X back to CPU
    X_new = Array{Float32}(X_new_view)

    return X_new
end


