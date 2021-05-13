

import ScikitLearnBase: fit!

export fit!, fit_line_search!


function fit!(model::MatFacModel, A::AbstractMatrix;
              inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
              max_iter::Integer=100, 
              lr::Real=0.001, momentum::Real=0.5,
              abs_tol::Real=1e-3, rel_tol::Real=1e-7,
              loss_iter::Integer=10, 
              K_opt_X::Union{Nothing,Integer}=nothing, 
              K_opt_Y::Union{Nothing,Integer}=nothing)

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    @assert size(model.X,1) == size(model.Y,1)

    # Distinguish between (1) K, the hidden dimension;
    #                     (2) the subsets 1:K_opt_X, 1:K_opt_Y of 1:K that is optimized;
    #                     (3) the subset of optimized dims that are regularized.
    # The regularized dim is determined by the lists of matrices
    # provided during MatFacModel construction.
    # K_opt_X, K_opt_Y have default value K.
    if K_opt_X == nothing
        K_opt_X = K
    end
    if K_opt_Y == nothing
        K_opt_Y = K
    end
    @assert K_opt_X <= K
    @assert K_opt_Y <= K

    lr = Float32(lr)

    # Move data and model to the GPU.
    # Float32 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float32}(A)
    XY_d = CUDA.zeros(Float32, M,N)
 
    # Some frequently-used views of the X and Y arrays
    X_d_opt_view = view(X_d, 1:K_opt_X, :)
    X_d_grady_view = view(X_d, 1:K_opt_Y, :)

    Y_d_opt_view = view(Y_d, 1:K_opt_Y, :)
    Y_d_gradx_view = view(Y_d, 1:K_opt_X, :)

    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float32(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float32(0.5) : Float32(x)
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

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients and velocities
    grad_X = CUDA.zeros(Float32, (K_opt_X, size(X_d,2)))
    vel_X = CUDA.zeros(Float32, (K_opt_X, size(X_d,2)))

    grad_Y = CUDA.zeros(Float32, (K_opt_Y, size(Y_d,2)))
    vel_Y = CUDA.zeros(Float32, (K_opt_Y, size(Y_d,2)))

    while iter < max_iter 

        ############################
        # Update X 

        # take momentum "half-step"
        X_d_opt_view .+= (momentum.*vel_X)

        # compute gradient at half-step
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        compute_grad_X!(grad_X, X_d, Y_d_gradx_view, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        inst_reg_mats_d)
      
        # scale gradient by learning rate
        grad_X .*= lr

        # update velocity
        vel_X .*= momentum
        vel_X .-= grad_X

        # complete X update
        X_d_opt_view .-= grad_X


        ############################
        # Update Y

        # take momentum "half-step"
        Y_d_opt_view .+= (momentum.*vel_Y)

        # compute gradient at half-step
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask

        compute_grad_Y!(grad_Y, X_d_grady_view, Y_d, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        feat_reg_mats_d)


        # scale gradient by learning rate
        grad_Y .*= lr

        # update velocity
        vel_Y .*= momentum
        vel_Y .-= grad_Y

        # complete Y update 
        Y_d_opt_view .-= grad_Y

        iter += 1
        print_str = "Iteration: $iter"

        ############################
        # Every so often, compute the loss
        if (iter % loss_iter == 0)
            XY_d .= transpose(X_d)*Y_d
            XY_d .*= obs_mask
           
            new_loss = compute_loss!(X_d, Y_d, XY_ql_view, XY_ll_view, XY_pl_view,
                                               A_ql_view, A_ll_view, A_pl_view,
                                               inst_reg_mats_d, feat_reg_mats_d)
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

    # Move model X and Y back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
   
    return model 
end


function fit_line_search!(model::MatFacModel, A::AbstractMatrix;
              inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
              max_iter::Integer=100,
              alpha=1.0, c1=1e-5, c2=0.9, grow=1.5, shrink=0.5, 
              line_search_max_iter=10,
              abs_tol::Real=1e-3, rel_tol::Real=1e-7,
              K_opt_X::Union{Nothing,Integer}=nothing, 
              K_opt_Y::Union{Nothing,Integer}=nothing)

    # Setup
    iter = 0
    cur_loss = Inf
    new_loss = Inf

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    @assert size(model.X,1) == size(model.Y,1)

    # Distinguish between (1) K, the hidden dimension;
    #                     (2) the subsets 1:K_opt_X, 1:K_opt_Y of 1:K that is optimized;
    #                     (3) the subset of optimized dims that are regularized.
    # The regularized dim is determined by the lists of matrices
    # provided during MatFacModel construction.
    # K_opt_X, K_opt_Y have default value K.
    if K_opt_X == nothing
        K_opt_X = K
    end
    if K_opt_Y == nothing
        K_opt_Y = K
    end
    @assert K_opt_X <= K
    @assert K_opt_Y <= K

    # Move data and model to the GPU.
    # Float32 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float32}(A)
    XY_d = CUDA.zeros(Float32, M,N)
 
    # Some frequently-used views of the X and Y arrays
    X_d_opt_view = view(X_d, 1:K_opt_X, :)
    X_d_grady_view = view(X_d, 1:K_opt_Y, :)

    Y_d_opt_view = view(Y_d, 1:K_opt_Y, :)
    Y_d_gradx_view = view(Y_d, 1:K_opt_X, :)

    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float32(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float32(0.5) : Float32(x)
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

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients
    grad_X = CUDA.zeros(Float32, (K_opt_X, size(X_d,2)))
    grad_Y = CUDA.zeros(Float32, (K_opt_Y, size(Y_d,2)))

    # These functions will be accepted as arguments
    # to the line search procedure
    function grad_fn_x(X_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        compute_grad_X!(grad_X, X_d, Y_d_gradx_view, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        inst_reg_mats_d)
        return grad_X
    end

    function grad_fn_y(Y_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        compute_grad_Y!(grad_Y, X_d_grady_view, Y_d, XY_d, 
                        XY_ql_view, XY_ll_view, XY_pl_view,
                        A_ql_view, A_ll_view, A_pl_view,
                        feature_scales,
                        feat_reg_mats_d)
        return grad_Y
    end

    function loss_fn_x(X_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        return compute_loss!(X_d, Y_d, XY_ql_view, XY_ll_view, XY_pl_view,
                             A_ql_view, A_ll_view, A_pl_view,
                             inst_reg_mats_d, feat_reg_mats_d)
    end

    function loss_fn_y(Y_)
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        return compute_loss!(X_d, Y_d, XY_ql_view, XY_ll_view, XY_pl_view,
                             A_ql_view, A_ll_view, A_pl_view,
                             inst_reg_mats_d, feat_reg_mats_d)
    end

    alpha_x = alpha
    alpha_y = alpha

    while iter < max_iter 

        ############################
        # Update X 
        alpha_x, _ = grad_descent_line_search!(X_d_opt_view, grad_X, 
                                               loss_fn_x, grad_fn_x;
                                               alpha=alpha_x, 
                                               c1=c1, c2=c2, 
                                               grow=grow, shrink=shrink, 
                                               max_iter=line_search_max_iter)


        ############################
        # Update Y
        alpha_y, new_loss = grad_descent_line_search!(Y_d_opt_view, grad_Y, 
                                                      loss_fn_y, grad_fn_y;
                                                      alpha=alpha_y, 
                                                      c1=c1, c2=c2, 
                                                      grow=grow, shrink=shrink, 
                                                      max_iter=line_search_max_iter)

        ############################
        # Print some information
        iter += 1
        print_str = "Iteration: $iter"

        print_str = string(print_str, "\tLoss: $new_loss")
        println(print_str)

        ############################
        # Check termination conditions
        if abs(new_loss - cur_loss) < abs_tol
            println(string("Absolute change <", abs_tol, ". Terminating."))
            break
        end
        if abs((new_loss - cur_loss)/cur_loss) < rel_tol
            println(string("Relative change <", rel_tol, ". Terminating."))
            break
        end
        cur_loss = new_loss


    end # while

    # Move model X and Y back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
   
    return model 
end



