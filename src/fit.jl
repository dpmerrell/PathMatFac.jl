

export fit!, fit_cuda!


function fit!(model::MatFacModel, A::AbstractMatrix;
              inst_reg_weight=1.0, feat_reg_weight=1.0,
              max_iter::Int=100, lr::Float64=0.001, 
              abs_tol::Float64=1e-6, rel_tol::Float64=1e-5)

    # Setup
    loss = Inf
    iter = 0

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    # graph coloring
    x_union_graph = graph_union(model.instance_reg_mats)
    x_colors = compute_coloring(x_union_graph) 

    y_union_graph = graph_union(model.feature_reg_mats)
    y_colors = compute_coloring(y_union_graph)


    while iter < max_iter # and abs(loss - new_loss) > abs_tol and abs(loss - new_loss)/loss > rel_tol

        # Update X columns
        # Loop over rows of data
        for x_color in x_colors 
            Threads.@threads for i in x_color

                # Accumulate loss gradients
                g = zeros(K)
                #X_view = view(model.X,:,i)
                xtY = BLAS.gemv('T', model.Y, model.X[:,i])
                for j=1:N
                    if !isnan(A[i,j])
                        accum_grad_x!(g, model.losses[j], model.Y[:,j], xtY[j], A[i,j])
                    end 
                end
                # Add regularizer gradient
                g += inst_reg_weight .* reg_grad(model.X, i, model.instance_reg_mats)

                # Update with gradient

                model.X[:,i] -= lr.*g
            end # i loop
        end # x_color loop

        # Update Y columns 
        # Loop over columns of data
        for y_color in y_colors
            Threads.@threads for j in y_color
                loss_fn = model.losses[j]
                # Accumulate loss gradients
                g = zeros(K)
                #Y_view = view(model.Y,:,j)
                Xty = BLAS.gemv('T', model.X, model.Y[:,j])
                for i=1:M
                    if !isnan(A[i,j])
                        accum_grad_y!(g, loss_fn, model.X[:,i], Xty[i], A[i,j])
                    end
                end
                # Add regularizer gradient
                g += feat_reg_weight .* reg_grad(model.Y, j, model.feature_reg_mats)

                # Update with gradient
                model.Y[:,j] -= lr.*g

            end # j loop
        end # y_color loop

        iter += 1
        println(iter)

    end # while

end


function fit_cuda!(model::MatFacModel, A::AbstractMatrix;
                   inst_reg_weight::Real=1.0, feat_reg_weight::Real=1.0,
                   max_iter::Integer=100, lr::Real=0.001, 
                   abs_tol::Real=1e-6, rel_tol::Real=1e-5,
                   loss_iter::Integer=10, 
                   K_opt_X::Union{Nothing,Integer}=nothing, 
                   K_opt_Y::Union{Nothing,Integer}=nothing)

    # Setup
    iter = 0

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
    print("X OPT DIM: ", K_opt_X)
    print("Y OPT DIM: ", K_opt_Y)
    @assert K_opt_X <= K
    @assert K_opt_Y <= K

    lr = Float32(lr)

    # Move data and model to the GPU.
    # Float16 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float16}(A)
    XY_d = CUDA.zeros(Float16, M,N)
   
    # Some bookkeeping for missing values.
    obs_mask = (!isnan).(A_d)
    missing_mask = (isnan.(A_d) .* Float16(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float16(0.5) : Float16(x)
    map!(mask_func, A_d, A_d)

    # Scaling factors for the columns
    scales = [loss.scale for loss in model.losses]
    feature_scales = CuArray{Float32}(transpose(scales))
    
    # Bookkeeping for the loss functions
    ql_idx = CuVector{Int64}(findall(typeof.(model.losses) .== QuadLoss))
    ll_idx = CuVector{Int64}(findall(typeof.(model.losses) .== LogisticLoss))
    pl_idx = CuVector{Int64}(findall(typeof.(model.losses) .== PoissonLoss))

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients
    grad_X = CUDA.zeros(Float32, (K_opt_X, size(X_d,2)))
    grad_Y = CUDA.zeros(Float32, (K_opt_Y, size(Y_d,2)))

    while iter < max_iter 

        ############################
        # Update X 
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask 
        CUDA.@sync compute_loss_delta!(XY_d, A_d, ql_idx, ll_idx, pl_idx)
        println("LOSS DELTAS:")
        println("\tfrac nonzero: ", sum(XY_d .!= 0.0)/M/N)
        println("\taverage abs: ", sum(abs.(XY_d./N))/M)
        readline()
        XY_d .*= feature_scales
        println("FEATURE-SCALED LOSS DELTAS:")
        println("\tfrac nonzero: ", sum(XY_d .!= 0.0)/M/N)
        println("\taverage abs: ", sum(abs.(XY_d./N))/M)
        readline()
        grad_X .= (Y_d[1:K_opt_X,:] * transpose(XY_d)) ./ N
        println("GRADIENT_X:")
        println("\tfrac nonzero: ", sum(grad_X .!= 0.0)/M/K)
        println("\taverage abs: ", sum(abs.(XY_d./K))/M)
        readline()


        CUDA.@sync add_reg_grad!(grad_X, X_d, inst_reg_mats_d) 
        grad_X .*= lr
        X_d[1:K_opt_X,:] .-= grad_X

        ############################
        # Update Y 
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        CUDA.@sync compute_loss_delta!(XY_d, A_d, ql_idx, ll_idx, pl_idx)
        XY_d .*= feature_scales
        grad_Y .= (X_d[1:K_opt_Y,:] * XY_d) ./ M

        CUDA.@sync add_reg_grad!(grad_Y, Y_d, feat_reg_mats_d)
        grad_Y .*= lr
        Y_d[1:K_opt_Y,:] .-= grad_Y

        iter += 1
        print_str = "Iteration: $iter"

        ############################
        # Every so often, compute the loss
        if (iter % loss_iter == 0)
            XY_d .= transpose(X_d)*Y_d
            XY_d .*= obs_mask
            
            loss = compute_loss!(XY_d, A_d, ql_idx, ll_idx, pl_idx)
            loss += compute_reg_loss(X_d, inst_reg_mats_d)
            loss += compute_reg_loss(Y_d, feat_reg_mats_d)
            print_str = string(print_str, "\tLoss: $loss")
        end
        println(print_str)

    end # while

    # Move model X and Y back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)
   
    return nothing
end


