

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


using CUDA
using CUDA.CUSPARSE

function fit_cuda!(model::MatFacModel, A::AbstractMatrix;
                   inst_reg_weight=1.0, feat_reg_weight=1.0,
                   max_iter::Int=100, lr::Float64=0.001, 
                   abs_tol::Float64=1e-6, rel_tol::Float64=1e-5,
                   loss_iter::Int64=10)


    # Setup
    loss = Inf
    iter = 0

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    # Move data and model to the GPU.
    # Float16 is sufficiently precise.
    X_d = CuArray{Float32}(model.X)
    Y_d = CuArray{Float32}(model.Y)
    A_d = CuArray{Float16}(A)
    XY_d = CUDA.zeros(Float16, M,N)
   
    # Some bookkeeping for missing values.
    obs_mask = map(!isnan, A_d)
    missing_mask = (map(isnan, A_d) .* Float16(0.5))
    # Convert NaNs in the data --> zeros so CuArray arithmetic works
    mask_func(x) = isnan(x) ? Float16(0.5) : Float16(x)
    map!(mask_func, A_d, A_d)

    # TODO: bookkeeping for the loss functions
    #       * indices for each loss function type
    #       * feature_scales for each column
    feature_scales = CUDA.ones(Float16, (1, size(Y_d,2)))

    # Convert regularizers to CuSparse matrices
    inst_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* inst_reg_weight) for mat in model.instance_reg_mats]
    feat_reg_mats_d = [CuSparseMatrixCSC{Float32}(mat .* feat_reg_weight) for mat in model.feature_reg_mats]

    # Arrays for holding gradients
    grad_X = CUDA.zeros(Float32, size(X_d))
    grad_Y = CUDA.zeros(Float32, size(Y_d))

    while iter < max_iter 

        ############################
        # Update X 
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask 
        # TODO: loop over the different loss function types
        compute_delta_logloss!(XY_d, A_d)
        XY_d .*= feature_scales
        grad_X .= (Y_d * transpose(XY_d))

        add_reg_grad!(grad_X, X_d, inst_reg_mats_d) 
        grad_X .*= lr
        X_d .-= grad_X

        ############################
        # Update Y 
        XY_d .= transpose(X_d)*Y_d
        XY_d .*= obs_mask
        XY_d .+= missing_mask
        # TODO: loop over the different loss function types
        compute_delta_logloss!(XY_d, A_d) 
        XY_d .*= feature_scales
        grad_Y .= (X_d * XY_d)

        add_reg_grad!(grad_Y, Y_d, feat_reg_mats_d)
        grad_Y .*= lr
        Y_d .-= grad_Y

        iter += 1
        print_str = "Iteration: $iter"

        # Every so often, compute the loss
        if iter % loss_iter == 0
            XY_d .= transpose(X_d)*Y_d
            XY_d .*= obs_mask
            
            # TODO: loop over loss function types
            loss = compute_logloss(XY_d, A_d)
            loss += compute_reg_loss(X_d, inst_reg_mats_d)
            loss += compute_reg_loss(Y_d, feat_reg_mats_d)
            print_str = string(print_str, "\tLoss: $loss")
        end
        println(print_str)

    end # while

    # Move model X and Y back to CPU
    model.X = Array{Float32}(X_d)
    model.Y = Array{Float32}(Y_d)

end


