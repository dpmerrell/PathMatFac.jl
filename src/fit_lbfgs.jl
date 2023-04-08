# Code for initializing model *factors* (i.e., X and Y) via L-BFGS.

function full_loss(model::MF.MatFacModel, D::AbstractMatrix; capacity=10^8)
    data_loss = MF.batched_data_loss(model, D; capacity=capacity)
    #X_reg_loss = model.X_reg(model.X)
    #Y_reg_loss = model.Y_reg(model.Y)
    return data_loss 
end


function full_gradient(model::MF.MatFacModel, D::Matrix; capacity=10^8)
    
    M, N = size(D)
    nthread = Threads.nthreads()
    capacity = min(capacity, M*N)
    row_batch_size = max(div(capacity,(N*nthread)),1)
    row_batches = MF.batch_iterations(M, row_batch_size)
    n_batches = length(row_batches)

    X_g_buffer = similar(model.X)
    Y_g_buffer = [similar(model.Y) for _=1:nthread]
     
    # Accumulate likelihood gradients
    Threads.@threads :static for i=1:n_batches

        row_batch=row_batches[i]
        D_v = view(D, row_batch, :)
        X_v = view(model.X, :, row_batch)

        row_trans_view = view(model.row_transform, row_batch, 1:N)
        col_trans_view = view(model.col_transform, row_batch, 1:N)

        grads = Zygote.gradient((x,y) -> MF.likelihood(x,y,
                                                       row_trans_view,
                                                       col_trans_view,
                                                       model.noise_model,
                                                       D_v), 
                                X_v, model.Y)
        
        X_g_v = view(X_g_buffer, :, row_batch)
        X_g_v .= grads[1]
        
        th = Threads.threadid()
        Y_g_buffer[th] .= grads[2]
    end
   
    ## Include regularizer gradients 
    #X_reg_grads = Zygote.gradient(x->model.X_reg(x), model.X)
    #X_g_buffer .+= X_reg_grads[1]

    #Y_reg_grads = Zygote.gradient(y->model.Y_reg(y), model.Y)
    #Y_g_buffer[1] .+= Y_reg_grads[1]

    MF.accumulate_sum!(Y_g_buffer)

    return (X=X_g_buffer, Y=Y_g_buffer[1])
end


function full_gradient(model::MF.MatFacModel, D::CuMatrix; capacity=10^8)

    M, N = size(D_list[1])
    row_batch_size = div(capacity, N)

    X_g = similar(model.X)
    Y_g = zeros_like(model.Y)

    for row_batch in BatchIter(M, row_batch_size)
        D_v = view(D, row_batch, :)
        X_v = view(model.X, :, row_batch)
        row_trans_view = view(model.row_transform, row_batch, 1:N)
        col_trans_view = view(model.col_transform, row_batch, 1:N)
        grads = Zygote.gradient((x,y) -> MF.likelihood(x,y,
                                                       row_trans_view,
                                                       col_trans_view,
                                                       model.noise_model,
                                                       D_v), 
                                X_v, model.Y)
        X_v .= grads[1]
        Y_v .+= grads[2]
    end
   
    # Include regularizer gradients 
    X_reg_grads = Zygote.gradient(x->model.X_reg(x), model.X)
    X_g .+= X_reg_grads[1]
    Y_reg_grads = Zygote.gradient(y->model.Y_reg(y), model.Y)
    Y_g .+= Y_reg_grads[1]

    return (X=X_g, Y=Y_g)
end


# Some arithmetic operations for (X,Y) pairs
plus(p1::NamedTuple, p2::NamedTuple) = (X=p1.X .+ p2.X, Y=p1.Y .+ p2.Y)
minus(p1::NamedTuple, p2::NamedTuple) = (X=p1.X .- p2.X, Y=p1.Y .- p2.Y)
function minus!(p1::NamedTuple, p2::NamedTuple)
    p1.X .-= p2.X
    p1.Y .-= p2.Y
end

scalar_mult(p1::NamedTuple, k::Number) = (X=p1.X .* k, Y=p1.Y .* k)
scalar_mult(k::Number, p1::NamedTuple) = scalar_mult(p1, k)
function scalar_mult!(p1::NamedTuple, k::Number)
    p1.X .*= k
    p1.Y .*= k
end

inner_prod(p1::NamedTuple, p2::NamedTuple) = sum(p1.X .* p2.X) + sum(p1.Y .* p2.Y)
 
# Test whether l1 is sufficiently smaller than l0
sufficient_decrease(l0, l1, s, dd; c=1e-4) = (l1 <= l0 + c*s*dd)


function backtrack!(p::NamedTuple, model::MatFacModel, D::AbstractMatrix, l0::Number, g::NamedTuple; 
                    shrinkage=0.8, capacity=10^8, max_iter=20)

    # Store the original factors
    orig_X = deepcopy(model.X)
    orig_Y = deepcopy(model.Y)

    # Compute the directional derivative for the
    # search vector, p
    p_norm = sqrt(inner_prod(p, p))
    dd = inner_prod(p, g) / p_norm
 
    l1 = Inf
    iter = 1
    while iter <= max_iter
        model.X .= orig_X .+ p.X
        model.Y .= orig_Y .+ p.Y
        l1 = full_loss(model, D; capacity=capacity)
        # Terminate if we satisfy sufficient decrease   
        if sufficient_decrease(l0, l1, p_norm, dd)
            break
        end

        # Update the search vector and its norm
        scalar_mult!(p, shrinkage)
        p_norm *= shrinkage
        iter += 1
    end

    return l1
end


function inner_loop!(p, p_queue, y_queue, rho_queue)

    m = length(p_queue)
    alphas = zeros(m)
    for k=1:m
        alphas[k] = rho_queue[k]*inner_prod(p_queue[k], p)
        minus!(p, scalar_mult(y_queue[k], alphas[k]))
    end
    # *Multiply by a scaled identity*
    if length(p_queue) > 0
        scalar_mult!(p, inner_prod(p_queue[1], y_queue[1])/inner_prod(y_queue[1],y_queue[1]))
    end
    for k=m:-1:1
        beta = rho_queue[k]*inner_prod(y_queue[k],p)
        minus!(p, scalar_mult(p_queue[k], beta - alphas[k])) 
    end
end


function fit_lbfgs!(model::MatFacModel, D::AbstractMatrix; capacity=10^8,
                                                           m=10,
                                                           max_iter=1000,
                                                           rel_tol=1e-10,
                                                           abs_tol=1e-6,
                                                           backtrack_shrinkage=0.8,
                                                           print_prefix="",
                                                           print_iter=1,
                                                           verbosity=1
                                                           )
    # A queue of m previous gradient differences
    y_queue = []
    
    # A queue of m previous update vectors
    p_queue = []

    # A queue of m previous inner products 1/(y*p)
    rho_queue = []

    old_grad = nothing
    p = nothing
    iter = 0
    cur_loss = full_loss(model, D; capacity=capacity)
    for i=1:max_iter

        if i % print_iter == 0
            v_println("(", i, ") L-BFGS; Loss=",cur_loss; prefix=print_prefix, verbosity=verbosity)
        end 
        cur_grad = full_gradient(model, D; capacity=capacity)

        rho = 1
        if old_grad != nothing
            if length(y_queue) >= m
                pop!(p_queue)
                pop!(y_queue)
                pop!(rho_queue)
            end
            pushfirst!(p_queue, p)
            pushfirst!(y_queue, minus(cur_grad, old_grad))
            rho = 1/inner_prod(y_queue[1], p_queue[1])
            pushfirst!(rho_queue, rho)
        end

        p = deepcopy(cur_grad)
        scalar_mult!(p, -1)

        if rho > 0
            inner_loop!(p, p_queue, y_queue, rho_queue)
        else
            p_queue = []
            y_queue = []
            rho_queue = []
            old_grad = nothing
        end

        new_loss = backtrack!(p, model, D, cur_loss, cur_grad; 
                              shrinkage=backtrack_shrinkage,
                              capacity=capacity)

        loss_diff = cur_loss - new_loss
        if abs(loss_diff) < abs_tol
            v_println("Loss decrease < ", abs_tol; prefix=print_prefix, verbosity=verbosity)
            break
        elseif abs(loss_diff/new_loss) < rel_tol
            v_println("Relative loss decrease < ", rel_tol; prefix=print_prefix, verbosity=verbosity)
            break
        end

        cur_loss = new_loss
        old_grad = cur_grad
        iter += 1
    end 

end



