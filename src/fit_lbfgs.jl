# Code for fitting model *factors* (i.e., X and Y) via L-BFGS.

function full_loss(model::MF.MatFacModel, D::AbstractMatrix; capacity=10^8)
    data_loss = MF.batched_data_loss(model, D; capacity=capacity)
    X_reg_loss = model.X_reg(model.X)
    Y_reg_loss = model.Y_reg(model.Y)
end


function full_gradient(model::MF.MatFacModel, D::Matrix)

end


function full_loss(model::MF.MatFacModel, D::CuMatrix)
    
    data_loss = MF.batched_data_loss(model, D; capacity=capacity)

end

function full_gradient(model::MF.MatFacModel, D::CuMatrix)

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
sufficient_decrease(l0, l1, s, dd; c=1e-4) = (l1 <= l0 + c*s*d)


function backtrack!(p::NamedTuple, model::MatFacModel, D::AbstractMatrix, l0::Number, g::NamedTuple; shrinkage=0.8)

    # Store the original factors
    orig_X = model.X
    orig_Y = model.Y

    # Compute the directional derivative for the
    # search vector, p
    p_norm = sqrt(inner_prod(p, p))
    dd = inner_prod(p, g) / p_norm
   
    l1 = Inf
 
    while true
        model.X = orig_X .+ p.X
        model.Y = orig_Y .+ p.Y
        l1 = full_loss(model, D)

        # Terminate if we satisfy sufficient decrease   
        if sufficient_decrease(l0, l1, p_norm, dd)
            break
        end

        # Update the search vector and its norm
        scalar_mult!(p, shrinkage)
        p_norm *= shrinkage
    end

    return l1
end


function inner_loop!(p, p_queue, y_queue, rho_queue)

    m = length(p_queue)
    alphas = zeros(m)
    for k=1:m
        alphas[k] = rho_queue[k]*inner_prod(p_queue[k], p)
        minus!(p, scalar_mult(y_queue, alphas[k]))
    end
    # *Multiply by a scaled identity*
    scalar_mult!(p, inner_prod(p_queue[1], y_queue[1])/inner_prod(y_queue[1],y_queue[1]))
    for k=m:-1:1
        beta = rho_queue[k]*inner_prod(y_queue[k],p)
        minus!(p, scalar_mult(p_queue[k], beta - alphas[k])) 
    end
end


function fit_lbfgs!(model::MatFacModel, D::AbstractMatrix; capacity=10^8,
                                                           m=10,
                                                           max_iter=1000,
                                                           rel_tol=1e-9,
                                                           abs_tol=1e-5,
                                                           backtrack_shrinkage=0.8
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
    for i=1:max_iter

        v_println("(", i, ") L-BFGS")      
        cur_grad = full_gradient(model, D)

        if i > 1
            if length(y_queue) > m
                pop!(p_queue)
                pop!(y_queue)
                pop!(rho_queue)
            end
            pushfirst!(p_queue, p)
            pushfirst!(y_queue, minus(cur_grad, old_grad))
            pushfirst!(rho_queue, 1/inner_product(y_queue[1], p_queue[1]))
        end

        p = deepcopy(cur_grad)
        scalar_mult!(p, -1)
        inner_loop!(p, p_queue, y_queue, rho_queue)

        new_loss = backtrack!(p, model, D, cur_loss, cur_grad; 
                              shrinkage=backtrack_shrinkage)

        loss_diff = cur_loss - new_loss
        if loss_diff < abs_tol
            v_println("Loss decrease < ", abs_tol)
            break
        elseif loss_diff/new_loss < rel_tol
            v_println("Relative loss decrease < ", rel_tol)
            break
        end

        old_grad = cur_grad
        iter += 1
    end 

end



