

export Loss, QuadLoss, LogisticLoss, PoissonLoss

abstract type Loss end

###############################
# Quadratic loss (Normal data)
###############################
struct QuadLoss <: Loss 
    scale::Number
end

function evaluate(ql::QuadLoss, x, y, a)
    return ql.scale * 0.5 * (dot(x,y) - a)^2
end

function grad_x(ql::QuadLoss, x, y, a)
    return ql.scale * (dot(x,y) - a) .* y
end

function grad_y(ql::QuadLoss, x, y, a)
    return ql.scale * (dot(x,y) - a) .* x
end

function compute_quadloss!(XY, A)
    return 0.5*sum( (XY - A).^2 )
end

function compute_quadloss_delta!(XY, A)
    XY .-= A
    return nothing
end

###############################
# Logistic loss (binary data)
###############################
struct LogisticLoss <: Loss 
    scale::Number
end

function evaluate(ll::LogisticLoss, x, y, a)
    z = dot(x,y)
    return ll.scale * ( log(1.0 + exp(-z)) + (1.0-a)*z )
end

function accum_grad_x!(g, ll::LogisticLoss, y, xy, a)
    BLAS.axpy!(ll.scale * ( (1.0-a) - 1.0/(1.0 + exp(xy)) ), y, g)
    return
end

function accum_grad_y!(g, ll::LogisticLoss, x, xy, a)
    BLAS.axpy!(ll.scale * ( (1.0-a) - 1.0/(1.0 + exp(xy)) ), x, g)
    return 
end

function compute_logloss!(XY, A)
    XY .= exp.(-XY)
    XY .+= 1.0
    XY .= 1.0 ./XY
    return -sum( A .* log.(1e-5 .+ XY) + (1.0 .- A).*log.(1e-5 + 1.0 .- XY) )
end


function compute_logloss_delta!(XY, A)
    XY .= exp.(-XY)
    XY .+= 1.0
    XY .= 1.0 ./ XY
    XY .-= A
    return nothing 
end

###############################
# Poisson loss (count data)
###############################
struct PoissonLoss <: Loss 
    scale::Number
end

function evaluate(pl::PoissonLoss, x, y, a)
    z = dot(x,y)
    return pl.scale * ( a*z - exp(z) )
end

function grad_x(pl::PoissonLoss, x, y, a)
    return pl.scale * ( a - exp(dot(x,y)) ) .* y
end

function grad_y(pl::PoissonLoss, x, y, a)
    return pl.scale * ( a - exp(dot(x,y)) ) .* x
end

function compute_poissonloss!(XY, A)
    return sum(XY.*A - exp.(XY))
end

function compute_poissonloss_delta!(XY, A)
    XY .= exp.(XY)
    XY .*= -1.0
    XY .+= A
    return nothing
end


##################################
# Other functions
##################################
function compute_loss!(X, Y, XY_ql, XY_ll, XY_pl, A, X_reg_mats, Y_reg_mats)
    loss = compute_quadloss!(XY_ql, A)
    loss += compute_logloss!(XY_ll, A)
    loss += compute_poissonloss!(XY_pl, A)
    loss += compute_reg_loss(X, X_reg_mats)
    loss += compute_reg_loss(Y, Y_reg_mats)
    return loss
end


function compute_grad_delta!(X, Y, XY, XY_ql, XY_ll, XY_pl, A)
    compute_quadloss_delta!(XY_ql_view, A_ql_view)
    compute_logloss_delta!(XY_ll_view, A_ll_view)
    compute_poissonloss_delta!(XY_pl_view, A_pl_view)
    
end


function compute_grad_X(X, Y, XY, XY_ql, XY_ll, XY_pl, A, feature_scales)

    compute_grad_delta!(X,Y, XY, XY_ql, XY_ll, XY_pl, A)

    XY_d .*= feature_scales
    grad_X .= (Y_d_grad_view * transpose(XY_d)) ./ N

    add_reg_grad!(grad_X, X_d, inst_reg_mats_d) 
end


