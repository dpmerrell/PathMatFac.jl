

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
    return sum(1.0.*XY.*A - exp.(1.0.*XY))
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
function compute_loss!(X, Y, XY_ql, XY_ll, XY_pl, 
                             A_ql, A_ll, A_pl, 
                             X_reg_mats, Y_reg_mats)
    loss = compute_quadloss!(XY_ql, A_ql)
    loss += compute_logloss!(XY_ll, A_ll)
    loss += compute_poissonloss!(XY_pl, A_pl)
    loss += compute_reg_loss(X, X_reg_mats)
    loss += compute_reg_loss(Y, Y_reg_mats)
    return loss
end


function compute_grad_delta!(XY_ql, XY_ll, XY_pl, 
                             A_ql, A_ll, A_pl)
    compute_quadloss_delta!(XY_ql, A_ql)
    compute_logloss_delta!(XY_ll, A_ll)
    compute_poissonloss_delta!(XY_pl, A_pl)
end


function compute_grad_X!(grad_X, X, Y, XY, 
                         XY_ql, XY_ll, XY_pl, 
                         A_ql, A_ll, A_pl, 
                         feature_scales, inst_reg_mats)

    compute_grad_delta!(XY_ql, XY_ll, XY_pl, 
                        A_ql, A_ll, A_pl)

    XY .*= feature_scales
    N = size(XY,2)
    grad_X .= (Y * transpose(XY)) ./ N

    add_reg_grad!(grad_X, X, inst_reg_mats) 
end


function compute_grad_Y!(grad_Y, X, Y, XY, 
                         XY_ql, XY_ll, XY_pl, 
                         A_ql, A_ll, A_pl, 
                         feature_scales, feat_reg_mats)

    compute_grad_delta!(XY_ql, XY_ll, XY_pl, 
                        A_ql, A_ll, A_pl)

    XY .*= feature_scales
    M = size(XY,1)
    grad_Y .= X * XY ./ M

    add_reg_grad!(grad_Y, Y, feat_reg_mats) 
end


