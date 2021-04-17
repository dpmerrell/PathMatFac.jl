

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

 
