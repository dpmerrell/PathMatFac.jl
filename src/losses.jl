

abstract type Loss end

###############################
# Quadratic loss (Normal data)
###############################
struct QuadLoss <: Loss 
    scale::Number
end

function evaluate(ql::QuadLoss, x, y, a)
    return scale * 0.5 * (dot(x,y) - a)^2
end

function grad_x(ql::QuadLoss, x, y, a)
    return scale * (dot(x,y) - a) .* y
end

function grad_y(ql::QuadLoss, x, y, a)
    return scale * (dot(x,y) - a) .* x
end


###############################
# Logistic loss (binary data)
###############################
struct LogisticLoss <: Loss 
    scale::Number
end

function evaluate(ll::LogisticLoss, x, y, a)
    z = dot(x,y)
    return scale * ( log(1.0 + exp(-z)) + (1.0-a)*z )
end

function grad_x(ll::LogisticLoss, x, y, a)
    z = dot(x,y)
    return scale * ( 1.0/(1.0 + exp(z)) + (1.0-a) ) .* y 
end

function grad_y(ll::LogisticLoss, x, y, a)
    z = dot(x,y)
    return scale * ( 1.0/(1.0 + exp(z)) + (1.0-a) ) .* x 
end


###############################
# Poisson loss (count data)
###############################
struct PoissonLoss <: Loss 
    scale::Number
end

function evaluate(pl::PoissonLoss, x, y, a)
    z = dot(x,y)
    return scale * ( a*z - exp(z) )
end

function grad_x(pl::PoissonLoss, x, y, a)
    return scale * ( a - exp(dot(x,y)) ) .* y
end

function grad_y(pl::PoissonLoss, x, y, a)
    return scale * ( a - exp(dot(x,y)) ) .* x
end

 
