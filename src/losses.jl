

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
precompile(compute_quadloss!, (CuArray{Float16,2},
                               CuArray{Float16,2}))

function compute_quadloss_delta!(XY, A)
    XY .-= A
    return nothing
end
precompile(compute_quadloss_delta!, (CuArray{Float16,2},
                                     CuArray{Float16,2}))

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
#precompile(compute_logloss!, (CuArray{Float16,2},
#                              CuArray{Float16,2}))


function compute_logloss_delta!(XY, A)
    XY .= exp.(-XY)
    XY .+= 1.0
    XY .= 1.0 ./ XY
    XY .-= A
    println("FINISHED COMPUTING LOGLOSS DELTA:")
    println("\tfrac nonzero: ", sum(XY .!= 0.0)/M/N)
    println("\taverage abs: ", sum(abs.(XY./N))/M)
    readline()
    return nothing 
end
#precompile(compute_logloss_delta!, (CuArray{Float16,2},
#                                    CuArray{Float16,2}))

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

function compute_poissonloss!(XY::CuArray, A::CuArray)
    return sum(XY.*A - exp.(XY))
end
precompile(compute_poissonloss!, (CuArray{Float16,2},
                                  CuArray{Float16,2}))

function compute_poissonloss_delta!(XY, A)
    XY .= exp.(XY)
    XY .*= -1.0
    XY .+= A
    return nothing
end
precompile(compute_poissonloss_delta!, (CuArray{Float16,2},
                                        CuArray{Float16,2}))

#########################################
# Other functions
function compute_loss!(XY, A, ql_idx, ll_idx, pl_idx)
    loss = 0.0
    if length(ql_idx) > 0
        loss += compute_quadloss!(XY[:,ql_idx], A[:,ql_idx])
    end
    if length(ll_idx) > 0
        loss += compute_logloss!(XY[:,ll_idx], A[:,ll_idx])
    end
    if length(pl_idx) > 0
        loss += compute_poissonloss!(XY[:,pl_idx], A[:,pl_idx])
    end
    return loss
end
precompile(compute_loss!, (CuArray{Float16,2}, CuArray{Float16,2},
                           CuVector{Int64}, CuVector{Int64},
                           CuVector{Int64}))


function compute_loss_delta!(XY, A, ql_idx, ll_idx, pl_idx)
    println("XY:")
    println("\tfrac nonzero: ", sum(XY .!= 0.0)/M/N)
    println("\taverage abs: ", sum(abs.(XY./N))/M)
    readline()
    if length(ql_idx) > 0
        compute_quadloss_delta!(XY[:,ql_idx], A[:,ql_idx])
    end
    if length(ll_idx) > 0
        compute_logloss_delta!(XY[:,ll_idx], A[:,ll_idx])
    end
    if length(pl_idx) > 0
        compute_poissonloss_delta!(XY[:,pl_idx], A[:,pl_idx])
    end
    return nothing
end
precompile(compute_loss_delta!, (CuArray{Float16,2}, CuArray{Float16,2},
                                 CuVector{Int64}, CuVector{Int64},
                                 CuVector{Int64}))


