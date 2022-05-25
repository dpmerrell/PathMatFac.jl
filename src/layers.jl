

import Base: view


##################################
# Column Scale

mutable struct ColScale
    logsigma::AbstractVector
end

@functor ColScale

function ColScale(N::Int)
    return ColScale(zeros(N))
end


function (cs::ColScale)(Z::AbstractMatrix)
    return Z .* transpose(exp.(cs.logsigma))
end


function view(cs::ColScale, idx1, idx2)
    return ColScale(view(cs.logsigma, idx2))
end

function ChainRulesCore.rrule(cs::ColScale, Z::AbstractMatrix)
    
    sigma = exp.(cs.logsigma)
    result = transpose(sigma).*Z

    function colscale_pullback(result_bar)
        logsigma_bar = vec(sum(result_bar; dims=1)) .* sigma
        Z_bar = transpose(sigma) .* result_bar
        return ChainRulesCore.Tangent{ColScale}(logsigma=logsigma_bar), Z_bar
    end

    return result, colscale_pullback

end

##################################
# Column Shift

mutable struct ColShift
    mu::AbstractVector
end

@functor ColShift

function ColShift(N::Int)
    return ColShift(randn(N) .* 1e-4)
end


function (cs::ColShift)(Z::AbstractMatrix)
    return Z .+ transpose(cs.mu)
end


function view(cs::ColShift, idx1, idx2)
    return ColShift(view(cs.mu, idx2))
end

function ChainRulesCore.rrule(cs::ColShift, Z::AbstractMatrix)
    
    result = transpose(cs.mu) .+ Z

    function colshift_pullback(result_bar)
        mu_bar = vec(sum(result_bar; dims=1))
        Z_bar = copy(result_bar)
        return ChainRulesCore.Tangent{ColShift}(mu=mu_bar), Z_bar
    end

    return result, colshift_pullback

end

#################################
# Batch Scale

mutable struct BatchScale
    logdelta::BatchArray
end

@functor BatchScale

function BatchScale(col_batches, row_batches)

    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    logdelta = BatchArray(col_batches, row_batches, values)

    return BatchScale(logdelta)
end


function (bs::BatchScale)(Z::AbstractMatrix)
    return Z * exp(bs.logdelta)
end


function view(bs::BatchScale, idx1, idx2)
    if typeof(idx2) == Colon
        idx2 = 1:bs.logdelta.col_ranges[end].stop
    end
    return BatchScale(view(bs.logdelta, idx1, idx2)) 
end


function ChainRulesCore.rrule(bs::BatchScale, Z::AbstractMatrix)
    
    result, pb = Zygote.pullback((z,d) -> z*exp(d), Z, bs.logdelta)

    function batchscale_pullback(result_bar)
        Z_bar, logdelta_bar = pb(result_bar) 
        return ChainRulesCore.Tangent{BatchScale}(logdelta=logdelta_bar),
               Z_bar
    end

    return result, batchscale_pullback
end


##################################
# Batch Shift

mutable struct BatchShift
    theta::BatchArray
end

@functor BatchShift

function BatchShift(col_batches, row_batches)
    
    values = [Dict(urb => 0.0 for urb in unique(rbv)) for rbv in row_batches]
    theta = BatchArray(col_batches, row_batches, values)

    return BatchShift(theta)
end


function (bs::BatchShift)(Z::AbstractMatrix)
    return Z + bs.theta
end


function view(bs::BatchShift, idx1, idx2)
    if typeof(idx2) == Colon
        idx2 = 1:bs.theta.col_ranges[end].stop
    end
    return BatchShift(view(bs.theta, idx1, idx2))
end


function ChainRulesCore.rrule(bs::BatchShift, Z::AbstractMatrix)
    
    result, pb = Zygote.pullback((z,t) -> z + t, Z, bs.theta)

    function batchshift_pullback(result_bar)
        Z_bar, theta_bar = pb(result_bar) 
        return ChainRulesCore.Tangent{BatchShift}(theta=theta_bar),
               Z_bar
    end

    return result, batchshift_pullback
end


#########################################
# Define a PMLayers functor
#########################################
mutable struct PMLayers
    cscale::ColScale
    cshift::ColShift
    bscale::BatchScale
    bshift::BatchShift
end

@functor PMLayers

function PMLayers(model_assays, sample_batch_ids)

    N = length(model_assays)

    return PMLayers(ColScale(N),
                    ColShift(N),
                    BatchScale(model_assays, sample_batch_ids),
                    BatchShift(model_assays, sample_batch_ids))
end

function (bmf::PMLayers)(Z::AbstractMatrix)
    return bmf.bshift(
            bmf.bscale(
             bmf.cshift(
              bmf.cscale(Z)
             )
            )
           )
end


function view(l::PMLayers, idx1, idx2)
    return PMLayers(view(l.cscale, idx1, idx2),
                             view(l.cshift, idx1, idx2),
                             view(l.bscale, idx1, idx2),
                             view(l.bshift, idx1, idx2))
end



