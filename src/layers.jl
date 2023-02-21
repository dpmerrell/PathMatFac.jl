

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

function Base.getindex(cs::ColScale, idx1, idx2)
    return ColScale(cs.logsigma[idx2])
end


function ChainRulesCore.rrule(cs::ColScale, Z::AbstractMatrix)
    
    sigma = exp.(cs.logsigma)
    result = transpose(sigma).*Z

    function colscale_pullback(result_bar)
        Z_bar = transpose(sigma) .* result_bar
        logsigma_bar = vec(sum(Z_bar; dims=1))

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

function Base.getindex(cs::ColShift, idx1, idx2)
    return ColShift(cs.mu[idx2])
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

function BatchScale(col_batches, batch_dict)

    unq_cbi = unique(col_batches)
    values = [Dict(urb => 0.0 for urb in unique(batch_dict[k])) for k in unq_cbi]
    logdelta = BatchArray(col_batches, batch_dict, values)

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

function Base.getindex(bs::BatchScale, idx1, idx2)
    return view(bs, idx1, idx2)
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

function BatchShift(col_batches, batch_dict)
    
    unq_cbi = unique(col_batches)
    values = [Dict(urb => 0.0 for urb in unique(batch_dict[k])) for k in unq_cbi]
    theta = BatchArray(col_batches, batch_dict, values)

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

function Base.getindex(bs::BatchShift, idx1, idx2)
    return view(bs, idx1, idx2)
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


###########################################
# COMPOSE THE LAYERS
###########################################

mutable struct ViewableComposition
    layers::Tuple
end

@functor ViewableComposition

function (vc::ViewableComposition)(Z::AbstractMatrix)
    return reduce((f,g)->(x->g(f(x))), vc.layers)(Z)
end

function view(vc::ViewableComposition, idx1, idx2)
    return ViewableComposition(map(layer->(isa(layer, Function) ? layer : view(layer, idx1, idx2)), vc.layers))
end

function Base.getindex(vc::ViewableComposition, idx1, idx2)
    return ViewableComposition(map(layer-> isa(layer, Function) ? layer : layer[idx1, idx2], vc.layers))
end

function construct_model_layers(feature_views, batch_dict)

    N = length(feature_views)
    layer_ls = [ColScale(N), ColShift(N), x->x, x->x]
    unq_views = unique(feature_views)

    if batch_dict != nothing
        layer_ls[3] = BatchScale(feature_views, batch_dict)
        layer_ls[4] = BatchShift(feature_views, batch_dict)
    end
    layer_obj = ViewableComposition(Tuple(layer_ls))

    return layer_obj
end 


#############################################
# Selectively freeze a layer
#############################################

mutable struct FrozenLayer
    layer
end

@functor FrozenLayer

function (fl::FrozenLayer)(args...)
    fl.layer(args...)
end


function ChainRulesCore.rrule(fl::FrozenLayer, args...)

    # Treat the layer as a pure function; 
    # do not differentiate w.r.t. it
    res, pb = Zygote.pullback(fl.layer, args...)
    function FrozenLayer_pullback(res_bar)
        args_bar = pb(res_bar)
        return NoTangent(), args_bar...
    end

    return res, FrozenLayer_pullback 
end


function view(fl::FrozenLayer, idx1, idx2)
    return FrozenLayer(view(fl.layer, idx1, idx2))
end

function Base.getindex(fl::FrozenLayer, idx1, idx2)
    return FrozenLayer(fl.layer[idx1, idx2])
end


function freeze_layer!(vc::ViewableComposition, idx::Integer)
    if !isa(vc.layers[idx], FrozenLayer)
        vc.layers = (vc.layers[1:idx-1]..., 
                     FrozenLayer(vc.layers[idx]),
                     vc.layers[idx+1:end]...)
    end
end

function freeze_layer!(vc::ViewableComposition, idx::AbstractVector)
    for i in idx
        freeze_layer!(vc, i)
    end
end

function unfreeze_layer!(vc::ViewableComposition, idx::Integer)
    if isa(vc.layers[idx], FrozenLayer)
        vc.layers = (vc.layers[1:idx-1]..., 
                     vc.layers[idx].layer,
                     vc.layers[idx+1:end]...)
    end
end

function unfreeze_layer!(vc::ViewableComposition, idx::AbstractVector)
    for i in idx
        unfreeze_layer!(vc, i)
    end
end


