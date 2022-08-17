
#######################################
# Termination conditions
#######################################
function iter_termination(model, iter)
    return iter >= 10
end

function precision_termination(model, iter)

end

#######################################
# Callback structs
#######################################
mutable struct OuterCallback
    history::AbstractVector    
    history_json::String
end

function OuterCallback(; history_json="histories.json")
    return OuterCallback(Any[], history_json)
end

function (ocb::OuterCallback)(model::MultiomicModel, inner_callback)

    Y_pred = cpu(model.matfac.Y)
    Y_true = cpu(map( v->(!).(v), model.matfac.Y_reg.l1_feat_idx))
    #println(model.matfac.Y_reg.l1_feat_idx)
    #println(Y_true)
    pathway_av_precs = [average_precision(Y_pred[i,:], y_t) for (i, y_t) in enumerate(Y_true)]

    results = Dict("lambda_Y" => model.matfac.lambda_Y,
                   "history" => inner_callback.history,
                   "average_precisions" => pathway_av_precs)

    push!(ocb.history, results)

    open(ocb.history_json, "w") do f
        JSON.print(f, ocb.history)
    end
end


