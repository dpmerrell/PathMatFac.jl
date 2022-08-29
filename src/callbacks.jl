
####################################################
# Selection criteria (for hyperparameter selection)
####################################################

function latest_model(model, best_model, D, iter)
    return true
end

function precision_selection(model, best_model, D, iter)
    new_av_precs = model_Y_average_precs(model)
    best_av_precs = model_Y_average_precs(best_model)
    return minimum(new_av_precs) > minimum(best_av_precs)
end

########################################################
# Termination conditions (for hyperparameter selection)
########################################################
function iter_termination(model, best_model, D, iter)
    return iter >= 10
end

function precision_termination(model, best_model, D, iter; prec_threshold=0.25)
    pathway_av_precs = model_Y_average_precs(model)
    return minimum(pathway_av_precs) < prec_threshold
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

    pathway_av_precs = model_Y_average_precs(model) 

    results = Dict("lambda_Y" => model.matfac.lambda_Y,
                   "history" => inner_callback.history,
                   "average_precisions" => pathway_av_precs)

    push!(ocb.history, results)

    open(ocb.history_json, "w") do f
        JSON.print(f, ocb.history)
    end

end


