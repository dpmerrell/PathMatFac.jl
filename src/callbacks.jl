
####################################################
# Selection criteria (for hyperparameter selection)
####################################################

function latest_model(model, best_model, D, iter)
    return true
end

"""
    precision_selection(model, best_model, D, iter;
                        prec_threshold=0.8)

    Determine whether the new model has (a) lower data loss
    while (b) maintaining the Y matrix average precision
    above a certain threshold. 
"""
function precision_selection(model, best_model, D, iter; 
                             qntl=0.25, prec_threshold=0.75, capacity=Int(25e6))
    best_loss = MF.batched_data_loss(best_model.matfac, D; capacity=capacity)
    new_loss = MF.batched_data_loss(model.matfac, D; capacity=capacity)
    new_av_precs = model_Y_average_precs(model)
    return (quantile(new_av_precs, qntl) > prec_threshold) & (new_loss < best_loss)
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


