
import BSON

export save_model, load_model

################################
# Save/load BSON
################################
function save_model(model::PathMatFacModel, filename; save_data=false)
    if !save_data
        model.data = nothing
    end
    BSON.@save filename model
end

function load_model(filename::String)
    d = BSON.load(filename, @__MODULE__) 
    return d[:model]
end



