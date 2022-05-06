
import BSON

export save_model, load_model


function save_model(filename, model::MultiomicModel)
    BSON.@save filename model
end

function load_model(filename::String)
    d = BSON.load(filename, @__MODULE__) 
    return d[:model]
end




