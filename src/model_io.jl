

import BSON

export save_model, load_model


function save_model(filename, model::MultiomicModel)
    BSON.@save filename model
end

function load_model(filename::MultiomicModel)
    BSON.@load filename model
    return model
end




