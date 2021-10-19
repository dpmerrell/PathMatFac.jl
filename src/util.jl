
using GPUMatFac
import GPUMatFac: LogisticLoss, QuadLoss, PoissonLoss, NoLoss

export DEFAULT_ASSAYS, sort_features


#DEFAULT_ASSAY_LOSSES = Dict("cna" => LogisticLoss,
#                           "mutation" => LogisticLoss,
#                           "methylation" => QuadLoss,
#                           "mrnaseq" => PoissonLoss, 
#                           "rppa" => QuadLoss
#                          )
DEFAULT_ASSAY_LOSSES = Dict("cna" => QuadLoss,
                           "mutation" => LogisticLoss,
                           "methylation" => QuadLoss,
                           "mrnaseq" => QuadLoss, 
                           "rppa" => QuadLoss
                          )



DEFAULT_ASSAYS = collect(keys(DEFAULT_OMIC_LOSSES))
DEFAULT_ASSAY_SET = Set(DEFAULT_ASSAYS)


DEFAULT_ASSAY_MAP = Dict("cna" => ["dna", 1],
                        "mutation" => ["dna", -1],
                        "methylation" => ["mrna", -1],
                        "mrnaseq" => ["mrna", 1],
                        "rppa" => ["protein", 1]
                        )


PWY_SIF_CODE = Dict("a" => "activation",
                    "b" => "abstract",
                    "c" => "compound",
                    "h" => "chemical",
                    "p" => "protein",
                    "f" => "family"
                   )



function value_to_idx(values)
    return Dict(v => idx for (idx, v) in enumerate(values))
end


function keymatch(l_keys, r_keys)

    rkey_to_idx = value_to_idx(r_keys) 
    rkeys = Set(keys(rkey_to_idx))

    l_idx = []
    r_idx = []

    for (i, lk) in enumerate(l_keys)
        if lk in rkeys
            push!(l_idx, i)
            push!(r_idx, rkey_to_idx[lk])
        end
    end

    return l_idx, r_idx
end

function get_loss(feature; assay_set=DEFAULT_ASSAY_SET, 
                           assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature; assay_set=assay_set)
    if assay in keys(assay_losses)
        return assay_losses[assay]
    else
        return NoLoss
    end
end

function srt_get_loss(feature; assay_set=DEFAULT_ASSAY_SET,
                               assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature; assay_set=assay_set)
    if assay in keys(assay_losses)
        return string(assay_losses[assay]), assay, feature
    else
        return "", "", feature_name
    end
end


function get_assay(feature; assay_set=DEFAULT_ASSAY_SET)
    assay = feature[2] 
    if assay in assay_set
        return assay
    else
        return ""
    end
end


function sort_features(features; assay_losses=DEFAULT_ASSAY_LOSSES)
    tuple_list = [srt_get_loss(feat; assay_losses=assay_losses) for feat in features]
    sort!(tuple_list)
    return [t[end] for t in tuple_list]
end




