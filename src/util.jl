

export DEFAULT_ASSAYS, sort_features


DEFAULT_ASSAY_LOSSES = Dict("cna" => "logistic",
                            "mutation" => "logistic",
                            "methylation" => "normal",
                            "mrnaseq" => "normal", 
                            "rppa" => "normal"
                            )



DEFAULT_ASSAYS = collect(keys(DEFAULT_ASSAY_LOSSES))
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


function get_assay(feature; assay_set=DEFAULT_ASSAY_SET)
    return feature[2] 
end


function get_loss(feature; assay_set=DEFAULT_ASSAY_SET, 
                           assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature; assay_set=assay_set)
    if assay in keys(assay_losses)
        return assay_losses[assay]
    else
        return "noloss"
    end
end


function srt_get_loss(feature; assay_set=DEFAULT_ASSAY_SET,
                               assay_losses=DEFAULT_ASSAY_LOSSES)
    assay = get_assay(feature; assay_set=assay_set)
    gene = feature[1]
    if assay in keys(assay_losses)
        return assay_losses[assay], assay, gene 
    else
        return "", "", gene
    end
end


function sort_features(features; assay_losses=DEFAULT_ASSAY_LOSSES)
    tuple_list = [srt_get_loss(feat; assay_losses=assay_losses) for feat in features]
    sort!(tuple_list)
    return [(t[end], t[2]) for t in tuple_list]
end


function nansum(x)
    return sum(filter(!isnan, x))
end


function nanmean(x)
    return mean(filter(!isnan, x))
end


function nanvar(x)
    return var(filter(!isnan, x))
end


