
using GPUMatFac
import GPUMatFac: LogisticLoss, QuadLoss, PoissonLoss, NoLoss

export DEFAULT_OMICS, sort_features


DEFAULT_OMIC_LOSSES = Dict("cna" => LogisticLoss,
                           "mutation" => LogisticLoss,
                           "methylation" => QuadLoss,
                           "mrnaseq" => PoissonLoss, 
                           "rppa" => QuadLoss
                          )

DEFAULT_OMICS = collect(keys(DEFAULT_OMIC_LOSSES))
DEFAULT_OMIC_SET = Set(DEFAULT_OMICS)


DEFAULT_OMIC_MAP = Dict("cna" => ["dna", 1],
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

function get_loss(feature_name; omic_type_set=DEFAULT_OMIC_SET, 
                                omic_losses=DEFAULT_OMIC_LOSSES)
    omic_type = get_omic_type(feature_name; omic_type_set=omic_type_set)
    if omic_type in keys(omic_losses)
        return omic_losses[omic_type]
    else
        return NoLoss
    end
end

function srt_get_loss(feature_name; omic_type_set=DEFAULT_OMIC_SET,
                                    omic_losses=DEFAULT_OMIC_LOSSES)
    omic_type = get_omic_type(feature_name)
    if omic_type in keys(omic_losses)
        return string(omic_losses[omic_type]), omic_type, feature_name
    else
        return "", "", feature_name
    end
end


function get_omic_type(feature_name; omic_type_set=DEFAULT_OMIC_SET)
    suff = split(feature_name, "_")[end] 
    if suff in omic_type_set
        return suff
    else
        return ""
    end
end


function sort_features(feature_names; omic_losses=DEFAULT_OMIC_LOSSES)
    tuple_list = [srt_get_loss(feat; omic_losses=omic_losses) for feat in feature_names]
    sort!(tuple_list)
    return [t[end] for t in tuple_list]
end




