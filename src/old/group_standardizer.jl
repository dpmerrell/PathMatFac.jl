
using Statistics

export GroupStandardizer, fit!, transform, inv_transform

# The GroupStandardizer shifts and scales
# the columns of a dataset, yielding
# transformed columns with mu=0, sigma=1.
# It acts on *groups* of rows; that is,
# it computes a mu and sigma for each group.
# Should only be applied to "normal" data.
mutable struct GroupStandardizer
    std_params::Dict
end


GroupStandardizer() = GroupStandardizer(Dict())

# Some helpers
nanmean(x) = mean(filter(!isnan, x))
nanmean(x, dims) = mapslices(nanmean, x; dims=dims)
nanstd(x) = std(filter(!isnan, x))
nanstd(x, dims) = mapslices(nanstd, x; dims=dims)


function fit!(gs::GroupStandardizer, X::AbstractArray, groups::Vector)

    std_params = Dict()

    m, n = size(X)
    gp_hierarchy = get_instance_hierarchy(collect(1:m), groups)

    for (gp, instances) in gp_hierarchy

        group_mus = nanmean(X[instances,:], 1)
        group_sigmas = nanstd(X[instances,:], 1)

        std_params[gp] = Dict( "sigma" => group_sigmas,
                               "mu" => group_mus)
    end
    gs.std_params = std_params

end


function transform(gs::GroupStandardizer, X::AbstractArray, groups::Vector)

    m, n = size(X)
    gp_hierarchy = get_instance_hierarchy(collect(1:m), groups)

    for (gp, instances) in gp_hierarchy
    
        if gp in keys(gs.std_params)
            X[instances,:] .= (X[instances,:] .- gs.std_params[gp]["mu"]) ./ 
                               gs.std_params[gp]["sigma"]
        else
            if size(instances,1) == 1
                println("WARNING: only one instance in group ", gp)
            end
            mus = nanmean(X[instances,:])
            sigmas = nanstd(X[instances,:])
            X[instances,:] .= (X[instances,:] .- mus)./sigmas
        end
    end
    return X
end


function inv_transform(gs::GroupStandardizer, X::AbstractArray, groups::Vector)

    m, n = size(X)
    gp_hierarchy = get_instance_hierarchy(collect(1:m), groups)

    for (gp, instances) in gp_hierarchy
    
        if gp in keys(gs.std_params)
            X[instances,:] .= X[instances,:].*gs.std_params[gp]["sigma"] .+ gs.std_params[gp]["mu"]
        else
            if size(instances,1) == 1
                println("WARNING: only one instance in group ", gp)
            end
            mus = nanmean(X[instances,:])
            sigmas = nanstd(X[instances,:])
            X[instances,:] .= X[instances,:].*sigmas .+ mus
        end
    end
    return X
end





