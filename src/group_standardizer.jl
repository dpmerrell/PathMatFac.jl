

export GroupStandardizer, fit_transform!

mutable struct GroupStandardizer
    std_params::Dict
    groups::Vector
end


GroupStandardizer() = GroupStandardizer(Dict(),String[])


nanmean(x) = mean(filter(!isnan, x))
nanmean(x, dims) = mapslices(nanmean, x; dims=dims)

nanstd(x) = std(filter(!isnan, x))
nanstd(x, dims) = mapslices(nanstd, x; dims=dims)


function fit_transform!(gs::GroupStandardizer, X::Matrix, 
                        losses::Vector, groups::Vector;
                        shift=true, scale=true)

    std_params = Dict()

    m, n = size(X)
    gp_hierarchy = get_instance_hierarchy(collect(1:m), groups)

    for (gp, instances) in gp_hierarchy

        patient_idx = Int[patient_to_idx[pat] for pat in p_vec]
       
        group_mus = zeros(n)
        if shift 
            group_mus = nanmean(X[instances,:], 1)
        end
        group_sigmas = ones(n)
        if scale
            group_sigmas = nanstd(X[instances,:], 1)
        end

        X[instances,:] = (X[instances,:] .- group_mus) ./ group_sigmas

        std_params[gp] = Dict( "sigma" => group_sigmas,
                               "mu" => group_mus)
    end

    gs.std_params = std_params

    return A
end


