
using PlotlyJS
using PlotlyBase
using PathwayMultiomics
using LinearAlgebra

PM = PathwayMultiomics

function nice_feature_ids(feature_ids, feature_views)
    better_ids = feature_ids
    if all(map(f->isa(f, Number), feature_ids))
        better_ids = map((f,v)->string(v, " ", f), feature_ids, feature_views)
    end
    return better_ids
end

function vis_embedding!(traces, labels, model::PM.PathMatFacModel)

    sample_conditions = model.sample_conditions
    unq_conditions = unique(sample_conditions)
    #cond2color = Dict([c=>i for (i,c) in enumerate(unq_conditions)])
    #sample_colors = map(c->cond2color[c], sample_conditions)

    for uc in unq_conditions
        push!(labels, "Embedded data (X)")
        idx = (sample_conditions .== uc)
        tr = scatter(x=model.matfac.X[1,idx],
                     y=model.matfac.X[2,idx],
                     z=model.matfac.X[3,idx],
                     type="scatter3d", 
                     mode="markers", name=uc)
        push!(traces, tr)
    end 
end


function vis_X_matrix!(traces, labels, model::PM.PathMatFacModel)
    K, M = size(model.matfac.X)
    sample_ids = model.sample_ids
    sample_conditions = model.sample_conditions
    combined_ids = map((i,c) -> string(i, "_", c), sample_ids, sample_conditions)

    push!(labels, "Embedded data (X)")
    push!(traces, 
          heatmap(z=model.matfac.X,
                  x=combined_ids,
                  y=collect(1:K),
                  type="heatmap", colorscale="Greys", reversescale=true)
         )
end


function vis_factors!(traces, labels, model::PM.PathMatFacModel)

    K,N = size(model.matfac.Y)
    feature_ids = nice_feature_ids(model.feature_ids, model.feature_views)

    for k=1:K
        push!(labels, "Linear factors (Y)")

        push!(traces,
              scatter(x=feature_ids, y=model.matfac.Y[k,:],
                      mode="lines", name=string("factor ",k)
                     )
             )
    end
end


function vis_factors_matrix!(traces, labels, model::PM.PathMatFacModel)

    push!(labels, "Linear factors (Y); matrix view")

    Y = model.matfac.Y
    K = size(Y,1)
    feature_ids = model.feature_ids

    trace = heatmap(z=float.(Y), x=feature_ids, y=collect(1:K), type="heatmap", colorscale="Greys", reversescale=true)
    push!(traces, trace)

    return trace
end

function vis_explained_variance!(traces, labels, model::PathMatFacModel)

    push!(labels, "Explained variance")

    norm_sq = vec(sum(model.matfac.Y.^2, dims=2))
    K = length(norm_sq)

    push!(traces,
          scatter(x=collect(1:K), y=norm_sq, mode="lines", name="Explained variance")
         )

end

function vis_mu!(traces, labels, model::PM.PathMatFacModel)

    N = size(model.matfac.Y,2)
    feature_ids = nice_feature_ids(model.feature_ids, model.feature_views)
    push!(labels, "Column shifts (μ)")
    push!(traces,
          scatter(x=feature_ids, y=model.matfac.col_transform.layers[3].mu,
                  mode="lines", name="Column shifts")
         )
end


function vis_sigma!(traces, labels, model::PM.PathMatFacModel)

    K,N = size(model.matfac.Y)
    feature_ids = nice_feature_ids(model.feature_ids, model.feature_views)
    push!(labels, "Column scales (σ)")
    push!(traces,
          scatter(x=feature_ids, y=sqrt(K).*exp.(model.matfac.col_transform.layers[1].logsigma),
                  mode="lines", name="Column scales")
         )
end


function vis_batch_shift!(traces, labels, model::PathMatFacModel)

    if isa(model.matfac.col_transform.layers[4], PM.BatchShift)
        theta = model.matfac.col_transform.layers[4].theta
        f_ids = nice_feature_ids(model.feature_ids, model.feature_views)

        for (cr, n, v) in zip(theta.col_ranges, theta.col_range_ids, theta.values)
            x = f_ids[cr]
            for k=1:size(v,1)
                push!(labels, "Batch shift (θ)")
                push!(traces,
                      scatter(x=x, y=v[k,:], mode="lines", name=string(n, " ", k))
                     )
            end
        end
    end
end


function vis_batch_scale!(traces, labels, model::PathMatFacModel)

    if isa(model.matfac.col_transform.layers[2], PM.BatchScale)
        logdelta = model.matfac.col_transform.layers[2].logdelta
        f_ids = nice_feature_ids(model.feature_ids, model.feature_views)

        for (cr, n, v) in zip(logdelta.col_ranges, logdelta.col_range_ids, logdelta.values)
            x = f_ids[cr]
            for k=1:size(v,1)
                push!(labels, "Batch scale (δ)")
                push!(traces,
                      scatter(x=x, y=exp.(v[k,:]), mode="lines", name=string(n, " ", k))
                     )
            end
        end
    end
end

function vis_assignments!(traces, labels, model::PathMatFacModel)
    if isa(model.matfac.Y_reg, PM.FeatureSetARDReg)
        reg = model.matfac.Y_reg
        L,K = size(reg.A)
        push!(labels, "Pathway-factor assignments (A)")
        push!(traces, 
              heatmap(z=transpose(reg.A),
                      x=reg.featureset_ids,
                      y=collect(1:K),
                      #y=collect(1:L),
                      type="heatmap", colorscale="Greys", reversescale=true
                     )
             )
    end
end


function make_buttons(labels)

    unq_labels = unique(labels)
    n_buttons = length(unq_labels)

    n_traces = length(labels)

    indicators = [(labels .== ul) for ul in unq_labels]
 
    buttons = [attr(label=l, method="update",
                    args=[attr(visible=indicators[i]),
                          attr(title=l,
                               annotations=[]
                               )
                          ]
                    ) for (i,l) in enumerate(unq_labels)]

    return buttons
end


function generate_plots(model, flag)
    traces = PlotlyBase.AbstractTrace[]
    labels = []

    if flag == "embedding"
        vis_embedding!(traces, labels, model)
    else
        vis_X_matrix!(traces, labels, model)
        vis_factors!(traces, labels, model)
        vis_factors_matrix!(traces, labels, model)
        vis_explained_variance!(traces, labels, model)
        vis_mu!(traces, labels, model)
        vis_batch_shift!(traces, labels, model)
        vis_batch_scale!(traces, labels, model)
        vis_sigma!(traces, labels, model)
        vis_assignments!(traces, labels, model)
    end

    buttons = make_buttons(labels)  
 
    layout = Layout(
                 updatemenus=[
                     attr(
                         active=-1,
                         buttons=buttons
                         )
                             ]
                   )

    fig = plot(traces, layout)

    return fig 
end


function main(args)

    model_bson = args[1]
    out_html = args[2]

    flag = nothing
    if length(args) > 2
        flag = args[3]
    end

    model = PM.load_model(model_bson)

    fig = generate_plots(model, flag)
    
    open(out_html, "w") do io
        PlotlyBase.to_html(io, fig.plot)
    end
end

main(ARGS)

