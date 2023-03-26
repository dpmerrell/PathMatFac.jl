
using PlotlyJS
using PlotlyBase
using PathwayMultiomics
using LinearAlgebra

PM = PathwayMultiomics


function vis_embedding!(traces, labels, model::PM.PathMatFacModel)

    push!(labels, "Embedded data (X)")

    sample_conditions = model.sample_conditions
    unq_conditions = unique(sample_conditions)
    cond2color = Dict([c=>i for (i,c) in enumerate(unq_conditions)])
    sample_colors = map(c->cond2color[c], sample_conditions)

    tr = scatter(x=model.matfac.X[1,:],
                 y=model.matfac.X[2,:],
                 z=model.matfac.X[3,:],
                 color=sample_colors,
                 type="scatter3d", mode="markers")
    push!(traces, tr)
 
end


function vis_factors!(traces, labels, model::PM.PathMatFacModel)

    K,N = size(model.matfac.Y)
    feature_ids = model.feature_ids

    for k=1:K
        push!(labels, "Linear factors (Y)")

        push!(traces,
              scatter(x=collect(1:N), y=model.matfac.Y[k,:],
                      mode="lines", name=string("factor ",k)
                     )
             )
    end

end


function vis_mu!(traces, labels, model::PM.PathMatFacModel)

    N = size(model.matfac.Y,2)
    push!(labels, "Column shifts (μ)")
    push!(traces,
          scatter(x=collect(1:N), y=model.matfac.col_transform.layers[3].mu,
                  mode="lines", name="Column shifts")
         )
end


function vis_sigma!(traces, labels, model::PM.PathMatFacModel)

    N = size(model.matfac.Y,2)
    push!(labels, "Column scales (σ)")
    push!(traces,
          scatter(x=collect(1:N), y=exp.(model.matfac.col_transform.layers[1].logsigma),
                  mode="lines", name="Column scales")
         )
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


function generate_plots(model)
    traces = PlotlyBase.AbstractTrace[]
    labels = []

    vis_embedding!(traces, labels, model)
    vis_factors!(traces, labels, model)
    vis_mu!(traces, labels, model)
    vis_sigma!(traces, labels, model)

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

    model = PM.load_model(model_bson)

    fig = generate_plots(model)
    
    open(out_html, "w") do io
        PlotlyBase.to_html(io, fig.plot)
    end
end

main(ARGS)

