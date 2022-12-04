

using PlotlyJS, JSON, DataFrames

function history_plot(full_ls, k)

    v = [step[k] for stage in full_ls for step in stage["history"]]

    tr = scatter(y=v, name=k)

    return tr
end


function total_loss_plot(full_ls)
    
    v = [sum(filter(x -> x != nothing, collect(values(step)))) for stage in full_ls for step in stage["history"]]
    return scatter(y=v, name="Total loss")

end


function history_plots(full_ls)

    ks = keys(full_ls[1]["history"][1])
    n_keys = length(ks)

    traces = GenericTrace[]
    for k in ks
        push!(traces, history_plot(full_ls, k))
    end

    return traces
end


function aucpr_plot(full_ls)
    M = length(full_ls)
    N = length(full_ls[1]["average_precisions"])
    X = zeros(M, N)
    for (k, stage) in enumerate(full_ls)
        X[k,:] .= stage["average_precisions"]
    end

    df = DataFrame(X, :auto)
    traces = GenericTrace[]
    for col in names(df)
        push!(traces, scatter(y=df[:,col], name=col,
                              line=attr(width=0.5),
                              showlegend=false
                             ),
             )
    end

    return traces
end


function aucpr_boxplot(full_ls)
    M = length(full_ls[1]["average_precisions"])
    N = length(full_ls)
    X = zeros(M, N)
    for (k, stage) in enumerate(full_ls)
        X[:,k] .= stage["average_precisions"]
    end

    df = DataFrame(X, :auto)
    traces = GenericTrace[]
    for (i, col) in enumerate(names(df))
        push!(traces, box(y=df[:,col], name=string("iteration ",i),
                          kind="box",
                          boxpoints="all",
                          showlegend=false,
                          ),
             )
    end

    return traces
end

function generate_plot(full_ls)

    history_traces = history_plots(full_ls)
    total_loss_trace = total_loss_plot(full_ls)
    push!(history_traces, total_loss_trace)

    aucpr_traces = aucpr_boxplot(full_ls) 

    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.5,0.5],
                        column_widths=[1.0],
                        vertical_spacing=0.1,
                        subplot_titles=reshape(["Loss Contributions";
                                                "Y-Matrix AUCPR Scores"], (2,1))
                       )

    # Plotly's addtraces! differs from add_trace!
    # in weird ways. So we'll add_trace!s individually,
    # which seems kind of clumsy.
    for tr in history_traces
        add_trace!(fig, tr; row=1, col=1)
    end
    for tr in aucpr_traces
        add_trace!(fig, tr; row=2, col=1)
    end
    return fig 
end


function main(args)

    history_json = args[1]
    out_html = args[2]

    full_ls = JSON.parsefile(history_json)
    fig = generate_plot(full_ls)

    open(out_html, "w") do io
        PlotlyBase.to_html(io, fig.plot)
    end
end


main(ARGS)


