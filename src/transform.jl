
import ScikitLearnBase: transform

export transform


function transform(model::MatFacModel, A::AbstractMatrix;
                   inst_reg_weight=1.0, feat_reg_weight=1.0,
                   max_iter::Int=100, lr::Float64=0.001, 
                   abs_tol::Float64=1e-6, rel_tol::Float64=1e-5)

    # Setup
    loss = Inf
    iter = 0

    M = size(A,1)
    N = size(A,2)
    K = size(model.X, 1)

    newX = 0.001*randn(K,M)

    # graph coloring
    x_union_graph = graph_union(model.instance_reg_mats)
    x_colors = compute_coloring(x_union_graph) 

    y_union_graph = graph_union(model.feature_reg_mats)
    y_colors = compute_coloring(y_union_graph)


    while iter < max_iter # and abs(loss - new_loss) > abs_tol and abs(loss - new_loss)/loss > rel_tol

        # Update X columns
        # Loop over rows of data
        for x_color in x_colors 
            Threads.@threads for i in x_color

                # Accumulate loss gradients
                g = zeros(K)
                #X_view = view(model.X,:,i)
                xtY = BLAS.gemv('T', model.Y, newX[:,i])
                for j=1:N
                    if !isnan(A[i,j])
                        accum_grad_x!(g, model.losses[j], model.Y[:,j], xtY[j], A[i,j])
                    end 
                end
                # Add regularizer gradient
                g += inst_reg_weight .* reg_grad(newX, i, model.instance_reg_mats)

                # Update with gradient

                newX[:,i] -= lr.*g
            end # i loop
        end # x_color loop

        iter += 1
        println(iter)

    end # while

end


