using PyPlot
using CSV
using DataFrames
using SparseArrays
using Statistics
using LinearAlgebra
using PathwayMultiomics
#using Profile
#using ProfileView


data = CSV.read("mnist_matrix.tsv", DataFrame; header=false, delim="\t");
data = convert(Matrix, data);
pixel_vals = data[:,1:end-1];
labels = data[:,end];
classes = sort(unique(labels));

unflatten_mnist(vec) = transpose(reshape(vec, (28,28)))

#matshow(unflatten_mnist(pixel_vals[110,:]))

function coord_to_idx(x, y, H)
    return (x-1)*H + y
end

function idx_to_pixel_coord(idx, H)
    return (div((idx-1),H), ((idx-1) % H))
end

function build_pixel_graph(H, W)
    pixel_graph = [Int64[] for idx=1:(H*W)]
    
    for x=1:W
        for y=1:H
            u_idx = coord_to_idx(x,y, H)
            
            # Left
            if x > 1
                v_idx = coord_to_idx(x-1, y, H)
                push!(pixel_graph[u_idx], v_idx)
                push!(pixel_graph[v_idx], u_idx)
            end
            # Up
            if y > 1
                v_idx = coord_to_idx(x, y-1, H)
                push!(pixel_graph[u_idx], v_idx)
                push!(pixel_graph[v_idx], u_idx)
            end
            
        end
    end
    
    return pixel_graph
end

function plot_graph(g, idx_to_coord_func)
    for (u, neighbors) in enumerate(g)
        u_xy = idx_to_coord_func(u)
        for v in neighbors
            if v < u
                v_xy = idx_to_coord_func(v)
                plot([u_xy[1], v_xy[1]], [u_xy[2], v_xy[2]])
            end
        end
    end
end

pixel_graph = build_pixel_graph(28,28);
mnist_idx_to_coord_func(idx) = idx_to_pixel_coord(idx,28)

function build_instance_graph(labels)
    
    classes = sort(unique(labels))
    class_to_idx = Dict(cls => length(labels)+i for (i, cls) in enumerate(classes))
    n_classes = length(classes)
                                      # Leaf nodes;  # class nodes;  # root node
    instance_graph = [Int64[] for i=1:(length(labels) + n_classes)] # + 1) ];

    for (leaf_idx, label) in enumerate(labels)
        cls_idx = class_to_idx[label]
        push!(instance_graph[leaf_idx], cls_idx)
        push!(instance_graph[cls_idx], leaf_idx)
    end
    
    #root_idx = length(instance_graph)
    #for cls in classes
    #    cls_idx = class_to_idx[cls]
    #    push!(instance_graph[cls_idx], root_idx)
    #    push!(instance_graph[root_idx], cls_idx)
    #end
    
    return instance_graph
end

instance_graph = build_instance_graph(labels);

println("BUILT PIXEL AND INSTANCE GRAPHS")

aug_pixel_vals = [pixel_vals; zeros(length(classes), 28*28)];

function graph_to_spmat(graph; epsilon=0.001)
   
    I = Int64[]
    J = Int64[]
    V = Float64[]
    N = length(graph)

    diag_entries = fill(epsilon, N)
    
    for (u, neighbors) in enumerate(graph)
        for v in neighbors
            push!(I, u)
            push!(J, v)
            push!(V, -1.0)
        end
        diag_entries[u] += length(neighbors)
    end
    for (i, v) in enumerate(diag_entries)
        push!(I, i)
        push!(J, i)
        push!(V, v)
    end
    
    mat = sparse(I, J, V, N, N)

    standardizer = sparse(1:N, 1:N, 1.0./sqrt.(diag_entries), N, N)
    mat = standardizer * mat * standardizer

    return mat
end

pixel_spmat = graph_to_spmat(pixel_graph);
instance_spmat = graph_to_spmat(instance_graph);

println("BUILT PIXEL AND INSTANCE MATRICES")

# Have an unregularized "offset" factor
k = 10 
instance_spmat_vec = [copy(instance_spmat) for i=1:k]
feature_spmat_vec = [copy(pixel_spmat) for i=1:k-1]
losses = [LogisticLoss(1.0) for i=1:28*28]
model = MatFacModel(instance_spmat_vec, feature_spmat_vec, losses; K=k);
model.X[k,:] .= 1.0


println("INITIALIZED MODEL")

BLAS.set_num_threads(1)

println("ABOUT TO FIT")
#fit!(model, aug_pixel_vals; inst_reg_weight=1.0, feat_reg_weight=1.0, max_iter=50, lr=0.005)
#fit_cuda!(model, aug_pixel_vals; inst_reg_weight=0.1, feat_reg_weight=0.1, max_iter=1000, lr=0.1, K_opt_X=(k-1), K_opt_Y=k)
fit_cuda!(model, aug_pixel_vals; inst_reg_weight=0.1, feat_reg_weight=0.1, max_iter=2000, loss_iter=1, lr=0.5, momentum=0.8, K_opt_X=(k-1), K_opt_Y=k, rel_tol=1e-9)

labels = convert(Vector{Int64}, labels)
colors = collect(keys(PyPlot.colorsm.TABLEAU_COLORS))

function embedding_scatter(X)
    F = LinearAlgebra.svd(X)
    M = length(labels)
    for lab in sort(unique(labels))
        lab_idx = findall(labels .== lab) 
        scatter3D(F.Vt[1, lab_idx], F.Vt[2,lab_idx], F.Vt[3,lab_idx], color=colors[lab+1], label=string(lab))
        #scatter(F.Vt[1, lab_idx], F.Vt[2,lab_idx], color=colors[lab+1], label=string(lab))
    end
    legend()
end

embedding_scatter(model.X)
savefig("embedding_scatter.png", dpi=200)
#clf()

#logistic(x) = 1.0 ./ (1.0 .+ exp.(-x))

for i=1:k
    matshow(unflatten_mnist(model.Y[i,:]))
    colorbar()
    savefig(string("embedding_basis_",i,".png"), dpi=200)
    #clf()
end

#Threads.nthreads()

#for i=1:20
#    matshow(unflatten_mnist(transpose(model.Y) * model.X[:,109+i]))
#end


