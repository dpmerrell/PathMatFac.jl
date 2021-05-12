using PyPlot
using CSV
using DataFrames
using SparseArrays
using Statistics
using LinearAlgebra
using PathwayMultiomics

data = CSV.read("brown_dataset_bow.tsv", DataFrame; delim="\t");
N = size(data,2)-1

bags_of_words = convert(Matrix, data[:,1:N])
labels = convert(Vector, data[:,N+1])

println("BAGS OF WORDS: ", size(bags_of_words))

classes = sort(unique(labels));

function build_instance_graph(labels)
    
    classes = sort(unique(labels))
    class_to_idx = Dict(cls => length(labels)+i for (i, cls) in enumerate(classes))
    n_classes = length(classes)
    n_bk = n_classes + 1
                                      # Leaf nodes;  # class nodes;  # root node
    instance_graph = [Int64[] for i=1:(length(labels) + n_bk) ];

    for (leaf_idx, label) in enumerate(labels)
        cls_idx = class_to_idx[label]
        push!(instance_graph[leaf_idx], cls_idx)
        push!(instance_graph[cls_idx], leaf_idx)
    end
    
    root_idx = length(instance_graph)
    for cls in classes
        cls_idx = class_to_idx[cls]
        push!(instance_graph[cls_idx], root_idx)
        push!(instance_graph[root_idx], cls_idx)
    end
    
    return instance_graph
end


instance_graph = build_instance_graph(labels);
aug_X = [bags_of_words; fill(NaN, length(classes)+1, N)];

println("INSTANCE GRAPH: ", size(instance_graph))

println("AUGMENTED X: ", size(aug_X))

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


############################################
# CONSTRUCT THE MODEL
k = 10 

instance_spmat = graph_to_spmat(instance_graph);
instance_spmat_vec = [copy(instance_spmat) for i=1:k]
feature_spmat_vec = [sparse(I, N, N) for i=1:k]
losses = [PoissonLoss(1.0) for i=1:N]

model = MatFacModel(instance_spmat_vec, feature_spmat_vec, losses; K=k);
# Have an unregularized "offset" factor
model.X[k,:] .= 1.0


fit!(model, aug_X; inst_reg_weight=0.1, feat_reg_weight=0.1, max_iter=1000, loss_iter=1, lr=0.0001, momentum=0.9, K_opt_X=(k-1), K_opt_Y=k, rel_tol=1e-9)

colors = collect(keys(PyPlot.colorsm.TABLEAU_COLORS))

function embedding_scatter(X, labels; F=nothing)
    projected = nothing
    if F == nothing
        F = LinearAlgebra.svd(X)
        projected = F.Vt
    else
        projected = diagm(1.0./F.S) * (transpose(F.U)* X)
    end
    M = length(labels)
    for (i, lab) in enumerate(sort(unique(labels)))
        lab_idx = findall(labels .== lab) 
        scatter3D(projected[1, lab_idx], projected[2,lab_idx], projected[3,lab_idx], color=colors[((i-1) % length(colors)) + 1], label=string(lab))
    end
    legend()

    return F
end


embedding_scatter(model.X, labels)
savefig("brown_embedding_scatter.png")


