using PyPlot
using CSV
using DataFrames
using SparseArrays
using Statistics
import StatsBase: sample
using LinearAlgebra
using PathwayMultiomics
#using Profile
#using ProfileView

showln(x) = (show(x); println())

################################
# LOAD DATA
data = CSV.read("mnist_matrix.tsv", DataFrame; header=false, delim="\t");
data = convert(Matrix, data);
full_X = data[:,1:end-1];
full_labels = data[:,end];
classes = sort(unique(full_labels));


################################
# TRAIN/TEST SPLIT
M = size(data,1)

train_idx = sort(sample(1:M, Int(0.9*M), replace=false))
train_idx_set = Set(train_idx)
test_idx = Int[i for i=1:M if !in(i, train_idx_set)]

train_X = full_X[train_idx,:]
train_labels = full_labels[train_idx]
showln("TRAIN SET: ") 
showln(size(train_X))

test_X = full_X[test_idx,:]
test_labels = full_labels[test_idx]
showln("TEST SET: ")
showln(size(test_X))

full_X = vcat(train_X, test_X)
full_labels = vcat(train_labels, test_labels)
showln("FULL DATA: ")
showln(size(full_X))

################################
# SOME USEFUL FUNCTIONS
unflatten_mnist(vec) = transpose(reshape(vec, (28,28)))


function coord_to_idx(x, y, H)
    return (x-1)*H + y
end

function idx_to_pixel_coord(idx, H)
    return (div((idx-1),H), ((idx-1) % H))
end


################################
# BUILD THE PIXEL GRAPH
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


####################################
# BUILD THE TRAIN INSTANCE GRAPH
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


####################################
# BUILD THE TRAIN INSTANCE MATRIX
train_instance_graph = build_instance_graph(train_labels);
aug_X = vcat( train_X, fill(NaN, length(classes)+1, 28*28) );

showln("AUGMENTED X: ") 
showln(size(aug_X))

showln("BUILT PIXEL AND INSTANCE GRAPHS")

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
train_inst_spmat = graph_to_spmat(train_instance_graph);


####################################
# BUILD THE TEST SET INSTANCE MATRIX
function build_test_graph(train_graph, train_labels, test_labels)

    M_train = length(train_labels)
    M_train_aug = length(train_graph)
    M_test = length(test_labels)
    M_full = M_train_aug + M_test

    classes = sort(unique(train_labels))
    n_classes = length(classes)
    n_bk = n_classes + 1
    @assert n_bk == M_train_aug - M_train 

    label_to_idx = Dict(l => (M_train + i) for (i,l) in enumerate(classes))

    test_graph = copy(train_graph)

    for i=1:M_test
        test_idx = M_train_aug + i
        parent_idx = label_to_idx[test_labels[i]]

        push!(test_graph, [parent_idx])
        push!(test_graph[parent_idx], test_idx)
    end

    return test_graph
end

test_instance_graph = build_test_graph(train_instance_graph, train_labels, test_labels)
test_spmat = graph_to_spmat(test_instance_graph)
println("TEST SPARSE MATRIX:")
showln(test_spmat)

showln("BUILT PIXEL AND INSTANCE MATRICES")


############################################
# CONSTRUCT THE MODEL
k = 10 

feature_spmat_vec = [copy(pixel_spmat) for i=1:k-1]
train_instance_spmat_vec = [copy(train_inst_spmat) for i=1:k]
test_instance_spmat_vec = [copy(test_spmat) for i=1:k]

losses = [LogisticLoss(1.0) for i=1:28*28]
model = MatFacModel(train_instance_spmat_vec, feature_spmat_vec, losses; K=k);
# Have an unregularized "offset" factor
model.X[k,:] .= 1.0

showln("INITIALIZED MODEL")


############################################
# TRAIN MODEL
showln("ABOUT TO FIT")
fit_abstracted!(model, aug_X; inst_reg_weight=0.1, feat_reg_weight=0.1, max_iter=1000, loss_iter=1, lr=0.5, momentum=0.8, K_opt_X=(k-1), K_opt_Y=k, rel_tol=1e-9)

train_labels = convert(Vector{Int64}, train_labels)
test_labels = convert(Vector{Int64}, test_labels)
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
    for lab in sort(unique(labels))
        lab_idx = findall(labels .== lab) 
        scatter3D(projected[1, lab_idx], projected[2,lab_idx], projected[3,lab_idx], color=colors[lab+1], label=string(lab))
    end
    legend()

    return F
end

embedding_F = embedding_scatter(model.X[1:(end-1),:], train_labels)
savefig("embedding_scatter.png", dpi=200)

for i=1:k
    matshow(unflatten_mnist(model.Y[i,:]))
    colorbar()
    savefig(string("embedding_basis_",i,".png"), dpi=200)
end


#####################################
# TRANSFORM THE HELDOUT DATA

new_factor = 0.001*randn(k, length(test_labels))
new_factor[k,:] .= 1.0

showln("NEW FACTOR:")
showln(size(new_factor))

transformed = PathwayMultiomics.transform(model, test_X; inst_reg_weight=0.0, max_iter=1000, loss_iter=1, lr=0.1, momentum=0.8, K_opt_X=(k-1), rel_tol=1e-9, X_new=new_factor) #, new_inst_reg_mats=test_instance_spmat_vec)

println("TRANSFORMED:")
println(size(transformed))

embedding_scatter(transformed[1:(end-1),:], test_labels; F=embedding_F)
savefig("transformed_scatter.png", dpi=200)

