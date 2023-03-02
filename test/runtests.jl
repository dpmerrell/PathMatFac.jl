

using Test, PathwayMultiomics, SparseArrays, LinearAlgebra, Zygote, Flux, StatsBase, MatFac

PM = PathwayMultiomics
MF = MatFac

function util_tests()

    values = ["cat","dog","fish","bird"]

    b_values = ["dog","bird","cat"]

    feature_list = [ ("GENE1","mrnaseq"), ("GENE2","methylation"), 
                    ("GENE3","cna"), ("GENE4", "mutation")] #, ("VIRTUAL1", "activation")]

    @testset "Utilities" begin

        @test PM.is_contiguous([2,2,2,1,1,4,4,4])
        @test PM.is_contiguous(["cat","cat","dog","dog","fish"])
        @test !PM.is_contiguous([1,1,5,5,1,3,3,3])

        @test PM.ids_to_ranges([2,2,2,1,1,4,4,4]) == [1:3, 4:5, 6:8]
        my_ranges = PM.ids_to_ranges(["cat","cat","dog","dog","fish"])
        @test my_ranges == [1:2,3:4,5:5]

        @test PM.subset_ranges(my_ranges, 2:5) == ([2:2,3:4,5:5], 1, 3)

        my_ind_mat = PM.ids_to_ind_mat([1,1,1,2,2,1,2,3,3,1,2,3,3])
        @test my_ind_mat == Bool[1 0 0;
                                 1 0 0;
                                 1 0 0;
                                 0 1 0;
                                 0 1 0;
                                 1 0 0;
                                 0 1 0;
                                 0 0 1;
                                 0 0 1;
                                 1 0 0;
                                 0 1 0;
                                 0 0 1;
                                 0 0 1] 
        # value_to_index
        vti = PM.value_to_idx(values)
        @test (vti["cat"] == 1) & (vti["dog"] == 2) & (vti["fish"] == 3) & (vti["bird"] == 4)
        
        # keymatch
        l_idx, r_idx = PM.keymatch(values, b_values)
        @test l_idx == [1, 2, 4]
        @test r_idx == [3, 1, 2]

        # nansum
        @test PM.nansum([1.0, NaN, 2.0, NaN, 3.0]) == 6.0
        @test PM.nansum([1.0, NaN, 2.0, Inf, 3.0]) == Inf

        # nanmean
        @test PM.nanmean([1.0, NaN, 2.0, NaN, 3.0]) == 2.0
        
        # nanvar
        @test PM.nanvar([1.0, NaN, 2.0, NaN, 3.0]) == 1.0

        # edgelist_to_spmat
        node_to_idx = Dict([string(c)=>idx for (idx,c) in enumerate("abcd")])
        edgelist = [["a","b",1], ["b","c",1], ["c","d",-1], ["d","a",-1]]
        test_spmat = SparseMatrixCSC([ 2.1 -1   0  1; 
                                      -1   2.1 -1  0; 
                                       0   -1  2.1 1;
                                       1    0   1 2.1])
        spmat = PM.edgelist_to_spmat(edgelist, node_to_idx; epsilon=0.1)

        @test isapprox(SparseMatrixCSC(spmat), test_spmat)

        d = PM.edgelist_to_dict(edgelist)
        @test d == Dict("a" => Dict("b" => 1, "d" => -1),
                        "b" => Dict("a" => 1, "c" => 1),
                        "c" => Dict("b" => 1, "d" => -1),
                        "d" => Dict("c" => -1, "a"=> -1))

        rec_edgelist = PM.dict_to_edgelist(d)
        @test Set([Set(edge) for edge in rec_edgelist]) == Set([Set(edge) for edge in edgelist])

        leafy_el = vcat(edgelist, [["d", "e", 1], ["e","g",-1], ["c", "f", 1], ["f", "h", -1]])
        pruned = PM.prune_leaves!(leafy_el, except=["f"])
        correct_pruned = vcat(edgelist, [["c","f",1]])
        @test Set([Set(edge) for edge in pruned]) == Set([Set(edge) for edge in correct_pruned])

    end
end


function batch_array_tests()
    
    @testset "Batch Arrays" begin

        col_batches = ["cat", "cat", "cat", "dog", "dog", "fish"]
        row_batches = Dict("cat"=>[1,1,1,2,2], "dog"=>[1,1,2,2,2], "fish"=>[1,1,1,1,2])
        values = [Dict(1=>3.14, 2=>2.7), Dict(1=>0.0, 2=>0.5), Dict(1=>-1.0, 2=>1.0)]
        A = zeros(5,6)

        ##############################
        # Constructor
        ba = PM.BatchArray(col_batches, row_batches, values)
        @test ba.col_ranges == (1:3, 4:5, 6:6)
        test_row_batches = (Bool[1 0; 1 0; 1 0; 0 1; 0 1],
                            Bool[1 0; 1 0; 0 1; 0 1; 0 1],
                            Bool[1 0; 1 0; 1 0; 1 0; 0 1])
        @test ba.row_batches == test_row_batches 
        @test ba.values == ([3.14, 2.7], [0.0, 0.5], [-1.0, 1.0])

        ##############################
        # View
        ba_view = view(ba, 2:4, 2:5)
        @test ba_view.col_ranges == (1:2, 3:4)
        @test ba_view.row_batches == test_row_batches[1:2]
        @test ba_view.row_idx == ba.row_idx[2:4]
        @test ba_view.values == ([3.14, 2.7],[0.0,0.5])

        ###############################
        # zero
        ba_zero = zero(ba)
        @test ba_zero.col_ranges == ba.col_ranges
        @test ba_zero.row_batches == ba.row_batches
        @test ba_zero.values == ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

        ###############################
        # Addition
        Z = A + ba
        test_mat = [3.14 3.14 3.14 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.5 0.5  1.0]
        @test Z == test_mat
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x+y), A, ba)
        @test A_grad == ones(size(A)...)
        @test ba_grad.values == ([9., 6.], [4., 6.], [4., 1.])

        ################################
        # Multiplication
        Z = ones(5,6) * ba
        @test Z == test_mat
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x*y), ones(5,6), ba)
        @test A_grad == Z
        @test ba_grad.values == ([9.0, 6.0],[4.0, 6.0],[4.0, 1.0])

        ################################
        # Exponentiation
        ba_exp = exp(ba)
        @test (ones(5,6) * ba_exp) == exp.(test_mat)
        (ba_grad,) = Zygote.gradient(x->sum(ones(5,6) * exp(x)), ba)
        @test ba_grad.values == ([9. * exp(3.14), 6. * exp(2.7)],
                                 [4. * exp(0.0), 6. * exp(0.5)],
                                 [4. * exp(-1.0), 1. * exp(1.0)])

        #################################
        # GPU
        ba_d = gpu(ba)
        @test ba.values == ([3.14, 2.7], [0.0, 0.5], [-1.0, 1.0])
       
        # view
        ba_d_view = view(ba_d, 2:4, 2:5)
        @test ba_d_view.col_ranges == (1:2, 3:4)
        @test ba_d_view.row_batches == ba_d.row_batches[1:2]
        @test ba_d_view.row_idx == ba_d.row_idx[2:4]
        @test ba_d_view.values == (gpu([3.14, 2.7]),gpu([0.0,0.5]))

        # zero 
        ba_d_zero = zero(ba_d)
        @test ba_d_zero.col_ranges == ba_d.col_ranges
        @test ba_d_zero.row_batches == ba_d.row_batches
        @test ba_d_zero.values == map(gpu, ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]))

        # addition
        A_d = gpu(A)
        Z_d = A_d + ba_d
        test_mat_d = gpu(test_mat)
        @test Z_d == test_mat_d
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x+y), A_d, ba_d)
        @test A_grad == gpu(ones(size(A)...))
        @test ba_grad.values == map(gpu, ([9., 6.], [4., 6.], [4., 1.]))

        # multiplication
        Z_d = gpu(ones(5,6)) * ba_d
        @test Z_d == test_mat_d
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x*y), gpu(ones(5,6)), ba_d)
        @test A_grad == Z_d
        @test ba_grad.values == map(gpu, ([9.0, 6.0],[4.0, 6.0],[4.0, 1.0]))

        # exponentiation
        ba_d_exp = exp(ba_d)
        @test isapprox((gpu(ones(5,6)) * ba_d_exp), exp.(test_mat_d))
        (ba_grad,) = Zygote.gradient(x->sum(gpu(ones(5,6)) * exp(x)), ba_d)
        @test all(map(isapprox, ba_grad.values, map(gpu, ([9. * exp(3.14), 6. * exp(2.7)],
                                                          [4. * exp(0.0), 6. * exp(0.5)],
                                                          [4. * exp(-1.0), 1. * exp(1.0)]))))


    end
end

function layers_tests()

    @testset "Layers" begin

        M = 20
        N = 30
        K = 4

        view_N = 10

        n_col_batches = 2
        n_row_batches = 4

        X = randn(K,M)
        Y = randn(K,N)
        xy = transpose(X)*Y

        X_view = view(X,:,1:M)
        Y_view = view(Y,:,1:view_N)
        view_xy = transpose(X_view)*Y_view
       
        feature_ids = map(x->string("x_",x), 1:N)
 
        ###################################
        # Column Scale
        cscale = PM.ColScale(N)
        @test size(cscale.logsigma) == (N,)
        @test cscale(xy) == xy .* transpose(exp.(cscale.logsigma))
        
        cscale_view = view(cscale, 1:M, 1:view_N)
        @test cscale_view(view_xy) == view_xy .* transpose(exp.(cscale.logsigma[1:view_N]))

        ###################################
        # Column Shift
        cshift = PM.ColShift(N)
        @test size(cshift.mu) == (N,)
        @test cshift(xy) == xy .+ transpose(cshift.mu)
        
        cshift_view = view(cshift, 1:M, 1:view_N)
        @test cshift_view(view_xy) == view_xy .+ transpose(cshift.mu[1:view_N])

        ##################################
        # Batch Scale
        col_batches = repeat([string("colbatch",i) for i=1:n_col_batches], inner=div(N,n_col_batches))
        row_batches = [repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches]
        batch_dict = Dict(zip(unique(col_batches), row_batches))

        bscale = PM.BatchScale(col_batches, batch_dict)
        @test length(bscale.logdelta.values) == n_col_batches
        @test size(bscale.logdelta.values[1]) == (n_row_batches,)
        @test bscale(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .* exp(bscale.logdelta.values[1][1])

        bscale_grads = Zygote.gradient((f,x)->sum(f(x)), bscale, xy)
        @test isapprox(bscale_grads[1].logdelta.values[end][end], sum(xy[16:20,16:30].*exp(bscale.logdelta.values[end][end])))
        @test bscale_grads[2] == zeros(M,N) + exp(bscale.logdelta)

        ##################################
        # Batch Shift
        bshift = PM.BatchShift(col_batches, batch_dict)
        @test length(bshift.theta.values) == n_col_batches
        @test size(bshift.theta.values[1]) == (n_row_batches,)
        @test bshift(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .+ bshift.theta.values[1][1]

        bshift_grads = Zygote.gradient((f,x)->sum(f(x)), bshift, xy)
        @test bshift_grads[1].theta.values == Tuple(ones(n_row_batches).*(div(M,n_row_batches)*div(N,n_col_batches)) for j=1:n_col_batches)
        @test bshift_grads[2] == ones(M,N)

        ###################################
        # Layer composition
        transform_nobatch = PM.construct_model_layers(col_batches, nothing)
        @test length(transform_nobatch.layers) == 4 
        @test typeof(transform_nobatch.layers[1]) <: PM.ColScale 
        @test typeof(transform_nobatch.layers[2]) <: PM.ColShift
        @test typeof(transform_nobatch.layers[3]) <: Function
        @test typeof(transform_nobatch.layers[4]) <: Function
        transform_grads = Zygote.gradient((f,x)->sum(f(x)), transform_nobatch, xy)
        @test transform_grads[1].layers[1].logsigma != nothing
        @test transform_grads[1].layers[2].mu != nothing
        @test transform_grads[1].layers[3] == nothing
        @test transform_grads[1].layers[4] == nothing
 
        transform_w_batch = PM.construct_model_layers(col_batches, batch_dict) 
        @test length(transform_w_batch.layers) == 4 
        @test typeof(transform_w_batch.layers[1]) <: PM.ColScale 
        @test typeof(transform_w_batch.layers[2]) <: PM.ColShift
        @test typeof(transform_w_batch.layers[3]) <: PM.BatchScale 
        @test typeof(transform_w_batch.layers[4]) <: PM.BatchShift
        transform_grads = Zygote.gradient((f,x)->sum(f(x)), transform_w_batch, xy)
        @test transform_grads[1].layers[1].logsigma != nothing
        @test transform_grads[1].layers[2].mu != nothing
        @test transform_grads[1].layers[3].logdelta != nothing
        @test transform_grads[1].layers[4].theta != nothing
        
        ###################################
        # FrozenLayer
        PM.freeze_layer!(transform_w_batch, 1) 
        @test isa(transform_w_batch.layers[1], PM.FrozenLayer)
        transform_grads = Zygote.gradient((f,X)->sum(f(X)), transform_w_batch, xy)
        @test transform_grads[1].layers[1] == nothing

        PM.unfreeze_layer!(transform_w_batch, 1)
        @test isa(transform_w_batch.layers[1], PM.ColScale)
        transform_grads = Zygote.gradient((f,X)->sum(f(X)), transform_w_batch, xy)
        @test isa(transform_grads[1].layers[1].logsigma, AbstractArray) 
    end
end


function preprocess_tests()

    test_sif_path = "test_pathway.sif"

    # Contents of a SIF file
    test_sif_contents = [["MOL:GTP",   "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["METAPHASE", "a>", "PLK1"],
                         ["PAK1",      "a>", "PLK1"],
                         ["RAB1A",     "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PLK1",      "a>", "SGOL1"],
                         ["PLK1",      "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PP2A-ALPHA B56", "a|", "SGOL1"]
                        ]

    # After loading a SIF file, we have an edgelist
    # whose entries are tagged by "_activation"
    test_pwy_edges = [["MOL:GTP", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["METAPHASE", "PLK1", 1],
                      ["PAK1", "PLK1", 1],
                      ["RAB1A", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["PLK1", "SGOL1", 1],
                      ["PLK1", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["PP2A-ALPHA B56", "SGOL1", -1]
                     ]
    test_pwy_genes = PM.get_all_entities(test_pwy_edges)
    tag_edge = edge -> [string(edge[1],"_activation"),
                        string(edge[2],"_activation"), edge[3]]
    test_pwy_edges = map(tag_edge, test_pwy_edges)
    test_pwy_nodes = PM.get_all_nodes(test_pwy_edges)
 
    # Our data may contain features that are not
    # involved in the pathway.
    # Establish the gene, "dogma level", and 
    # weight for each feature.
    feature_genes = ["PLK1","PLK1","PLK1", 
                     "PAK1", "PAK1", "PAK1",
                     "SGOL1", 
                     "BRCA", "BRCA"]
    feature_assays = ["cna", "mutation","mrnaseq", 
                      "rppa", "methylation", "mutation",
                      "mrnaseq",
                      "mrnaseq", "methylation"]
    ASSAY_TO_DOGMA = Dict("cna" => "dna",
                          "mutation" => "dna",
                          "mrnaseq" => "mrna",
                          "methylation" => "mrna",
                          "rppa" => "protein")
    ASSAY_TO_WEIGHT = Dict("cna" => 1.0,
                           "mutation" => -1.0,
                           "mrnaseq" => 1.0,
                           "methylation" => -1.0,
                           "rppa" => 1.0)
    feature_dogmas = map(x->ASSAY_TO_DOGMA[x], feature_assays)
    feature_weights = map(x->ASSAY_TO_WEIGHT[x], feature_assays)
    feature_ids = PM.construct_pwy_feature_ids(feature_genes, feature_dogmas, collect(1:9))

    # Find the data genes that overlap with the pathway.
    # And collect the corresponding features.
    relevant_genes = intersect(test_pwy_genes, Set(feature_genes))
    relevant_feature_idx = map(x -> in(x, relevant_genes), feature_genes)
    relevant_features = feature_ids[relevant_feature_idx]
    relevant_weights = feature_weights[relevant_feature_idx]

    # The "central dogma-related" edges should look like this
    test_dogma_edges = [["PLK1_dna", "PLK1_mrna", 1.],
                        ["PLK1_mrna", "PLK1_protein", 1.],
                        ["PLK1_protein", "PLK1_activation", 1.],
                        ["PAK1_dna", "PAK1_mrna", 1.],
                        ["PAK1_mrna", "PAK1_protein", 1.],
                        ["PAK1_protein", "PAK1_activation", 1.],
                        ["SGOL1_dna", "SGOL1_mrna", 1.],
                        ["SGOL1_mrna", "SGOL1_protein", 1.],
                        ["SGOL1_protein", "SGOL1_activation", 1.],
                       ]

    # The edges connecting data to the central dogma should look like this
    test_data_edges = [["PLK1_dna", "PLK1_dna_1", 1.],
                       ["PLK1_dna", "PLK1_dna_2", -1.],
                       ["PLK1_mrna", "PLK1_mrna_3",  1.],
                       ["PAK1_protein", "PAK1_protein_4",  1.],
                       ["PAK1_mrna","PAK1_mrna_5",  -1.],
                       ["PAK1_dna", "PAK1_dna_6",  -1.],
                       ["SGOL1_mrna", "SGOL1_mrna_7",  1.]
                      ]
  
    # The final set of edges should look like this. 
    test_all_edges = vcat(test_data_edges, test_dogma_edges, test_pwy_edges)
    PM.prune_leaves!(test_all_edges; except=relevant_features)
    test_all_edge_set = Set(map(Set, test_all_edges))

    @testset "Prep Pathways" begin

        # read_sif_file
        sif_data = PM.read_sif_file(test_sif_path)
        @test sif_data == test_sif_contents

        el = PM.sif_to_edgelist(sif_data)
        @test Set(el) == Set(test_pwy_edges) 

        # construct_dogma_edges
        dogma_edges = PM.construct_dogma_edges(relevant_genes)
        @test Set(dogma_edges) == Set(test_dogma_edges)

        # construct_data_edges
        data_edges = PM.construct_data_edges(relevant_features, relevant_weights)
        @test Set(map(Set, data_edges)) == Set(map(Set, test_data_edges))
        
        # prep pathways
        prepped_pwys, new_ids = prep_pathway_graphs([sif_data], feature_genes, feature_dogmas;
                                                   feature_weights=feature_weights)
        prepped_pwy = prepped_pwys[1]
        all_edge_set = Set(map(Set, prepped_pwy))
        @test all_edge_set == test_all_edge_set 

        # load_pathway_sifs
        prepped_pwys, new_ids = prep_pathway_graphs([test_sif_path], feature_genes, feature_dogmas;
                                          feature_weights=feature_weights)
        prepped_pwy = prepped_pwys[1]
        all_edge_set = Set(map(Set, prepped_pwy))
        @test all_edge_set == test_all_edge_set 

    end
end


function reg_tests()

    test_sif_path = "test_pathway.sif"
        
    feature_genes = ["PLK1","PLK1","PLK1", 
                     "PAK1", "PAK1", "PAK1",
                     "SGOL1", 
                     "BRCA", "BRCA"]
    feature_assays = ["cna", "mutation","mrnaseq", 
                      "rppa", "methylation", "mutation",
                      "mrnaseq",
                      "mrnaseq", "methylation"]
    feature_dogmas = ["dna", "dna", "mrna",
                      "protein", "mrna", "dna",
                      "mrna",
                      "mrna", "mrna"]

    @testset "Network regularizers" begin

        #############################################
        # Test on synthetic network
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3]
        nr = PM.NetworkRegularizer(data_features, edgelists)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (3,3)
        @test nr.AA[1] == sparse([1.1 -1. 0.;# 0;
                                  -1. 2.1 -1.;# 0;
                                  0. -1. 2.1])# 1;
                                  #0 0 1 2])
        @test size(nr.AB[1]) == (3,1)
        @test nr.AB[1] == sparse(reshape([0.;
                                          0.;
                                          -1.], (3,1)))
        @test size(nr.BB[1]) == (1,1)
        @test nr.BB[1] == sparse(ones(1,1)*1.1)
        #@test length(nr.BB_chol) == 2
        #@test size(nr.BB_chol[1]) == (1,1)

        model_features = [1,2,3,4]
        nr = PM.NetworkRegularizer(model_features, edgelists)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (4,4)
        @test size(nr.AB[1]) == (4,0)
        @test size(nr.BB[1]) == (0,0)
        #@test length(nr.BB_chol) == 2
        #@test size(nr.BB_chol[1]) == (0,0)

        ##############################################
        # Test on "real pathway"
        prepped_pwys, model_features = PM.prep_pathway_graphs([test_sif_path], 
                                                               feature_genes,
                                                               feature_dogmas)
        pwy_nodes = Set()
        for el in prepped_pwys
            for edge in el
                push!(pwy_nodes, edge[1])
                push!(pwy_nodes, edge[2])
            end
        end
        netreg = PM.NetworkRegularizer(model_features, prepped_pwys)
      
        n_obs = length(model_features) 
        n_unobs = length(setdiff(pwy_nodes, Set(model_features)))
        @test length(netreg.AA) == 1
        @test size(netreg.AA[1]) == (n_obs,n_obs)
        @test size(netreg.AB[1]) == (n_obs,n_unobs)
        @test size(netreg.BB[1]) == (n_unobs, n_unobs)

        #@test length(netreg.BB_chol) == 1 
        #@test size(netreg.BB_chol[1]) == (n_unobs, n_unobs)

        ################################################
        # Gradient test
        edgelists = [[[1, 2, 1.0],[1, 3, 1.0],[1, 4, 1.0]],
                    ]
        data_features = [2,3,4]
        nr = PM.NetworkRegularizer(data_features, edgelists)

        @test length(nr.AA) == 1
        @test nr.AA[1] == sparse([1.1 0.0 0.0;
                                  0.0 1.1 0.0;
                                  0.0 0.0 1.1])
        @test nr.AB[1] == sparse(reshape([-1.0;
                                          -1.0;
                                          -1.0;], (3,1)))
        @test nr.BB[1] == sparse(reshape([3.1], (1,1) ))

        mu_regularizer = (x, reg) -> reg(x)

        loss, grads = Zygote.withgradient(mu_regularizer, transpose([1.0, 1.0, 1.0]), nr)
        @test size(grads[1]) == (1,3) 
        
    end
    
    @testset "Selective L1 regularizers" begin

        K = 2
        N = 5
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3,4,5]
        test_Y = randn(2,5)
        reg = PM.L1Regularizer(data_features, edgelists)
  
        @test isapprox(reg.l1_idx, [0 0 0 0 1;
                                    0 0 0 0 1])
        @test isapprox(reg(test_Y), sum(abs.(reg.l1_idx .* test_Y)))
        
        loss, grads = Flux.withgradient((r,x) -> r(x), reg, ones(2,5)) 
        @test grads[1] == nothing
        test_grad = zeros(2,5)
        test_grad[:,5] .= 1
        @test isapprox(grads[2], test_grad)                                                     
    end

    @testset "Group regularizers" begin
        
        K = 2
        N = 5
        data_groups = [1,1,1,2,2]
        test_Y = randn(3,5)
        reg = PM.GroupRegularizer(data_groups; K=size(test_Y, 1))
        means = [mean(test_Y[1,1:3]) mean(test_Y[1,4:5]);
                 mean(test_Y[2,1:3]) mean(test_Y[2,4:5]);
                 mean(test_Y[3,1:3]) mean(test_Y[3,4:5])] 
        @test isapprox(reg(test_Y), 0.5*sum([sum((test_Y[1,1:3] .- means[1,1]).^2) sum((test_Y[1,4:5] .- means[1,2]).^2);
                                             sum((test_Y[2,1:3] .- means[2,1]).^2) sum((test_Y[2,4:5] .- means[2,2]).^2);
                                             sum((test_Y[3,1:3] .- means[3,1]).^2) sum((test_Y[3,4:5] .- means[3,2]).^2)]))

    end

    @testset "ARD regularizers" begin

        K = 3
        N = 5
        test_Y = randn(K,N)
        reg = PM.ARDRegularizer(test_Y)
        l, grads = Zygote.withgradient(reg, test_Y)
        test_precisions = (reg.alpha + 1) ./ (reg.beta .+ (test_Y.*test_Y))
        @test isapprox(l, 0.5*sum(test_precisions .* test_Y .* test_Y))
        @test length(grads) == 1
        @test isapprox(grads[1], test_precisions .* test_Y)
    end


    @testset "BatchArray regularizers" begin
        col_batches = ["cat", "cat", "cat", "dog", "dog", "fish"]
        row_batches = Dict("cat"=>[1,1,1,2,2], "dog"=>[1,1,2,2,2], "fish"=>[1,1,1,1,2])
        values = [Dict(1=>3.14, 2=>2.7), Dict(1=>0.0, 2=>0.5), Dict(1=>-1.0, 2=>1.0)]
        
        ba = PM.BatchArray(col_batches, row_batches, values)

        ba_reg = PM.BatchArrayReg(ba; weight=1.0)

        @test sum(map(sum, ba_reg.counts)) == 6*5
        grads = Zygote.gradient((r,ba)->r(ba), ba_reg, ba)

    end

    @testset "Composed regularizers" begin
        K = 2
        N = 5
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3,4,5]
        test_Y = randn(2,5)

        l1_reg = PM.L1Regularizer(data_features, edgelists)
        net_reg = PM.NetworkRegularizer(data_features, edgelists)
        composite_reg = PM.construct_composite_reg([l1_reg, net_reg])

        @test length(composite_reg.regularizers) == 2
        @test isapprox(composite_reg(test_Y), l1_reg(test_Y) + net_reg(test_Y))
    end
end

function featureset_ard_tests()

    N = 40
    K = 10
    Y = randn(K,N)
    feature_ids = collect(1:N)
    feature_views = repeat([1,2], inner=div(N,2))
    feature_sets = [collect(1:10),collect(11:20),collect(21:30),collect(31:40)]

    reg = PM.construct_featureset_ard(K, feature_ids, feature_views, feature_sets;
                                      beta0=1e-6, lr=0.1, tau_max=1e6)

    @testset "Featureset ARD constructor" begin

        @test size(reg.tau) == (K,N)
        @test length(reg.feature_views) == 2
        @test reg.feature_view_ids == [1,2]
        @test all(reg.alpha .== 1e-6)
        @test all(reg.scale .== 1.0)
        @test isapprox(reg.beta0, 1e-6)
        @test isapprox(reg.tau_max, 1e6)
        
        test_S = zeros(Bool, length(feature_sets), N)
        for (i, s) in enumerate(feature_sets)
            test_S[i,s] .= true
        end

        @test issparse(reg.S)
        @test isapprox(Matrix(reg.S), test_S)
    end
   

    @testset "Featureset ARD updates" begin

        # Update tau
        PM.update_tau!(reg, Y)
        # This should hold ONLY because alpha and scale have not been updated yet.
        @test isapprox(reg.tau, (transpose(reg.alpha) .+ 0.5) ./ (reg.beta0 .+ 0.5.*(Y.*Y)))

        # Update alpha and scale
        view_1_idx = feature_views .== 1
        view_2_idx = feature_views .== 2
        test_alpha = zeros(Float32, N)
        test_alpha[view_1_idx] .= mean(reg.tau[:,view_1_idx])^2 / (mean(reg.tau[:,view_1_idx].^2) .- mean(reg.tau[:,view_1_idx])^2)
        test_alpha[view_2_idx] .= mean(reg.tau[:,view_2_idx])^2 / (mean(reg.tau[:,view_2_idx].^2) .- mean(reg.tau[:,view_2_idx])^2)

        test_scale = zeros(Float32, N)
        test_scale[view_1_idx] .= test_alpha[view_1_idx] ./ (reg.beta0 .* quantile(vec(reg.tau[:,view_1_idx]), 0.9)) 
        test_scale[view_2_idx] .= test_alpha[view_2_idx] ./ (reg.beta0 .* quantile(vec(reg.tau[:,view_2_idx]), 0.9)) 
        
        PM.update_alpha_scale!(reg; q=0.9)
        @test isapprox(reg.alpha, test_alpha)
        @test isapprox(reg.scale, test_scale)

    end

    @testset "Featureset ARD regularization" begin

        @test isapprox(reg(Y), 0.5*sum(reg.tau .* Y .*Y))
        grads = Zygote.gradient(reg, Y)
        @test isapprox(grads[1], reg.tau .* Y)

    end 
end


function model_tests()

    M = 20
    N = 30
    K = 4

    n_col_batches = 2
    n_row_batches = 4

    X = randn(K,M)
    Y = randn(K,N)
    Z = transpose(X)*Y

    sample_ids = [string("sample_", i) for i=1:M]
    sample_conditions = repeat(["condition_1", "condition_2"], inner=div(M,2))

    sample_graph = [[s, "z", 1] for s in sample_ids]
    sample_graphs = fill(sample_graph, K)

    feature_ids = map(x->string("x_",x), 1:N)
    feature_views = repeat(1:n_col_batches, inner=div(N,n_col_batches))
    batch_dict = Dict(j => repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches)

    feature_graph = [[feat, "y", 1] for feat in feature_ids]
    feature_graphs = fill(feature_graph, K)

    @testset "Default constructor" begin

        model = PathMatFacModel(Z)
        @test size(model.matfac.X) == (10,M)
        @test size(model.matfac.Y) == (10,N)
        @test typeof(model.matfac.col_transform.layers[1]) == PM.ColScale
        @test typeof(model.matfac.col_transform.layers[2]) == PM.ColShift
        @test model.sample_ids == collect(1:M)
        @test model.feature_ids == collect(1:N)
        @test typeof(model.matfac.noise_model.noises[1]) == MF.NormalNoise
    end

    @testset "Batch effect model constructor" begin

        model = PathMatFacModel(Z; K=K, feature_views=feature_views, batch_dict=batch_dict)
        @test size(model.matfac.X) == (K,M)
        @test size(model.matfac.Y) == (K,N)
        @test typeof(model.matfac.col_transform.layers[1]) == PM.ColScale
        @test typeof(model.matfac.col_transform.layers[2]) == PM.ColShift
        @test typeof(model.matfac.col_transform.layers[3]) == PM.BatchScale
        @test typeof(model.matfac.col_transform.layers[4]) == PM.BatchShift
    end

    @testset "Y-regularized model constructor" begin
       
        # Graph regularized model construction 
        model = PathMatFacModel(Z; feature_ids=feature_ids, feature_graphs=feature_graphs, lambda_Y_graph=1.0)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[3]) <: PM.NetworkRegularizer
        
        # Vanilla L1-regularized model construction 
        model = PathMatFacModel(Z; K=7, lambda_Y_l1=3.14)
        @test size(model.matfac.Y) == (7,N)
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[1]) <: Function 
        @test isapprox(model.matfac.Y_reg(model.matfac.Y), 3.14*sum(abs.(model.matfac.Y)))

        # Selective L1-regularized model construction 
        model = PathMatFacModel(Z; feature_ids=feature_ids, feature_graphs=feature_graphs, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.L1Regularizer
        
        # Both, at the same time! 
        model = PathMatFacModel(Z; feature_ids=feature_ids, feature_graphs=feature_graphs, lambda_Y_graph=1.0, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.L1Regularizer
        @test typeof(model.matfac.Y_reg.regularizers[3]) <: PM.NetworkRegularizer

    end

    @testset "X-regularized model constructor" begin
        
        # Vanilla L2-regularized model construction 
        model = PathMatFacModel(Z; K=8, lambda_X_l2=3.14)
        @test size(model.matfac.X) == (8,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[1]) <: Function 
        @test isapprox(model.matfac.X_reg(model.matfac.X), 0.5*3.14*sum(model.matfac.X .* model.matfac.X))
       
        # Group-based X regularization 
        model = PathMatFacModel(Z; K=6, sample_conditions=sample_conditions, lambda_X_condition=3.14)
        @test size(model.matfac.X) == (6,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[2]) <: PM.GroupRegularizer
        
        # Graph-based X regularization 
        model = PathMatFacModel(Z; sample_ids=sample_ids, sample_graphs=sample_graphs)
        @test size(model.matfac.X) == (K,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test model.matfac.X_reg.regularizers[3].weight == 1.0
        
        # Combined X regularization 
        model = PathMatFacModel(Z; sample_ids=sample_ids, sample_conditions=sample_conditions,
                                   sample_graphs=sample_graphs, lambda_X_graph=1.234, lambda_X_condition=5.678)
        @test size(model.matfac.X) == (K,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[2]) <: PM.GroupRegularizer
        @test typeof(model.matfac.X_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test model.matfac.X_reg.regularizers[2].weight == 5.678 
        @test model.matfac.X_reg.regularizers[3].weight == 1.234
    end

    @testset "Full-featured model constructor" begin
        model = PathMatFacModel(Z; sample_ids=sample_ids, sample_conditions, sample_graphs=sample_graphs, 
                                   lambda_X_graph=1.234, lambda_X_condition=5.678, 
                                   feature_ids=feature_ids, feature_views=feature_views, 
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_Y_graph=1.0, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.X) == (K,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[2]) <: PM.GroupRegularizer
        @test typeof(model.matfac.X_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test model.matfac.X_reg.regularizers[2].weight == 5.678 
        @test model.matfac.X_reg.regularizers[3].weight == 1.234
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.L1Regularizer
        @test typeof(model.matfac.Y_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test model.matfac.Y_reg.regularizers[2].weight == 1.0 
        @test model.matfac.Y_reg.regularizers[3].weight == 1.0
    end

end


function score_tests()

    @testset "Average Precision tests" begin

        y_pred = [0, 0, 0.5, 0.5, 1.0]
        y_true = [false, false, false, true, true]

        @test isapprox(PM.average_precision(y_pred, y_true), 5/6)
        
        y_pred = [0, 0.5, 0, 1.0, 0.5]
        y_true = [false, false, false, true, true]

        @test isapprox(PM.average_precision(y_pred, y_true), 5/6)
    end

end



function fit_tests()
    
    M = 20
    N = 30
    K = 4

    n_col_batches = 2
    n_row_batches = 4

    X = randn(K,M)
    Y = randn(K,N)
    Z = transpose(X)*Y

    sample_ids = [string("sample_", i) for i=1:M]
    sample_conditions = repeat(["condition_1", "condition_2"], inner=div(M,2))

    sample_graph = [[s, "z", 1] for s in sample_ids]
    sample_graphs = fill(sample_graph, K)

    feature_ids = map(x->string("x_",x), 1:N)
    feature_views = repeat(1:n_col_batches, inner=div(N,n_col_batches))
    batch_dict = Dict(j => repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches)

    feature_graph = [[feat, "y", 1] for feat in feature_ids]
    feature_graphs = fill(feature_graph, K)

    @testset "Fit CPU" begin

        model = PathMatFacModel(Z; sample_conditions, feature_ids=feature_ids,  feature_views=feature_views,
                                                      feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                                      lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[3]) 
        fit!(model; verbosity=2, lr=0.25, max_epochs=1000, print_iter=1, rel_tol=1e-7, abs_tol=1e-7)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[3].logdelta.values)
                 ) # This should not have changed
    end

    @testset "Fit GPU" begin

        model = PathMatFacModel(Z; sample_conditions, feature_ids=feature_ids,  feature_views=feature_views,
                                                      feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                                      lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[3])

        model = gpu(model) 
        fit!(model; verbosity=2, lr=0.25, max_epochs=1000, print_iter=1, rel_tol=1e-7, abs_tol=1e-7)
        model = cpu(model)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[3].logdelta.values)
                 ) # This should not have changed
    end
end


function transform_tests()
    
    ###################################
    # Build the training set
    M = 20
    N = 40
    K = 4

    n_col_batches = 4
    n_row_batches = 4

    X = randn(K,M)
    Y = randn(K,N)
    Z = transpose(X)*Y

    sample_ids = [string("sample_", i) for i=1:M]
    sample_conditions = repeat(["condition_1", "condition_2"], inner=div(M,2))

    sample_graph = [[s, "z", 1] for s in sample_ids]
    sample_graphs = fill(sample_graph, K)

    feature_ids = map(x->string("x_",x), 1:N)
    feature_views = repeat(1:n_col_batches, inner=div(N,n_col_batches))
    batch_dict = Dict(j => repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j in unique(feature_views))

    feature_graph = [[feat, "y", 1] for feat in feature_ids]
    feature_graphs = fill(feature_graph, K)

    ########################################
    # Fit the model on the training set
    model = PathMatFacModel(Z; sample_conditions, feature_ids=feature_ids,  feature_views=feature_views,
                                                  feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                                  lambda_X_condition=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

    fit!(model; verbosity=-1, lr=0.25, max_epochs=1000, print_iter=1, rel_tol=1e-7, abs_tol=1e-7)

    ###################################
    # Build the test set
    M_new = 20
    N_new = 40
    new_sample_conditions = repeat(["condition_2","condition_3"], inner=div(M_new,2)) # partially-overlapping sample conditions
    new_feature_ids = map(x->string("x_",x+10), 1:N_new) # Partially-overlapping features
    new_feature_views = repeat((1:n_col_batches) .+ 1, inner=div(N_new,n_col_batches)) # partially-overlapping feature views
    new_batch_dict = Dict(j => repeat([string("rowbatch",i) for i=((1:n_row_batches) .+ 2)], inner=div(M,n_row_batches)) for j in unique(new_feature_views))
    D_new = randn(M_new, N_new) 

    @testset "Transform CPU" begin

        result = transform(model, D_new; feature_ids=new_feature_ids, feature_views=new_feature_views,
                                         sample_conditions=new_sample_conditions,
                                         verbosity=2, lr=0.25, max_epochs=1000, print_iter=1, rel_tol=1e-7, abs_tol=1e-7,
                                         use_gpu=false)
        
        @test size(result.matfac.X) == (K, M_new) 
        @test size(result.matfac.Y) == (K, 30) 
        @test all(result.sample_ids .== collect(1:M_new)) 
        @test all(result.sample_conditions .== new_sample_conditions) 
              
    end

    @testset "Transform GPU" begin
        
        result = transform(model, D_new; feature_ids=new_feature_ids, feature_views=new_feature_views,
                                         sample_conditions=new_sample_conditions,
                                         verbosity=2, lr=0.25, max_epochs=1000, print_iter=1, rel_tol=1e-7, abs_tol=1e-7,
                                         use_gpu=true)
        
        @test size(result.matfac.X) == (K, M_new) 
        @test size(result.matfac.Y) == (K, 30) 
        @test all(result.sample_ids .== collect(1:M_new)) 
        @test all(result.sample_conditions .== new_sample_conditions) 

    end
end


function model_io_tests()

    test_bson_path = "test.bson"

    M = 20
    N = 30
    K = 4

    n_col_batches = 2
    n_row_batches = 4

    X = randn(K,M)
    Y = randn(K,N)
    Z = transpose(X)*Y

    sample_ids = [string("sample_", i) for i=1:M]
    sample_conditions = repeat(["condition_1", "condition_2"], inner=div(M,2))

    sample_graph = [[s, "z", 1] for s in sample_ids]
    sample_graphs = fill(sample_graph, K)

    feature_ids = map(x->string("x_",x), 1:N)
    feature_views = repeat(1:n_col_batches, inner=div(N,n_col_batches))
    batch_dict = Dict(j => repeat([string("rowbatch",i) for i=1:n_row_batches], inner=div(M,n_row_batches)) for j=1:n_col_batches)

    feature_graph = [[feat, "y", 1] for feat in feature_ids]
    feature_graphs = fill(feature_graph, K)

    @testset "Model IO" begin

        model = PathMatFacModel(Z; sample_conditions, feature_ids=feature_ids,  feature_views=feature_views,
                                                      lambda_X_l2=0.1,# lambda_Y_l1=0.05, 
                                                      feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                                      lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        PM.save_model(model, test_bson_path)

        recovered_model = PM.load_model(test_bson_path)

        @test recovered_model == model

        rm(test_bson_path)
    end

end


function simulation_tests()
    
    test_sif_path = "test_pathway.sif"
    test_pwy_name = "test_pathway" 
    feature_genes = ["PLK1","PLK1","PLK1", 
                     "PAK1", "PAK1", "PAK1",
                     "SGOL1", 
                     "BRCA", "BRCA"]
    feature_assays = ["cna", "mutation","mrnaseq", 
                      "rppa", "mrnaseq", "mutation",
                      "mrnaseq",
                      "mrnaseq", "methylation"]
    M = 10
    m_groups = 2
    N = 9   

    sample_ids = [string("patient_",i) for i=1:M]
    sample_conditions = repeat([string("group_",i) for i=1:m_groups], inner=5)
        
    sample_batch_dict = Dict([k => copy(sample_conditions) for k in unique(feature_assays)])

    @testset "Data Simulation" begin

        n_pwys = 3
        pathway_sif_data = repeat([test_sif_path], n_pwys)
        pathway_names = [string("test_pwy_",i) for i=1:n_pwys]

        model, D = PM.simulate_data(pathway_sif_data, 
                                    pathway_names,
                                    sample_ids, 
                                    sample_conditions,
                                    feature_genes, 
                                    feature_assays,
                                    sample_batch_dict)
        @test size(D) == (M,N)
        @test all(isinteger.(D[:,[1,2,6]]))
    end
end


function main()

    #util_tests()
    #batch_array_tests()
    #layers_tests()
    #preprocess_tests()
    #reg_tests()
    featureset_ard_tests()
    #model_tests()
    #score_tests()
    #fit_tests()
    #transform_tests()
    #model_io_tests()
    #simulation_tests()

end

main()


