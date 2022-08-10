

using Test, PathwayMultiomics, SparseArrays, LinearAlgebra, ScikitLearnBase, Zygote, Flux

PM = PathwayMultiomics

function util_tests()

    values = ["cat","dog","fish","bird"]

    b_values = ["dog","bird","cat"]

    feature_list = [ ("GENE1","mrnaseq"), ("GENE2","methylation"), 
                    ("GENE3","cna"), ("GENE4", "mutation")] #, ("VIRTUAL1", "activation")]

    @testset "Utility functions" begin

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

        # get_assay
        @test PM.get_assay(("BRCA","mrnaseq")) == "mrnaseq"

        # get_loss
        @test PM.get_loss(("BRCA","mrnaseq")) == "normal"
        @test PM.get_loss(("BRCA","mutation")) == "bernoulli"

        # sort_features
        @test PM.sort_features(feature_list) == [("GENE1", "mrnaseq"),("GENE2","methylation"), 
                                                 ("GENE4", "mutation"), ("GENE3","cna"), 
                                                 #("VIRTUAL1", "activation")
                                                 ]

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
        row_batches = [[1,1,1,2,2], [1,1,2,2,2], [1,1,1,1,2]]
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
        bscale = PM.BatchScale(col_batches, row_batches)
        @test length(bscale.logdelta.values) == n_col_batches
        @test size(bscale.logdelta.values[1]) == (n_row_batches,)
        @test bscale(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .* exp(bscale.logdelta.values[1][1])

        bscale_grads = Zygote.gradient((f,x)->sum(f(x)), bscale, xy)
        @test isapprox(bscale_grads[1].logdelta.values[end][end], sum(xy[16:20,16:30].*exp(bscale.logdelta.values[end][end])))
        @test bscale_grads[2] == zeros(M,N) + exp(bscale.logdelta)

        ##################################
        # Batch Shift
        bshift = PM.BatchShift(col_batches, row_batches)
        @test length(bshift.theta.values) == n_col_batches
        @test size(bshift.theta.values[1]) == (n_row_batches,)
        @test bshift(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .+ bshift.theta.values[1][1]

        bshift_grads = Zygote.gradient((f,x)->sum(f(x)), bshift, xy)
        @test bshift_grads[1].theta.values == Tuple(ones(n_row_batches).*(div(M,n_row_batches)*div(N,n_col_batches)) for j=1:n_col_batches)
        @test bshift_grads[2] == ones(M,N)

    end
end


function preprocess_tests()

    test_sif_path = "test_pathway.sif"

    test_sif_contents = [["MOL:GTP",   "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["METAPHASE", "a>", "PLK1"],
                         ["PAK1",      "a>", "PLK1"],
                         ["RAB1A",     "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PLK1",      "a>", "SGOL1"],
                         ["PLK1",      "a>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PP2A-ALPHA B56", "a|", "SGOL1"]
                        ]

    test_pwy_edges = [["MOL:GTP", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["METAPHASE", "PLK1", 1],
                      ["PAK1", "PLK1", 1],
                      ["RAB1A", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["PLK1", "SGOL1", 1],
                      ["PLK1", "GRASP65/GM130/RAB1/GTP/PLK1", 1],
                      ["PP2A-ALPHA B56", "SGOL1", -1]
                     ]
    tuplify = edge -> [(edge[1],"activation"),(edge[2],"activation"),edge[3]]
    test_pwy_edges = map(tuplify, test_pwy_edges)
        
    feature_genes = ["PLK1","PLK1","PLK1", 
                     "PAK1", "PAK1", "PAK1",
                     "SGOL1", 
                     "BRCA", "BRCA"]
    feature_assays = ["cna", "mutation","mrnaseq", 
                      "rppa", "methylation", "mutation",
                      "mrnaseq",
                      "mrnaseq", "methylation"]
    full_geneset = Set(feature_genes)


    test_dogma_edges = [[("PLK1","dna"), ("PLK1","mrna"), 1],
                        [("PAK1","dna"), ("PAK1","mrna"), 1],
                        [("PAK1","mrna"), ("PAK1","protein"), 1],
                       ]
    #test_dogma_edges = map(tuplify, test_dogma_edges)


    test_data_edges = [[("PLK1","dna"), ("PLK1","cna"), 1],
                       [("PLK1","dna"), ("PLK1","mutation"), -1],
                       [("PLK1","mrna"), ("PLK1","mrnaseq"),  1],
                       [("PAK1","dna"), ("PAK1","mutation"),  -1],
                       [("PAK1","mrna"),("PAK1","methylation"),  -1],
                       [("PAK1","protein"), ("PAK1","rppa"),  1],
                       [("SGOL1","mrna"), ("SGOL1","mrnaseq"),  1],
                       [("BRCA","mrna"),("BRCA","mrnaseq"),  1],
                       [("BRCA","mrna"),("BRCA","methylation"),  -1]
                      ]
   
    test_all_edges = vcat(test_data_edges, test_dogma_edges)
    tuplify2 = edge -> [Tuple(split(edge[1], "_")), Tuple(split(edge[2],"_")), edge[3]]
    test_all_edges = vcat(test_all_edges, map(tuplify2, [["PLK1_mrna", "PLK1_protein", 1],
                                                         ["PLK1_protein", "PLK1_activation", 1],
                                                         ["PAK1_protein", "PAK1_activation", 1],
                                                         ["SGOL1_mrna", "SGOL1_protein", 1],
                                                         ["SGOL1_protein", "SGOL1_activation", 1]
                                                       ]),
                          test_pwy_edges
                          )
    test_all_edge_set = Set(map(Set, test_all_edges))
                        

    @testset "Prep Pathways" begin

        # read_sif_file
        sif_data = PM.read_sif_file(test_sif_path)
        @test sif_data == test_sif_contents

        el = PM.sif_to_edgelist(sif_data)
        @test Set(el) == Set(test_pwy_edges) 

        features = collect(zip(feature_genes, feature_assays))
     
        # construct_dogma_edges
        dogma_edges, dogmax = PM.construct_dogma_edges(features)
        @test Set(dogma_edges) == Set(test_dogma_edges)

        # construct_data_edges
        data_edges = PM.construct_data_edges(features)
        @test Set(map(Set, data_edges)) == Set(map(Set, test_data_edges))
        
        # connect_pwy_to_dogma
        dogma_edges = vcat(dogma_edges, data_edges)
        all_edges = PM.connect_pwy_to_dogma(dogma_edges, el, dogmax, full_geneset)
        all_edge_set = Set(map(Set, all_edges))
        @test all_edge_set == test_all_edge_set

        # prep pathways
        pwy_edgelists = PM.sifs_to_edgelists([sif_data])
        prepped_pwy = PM.extend_pathways(pwy_edgelists, features)[1]
        all_edge_set = Set(map(Set, prepped_pwy))
        @test all_edge_set == test_all_edge_set 

        # load_pathway_sifs
        pwy_edgelists = PM.sifs_to_edgelists([test_sif_path])
        prepped_pwy = PM.extend_pathways(pwy_edgelists, features)[1]
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

    @testset "Network regularizers" begin

        #############################################
        # Test on synthetic network
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        observed = [1,2,3]
        nr = PM.NetworkRegularizer(edgelists; observed=observed)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (3,3)
        @test nr.AA[1] == sparse([1. -1. 0.;# 0;
                                  -1. 2. -1.;# 0;
                                  0. -1. 2.])# 1;
                                  #0 0 1 2])
        @test size(nr.AB[1]) == (3,1)
        @test nr.AB[1] == sparse(reshape([0.;
                                          0.;
                                          -1.], (3,1)))
        @test size(nr.BB[1]) == (1,1)
        @test nr.BB[1] == sparse(ones(1,1)*1)
        @test size(nr.B_matrix) == (2, 1)


        nr = PM.NetworkRegularizer(edgelists)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (4,4)
        @test size(nr.AB[1]) == (4,0)
        @test size(nr.BB[1]) == (0,0)
        @test size(nr.B_matrix) == (2, 0)

        ##############################################
        # Test on "real pathway"
        model_features = collect(zip(feature_genes, feature_assays))
        pwy_edgelists = PM.sifs_to_edgelists([test_sif_path])
        prepped_pwys = PM.extend_pathways(pwy_edgelists, model_features)
        pwy_nodes = Set()
        for el in prepped_pwys
            for edge in el
                push!(pwy_nodes, edge[1])
                push!(pwy_nodes, edge[2])
            end
        end
        netreg = PM.NetworkRegularizer(prepped_pwys; observed=model_features)
      
        n_obs = length(model_features) 
        n_unobs = length(pwy_nodes) - n_obs
        @test length(netreg.AA) == 1
        @test size(netreg.AA[1]) == (n_obs,n_obs)
        @test size(netreg.AB[1]) == (n_obs,n_unobs)
        @test size(netreg.BB[1]) == (n_unobs, n_unobs)

        @test size(netreg.B_matrix) == (1, n_unobs)

        ################################################
        # Gradient test
        edgelists = [[[1, 2, 1.0],[1, 3, 1.0],[1, 4, 1.0]],
                    ]
        observed = [2,3,4]
        nr = PM.NetworkRegularizer(edgelists; observed=observed)

        @test length(nr.AA) == 1
        @test nr.AA[1] == sparse([1.0 0.0 0.0;
                                  0.0 1.0 0.0;
                                  0.0 0.0 1.0])
        @test nr.AB[1] == sparse(reshape([-1.0;
                                          -1.0;
                                          -1.0;], (3,1)))
        @test nr.BB[1] == sparse(reshape([3.0], (1,1) ))
        @test nr.B_matrix == Matrix(reshape([0.0], (1,1) ))

        mu_regularizer = (x, reg) -> reg(x)

        loss, grads = Zygote.withgradient(mu_regularizer, [1.0, 1.0, 1.0], nr)
        @test loss == 0.5
        @test isapprox(grads[1], [1.0, 1.0, 1.0] ./3.0 )
        @test isapprox(grads[2].B_matrix, reshape([-1.0], (1,1)))

    end
    
    @testset "Network L1 regularizers" begin

        #############################################
        # Test on synthetic network
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3,5]
        nr = PM.NetworkL1Regularizer(data_features, edgelists; epsilon=0.0)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (4,4)
        @test dropzeros(nr.AA[1]) == sparse([1. -1. 0.  0;
                                            -1.  2. -1. 0;
                                             0. -1. 2.  0;
                                             0   0  0   0])
        @test size(nr.AB[1]) == (4,1)
        @test nr.AB[1] == sparse(reshape([0.;
                                          0.;
                                         -1.;
                                          0 ], (4,1)))
        @test size(nr.BB[1]) == (1,1)
        @test nr.BB[1] == sparse(ones(1,1))
        @test size(nr.net_virtual[1]) == (1,)

        @test nr.l1_feat_idx[1] == [false,false,false,true] 

        nr = PM.NetworkL1Regularizer(data_features, edgelists; 
                                     l1_features=[[2,5],[2,5]], epsilon=0.0)
        @test length(nr.AA) == 2
        @test size(nr.AA[1]) == (4,4)
        @test size(nr.AB[1]) == (4,1)
        @test size(nr.BB[1]) == (1,1)
        @test size(nr.net_virtual[1]) == (1,)
        @test nr.l1_feat_idx[1] == [false,true,false,true] 

        ##############################################
        # Test on "real pathway"
        model_features = collect(zip(feature_genes, feature_assays))
        pwy_edgelists = PM.sifs_to_edgelists([test_sif_path])
        prepped_pwys = PM.extend_pathways(pwy_edgelists, model_features)
        pwy_nodes = Set()
        for el in prepped_pwys
            for edge in el
                push!(pwy_nodes, edge[1])
                push!(pwy_nodes, edge[2])
            end
        end
        netreg = PM.NetworkL1Regularizer(model_features, prepped_pwys; epsilon=0.0)
        
        n_unobs = length(setdiff(pwy_nodes, model_features))
        n_obs = length(model_features)

        @test length(netreg.AA) == 1
        @test size(netreg.AA[1]) == (n_obs,n_obs)
        @test size(netreg.AB[1]) == (n_obs,n_unobs)
        @test size(netreg.BB[1]) == (n_unobs, n_unobs)

        @test size(netreg.net_virtual[1]) == (n_unobs,)
        @test length(netreg.l1_feat_idx[1]) == n_obs 
        @test typeof(netreg.l1_feat_idx[1]) == Vector{Bool}

    end

    @testset "BatchArray regularizers" begin
        col_batches = ["cat", "cat", "cat", "dog", "dog", "fish"]
        row_batches = [[1,1,1,2,2], [1,1,2,2,2], [1,1,1,1,2]]
        values = [Dict(1=>3.14, 2=>2.7), Dict(1=>0.0, 2=>0.5), Dict(1=>-1.0, 2=>1.0)]
        
        ba = PM.BatchArray(col_batches, row_batches, values)

        ba_reg = PM.BatchArrayReg(ba; weight=1.0)

        @test sum(map(sum, ba_reg.counts)) == 6*5
        grads = Zygote.gradient((r,ba)->r(ba), ba_reg, ba)

    end
end


function assemble_model_tests()

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
    N = length(feature_genes)

    sample_ids = [string("sample_",i) for i=1:20]
    group_ids = repeat(["group_1","group_2"], inner=10)

    pwy_sif_data = PM.read_sif_file(test_sif_path)
    
    @testset "Model Assembly" begin

        # create_group_edgelist
        edgelist = PM.create_group_edgelist(sample_ids, group_ids)
        test_edgelist = [[gp, samp, 1] for (samp, gp) in zip(sample_ids, group_ids)]
        @test edgelist == test_edgelist 

        # assemble_model
        sample_batch_dict = Dict([k => copy(group_ids) for k in unique(feature_assays)])
        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, group_ids,
                               feature_genes, feature_assays,
                               sample_batch_dict)

        @test feature_genes == model.data_genes
        @test feature_assays == model.data_assays
        @test length(model.used_feature_idx) == length(feature_genes)
        @test sum(model.matfac.Y_reg.l1_feat_idx[1]) == 2
    end
end


function fit_tests()
    
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
    N = length(feature_genes)
    m_groups = 2
    m_batches = 5

    sample_ids = [string("patient_",i) for i=1:M]
    sample_conditions = repeat([string("group_",i) for i=1:m_groups], inner=div(M,m_groups))
    sample_batches = repeat([string("batch_",i) for i=1:m_batches], inner=div(M,m_batches))
        
    sample_batch_dict = Dict([k => copy(sample_batches) for k in unique(feature_assays)])

    omic_data = randn(M,N)

    logistic_cols = Int[i for (i, a) in enumerate(feature_assays) if a in ("mutation","methylation")]
    n_logistic = length(logistic_cols)
    omic_data[:,logistic_cols] .= rand([0.0,1.0], M, n_logistic)

    ordinal_cols = Int[i for (i, a) in enumerate(feature_assays) if a in ("cna",)]
    n_ordinal = length(ordinal_cols)
    omic_data[:,ordinal_cols] .= rand([1.0, 2.0, 3.0], M, n_ordinal)
   
    @testset "Fit CPU" begin

        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               feature_genes, feature_assays,
                               sample_batch_dict;
                               lambda_layer=0.1)
        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)

        fit!(model, omic_data; verbosity=1, lr=0.07, max_epochs=0)
        
        fit!(model, omic_data; verbosity=1, lr=0.07, max_epochs=10)

        @test true
        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
    end

    @testset "Fit GPU" begin

        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               feature_genes, feature_assays,
                               sample_batch_dict;
                               lambda_layer=0.1)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)

        model_gpu = gpu(model)
        omic_data_gpu = gpu(omic_data)

        fit!(model_gpu, omic_data_gpu; verbosity=1, lr=0.07, max_epochs=0)

        omic_data_gpu = gpu(omic_data)

        fit!(model_gpu, omic_data_gpu; verbosity=1, lr=0.07, max_epochs=10)

        model = cpu(model_gpu)

        @test true
        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
    end
end


function model_io_tests()

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

    sample_ids = [string("patient_",i) for i=1:M]
    sample_conditions = repeat([string("group_",i) for i=1:m_groups], inner=5)
        
    sample_batch_dict = Dict([k => copy(sample_conditions) for k in unique(feature_assays)])

    test_bson_path = "test_model.bson"

    @testset "Model IO" begin

        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               feature_genes, feature_assays,
                               sample_batch_dict)

        save_model(test_bson_path, model)

        recovered_model = load_model(test_bson_path)

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
    #assemble_model_tests()
    fit_tests()
    model_io_tests()
    simulation_tests()

end

main()


