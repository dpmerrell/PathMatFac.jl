

using Test, PathwayMultiomics, SparseArrays, LinearAlgebra, ScikitLearnBase


PM = PathwayMultiomics


function util_tests()

    values = ["cat","dog","fish","bird"]

    b_values = ["dog","bird","cat"]

    feature_list = [ ("GENE1","mrnaseq"), ("GENE2","methylation"), 
                     ("GENE3","cna"), ("GENE4", "mutation"), ("VIRTUAL1", "")]

    @testset "Utility functions" begin

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
                                                 ("VIRTUAL1", "")
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


function preprocess_tests()

    test_sif_path = "test_pathway.sif"

    test_sif_contents = [["MOL:GTP",   "hca>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["METAPHASE", "bpa>", "PLK1"],
                         ["PAK1",      "ppa>", "PLK1"],
                         ["RAB1A",     "pca>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PLK1",      "ppa>", "SGOL1"],
                         ["PLK1",      "pca>", "GRASP65/GM130/RAB1/GTP/PLK1"],
                         ["PP2A-ALPHA B56", "cpa|", "SGOL1"]
                        ]

    test_pwy_edges = [["MOL:GTP_chemical", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                      ["METAPHASE_abstract", "PLK1_activation", 1],
                      ["PAK1_activation", "PLK1_activation", 1],
                      ["RAB1A_activation", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                      ["PLK1_activation", "SGOL1_activation", 1],
                      ["PLK1_activation", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                      ["PP2A-ALPHA B56_compound", "SGOL1_activation", -1]
                     ]
    tuplify = edge -> [(edge[1],""),(edge[2],""),edge[3]]
    test_pwy_edges = map(tuplify, test_pwy_edges)
        
    feature_genes = ["PLK1","PLK1","PLK1", 
                     "PAK1", "PAK1", "PAK1",
                     "SGOL1", 
                     "BRCA", "BRCA"]
    feature_assays = ["cna", "mutation","mrnaseq", 
                      "rppa", "methylation", "mutation",
                      "mrnaseq",
                      "mrnaseq", "methylation"]

    test_dogma_edges = [["PLK1_dna", "PLK1_mrna", 1],
                        ["PAK1_dna", "PAK1_mrna", 1],
                        ["PAK1_mrna", "PAK1_protein", 1],
                       ]
    test_dogma_edges = map(tuplify, test_dogma_edges)


    test_data_edges = [[("PLK1_dna",""), ("PLK1","cna"), 1],
                       [("PLK1_dna",""), ("PLK1","mutation"), -1],
                       [("PLK1_mrna",""), ("PLK1","mrnaseq"),  1],
                       [("PAK1_dna",""), ("PAK1","mutation"),  -1],
                       [("PAK1_mrna",""),("PAK1","methylation"),  -1],
                       [("PAK1_protein",""), ("PAK1","rppa"),  1],
                       [("SGOL1_mrna",""), ("SGOL1","mrnaseq"),  1],
                       [("BRCA_mrna",""),("BRCA","mrnaseq"),  1],
                       [("BRCA_mrna",""),("BRCA","methylation"),  -1]
                      ]
   
    test_all_edges = vcat(test_data_edges, test_dogma_edges)
    test_all_edges = vcat(test_all_edges, map(tuplify, [["PLK1_mrna", "PLK1_protein", 1],
                                                        ["PLK1_protein", "PLK1_activation", 1],
                                                        ["PAK1_protein", "PAK1_activation", 1],
                                                        ["SGOL1_mrna", "SGOL1_protein", 1],
                                                        ["SGOL1_protein", "SGOL1_activation", 1]
                                                       ]),
                          test_pwy_edges
                          )
                        

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
        @test Set([Set(edge) for edge in data_edges]) == Set([Set(edge) for edge in test_data_edges])
        
        # connect_pwy_to_dogma
        dogma_edges = vcat(dogma_edges, data_edges)
        all_edges = PM.connect_pwy_to_dogma(dogma_edges, el, dogmax)
        all_edge_set = Set(map(Set, all_edges))
        test_all_edge_set = Set(map(Set, test_all_edges))
        @test Set([Set(edge) for edge in all_edges]) == Set([Set(edge) for edge in test_all_edges])

        # prep pathways
        pwy_edgelists = PM.sifs_to_edgelists([sif_data])
        prepped_pwy = PM.extend_pathways(pwy_edgelists, features)[1]
        all_edge_set = Set(map(Set, prepped_pwy))
        test_all_edge_set = Set(map(Set, test_all_edges))
        @test Set([Set(edge) for edge in prepped_pwy]) == Set([Set(edge) for edge in test_all_edges]) 

        ## load_pathway_sifs
        pwy_edgelists = PM.sifs_to_edgelists([test_sif_path])
        prepped_pwy = PM.extend_pathways(pwy_edgelists, features)[1]
        all_edge_set = Set(map(Set, prepped_pwy))
        test_all_edge_set = Set(map(Set, test_all_edges))
        @test Set([Set(edge) for edge in prepped_pwy]) == Set([Set(edge) for edge in test_all_edges]) 

    end
end


function network_reg_tests()

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
        @test nr.AA[1] == sparse([2. -1. 0.;# 0;
                                  -1. 3. -1.;# 0;
                                  0. -1. 3.])# 1;
                                  #0 0 1 2])
        @test size(nr.AB[1]) == (3,1)
        @test nr.AB[1] == sparse(reshape([0.;
                                          0.;
                                          -1.], (3,1)))
        @test size(nr.BB[1]) == (1,1)
        @test nr.BB[1] == sparse(ones(1,1)*2)
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

    end
    
    @testset "Network L1 regularizers" begin

        #############################################
        # Test on synthetic network
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3,5]
        nr = PM.NetworkL1Regularizer(data_features, edgelists)
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
                                     l1_features=[[2,5],[2,5]])
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
        netreg = PM.NetworkL1Regularizer(model_features, prepped_pwys)
        
        n_unobs = length([node for node in pwy_nodes if node[2] == ""])
        n_obs = length(model_features)

        @test length(netreg.AA) == 1
        @test size(netreg.AA[1]) == (n_obs,n_obs)
        @test size(netreg.AB[1]) == (n_obs,n_unobs)
        @test size(netreg.BB[1]) == (n_unobs, n_unobs)

        @test size(netreg.net_virtual[1]) == (n_unobs,)
        @test length(netreg.l1_feat_idx[1]) == n_obs 
        @test typeof(netreg.l1_feat_idx[1]) == Vector{Bool}
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

        @test feature_genes[model.feature_idx] == model.feature_genes
        @test feature_assays[model.feature_idx] == model.feature_assays
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

    logistic_cols = Int[i for (i, a) in enumerate(feature_assays) if a in ("mutation",)]
    n_logistic = length(logistic_cols)
    omic_data[:,logistic_cols] .= rand([0.0,1.0], M, n_logistic)

    ordinal_cols = Int[i for (i, a) in enumerate(feature_assays) if a in ("cna",)]
    n_ordinal = length(ordinal_cols)
    omic_data[:,ordinal_cols] .= rand([1.0, 2.0, 3.0], M, n_ordinal)
   
    @testset "Fit" begin

        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               feature_genes, feature_assays,
                               sample_batch_dict)

        fit!(model, omic_data; verbose=true, lr=0.07, max_epochs=10)

        @test true
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
    N = 7 # (the number of features that actually occur in our pathways) 

    sample_ids = [string("patient_",i) for i=1:M]
    sample_conditions = repeat([string("group_",i) for i=1:m_groups], inner=5)
        
    sample_batch_dict = Dict([k => copy(sample_conditions) for k in unique(feature_assays)])

    @testset "Data Simulation" begin

        n_pwys = 3
        pathway_sif_data = repeat([test_sif_path], n_pwys)
        pathway_names = [string("test_pwy_",i) for i=1:n_pwys]

        assay_moments_dict = Dict("mrnaseq"=>(5.0, 14.0),
                                  "rppa"=>(0.0, 1.0),
                                  "cna"=>(0.01,),
                                  "methylation"=>(3.0,10.0),
                                  "mutation"=>(0.001,)
                                 )

        model, params, D = PM.simulate_data(pathway_sif_data, 
                                            pathway_names,
                                            sample_ids, 
                                            sample_conditions,
                                            sample_batch_dict,
                                            feature_genes, 
                                            feature_assays,
                                            assay_moments_dict;
                                            mu_snr=10.0,
                                            delta_snr=10.0,
                                            logistic_snr=100.0,
                                            sample_snr=10.0
                                           )
        @test size(D) == (M,N)
    end
end

function main()

    util_tests()
    preprocess_tests()
    network_reg_tests()
    assemble_model_tests()
    fit_tests()
    model_io_tests()
    #simulation_tests()

end

main()


