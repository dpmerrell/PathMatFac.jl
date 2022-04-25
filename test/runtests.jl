

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
        @test PM.get_loss(("BRCA","mutation")) == "logistic"

        # sort_features
        @test PM.sort_features(feature_list) == [("VIRTUAL1", ""), ("GENE3","cna"),
                                                 ("GENE4", "mutation"), ("GENE2","methylation"),
                                                 ("GENE1", "mrnaseq")]

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
        test_spmat = SparseMatrixCSC([ 3.1 -1   0  1; 
                                      -1   3.1 -1  0; 
                                       0   -1  3.1 1;
                                       1    0   1 3.1])
        spmat = PM.edgelist_to_spmat(edgelist, node_to_idx; epsilon=0.1)

        @test isapprox(SparseMatrixCSC(spmat), test_spmat)

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

    test_extended_pwy = [["MOL:GTP_chemical", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                         ["METAPHASE_abstract", "PLK1_activation", 1],
                         ["PAK1_activation", "PLK1_activation", 1],
                         ["RAB1A_activation", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                         ["PLK1_activation", "SGOL1_activation", 1],
                         ["PLK1_activation", "GRASP65/GM130/RAB1/GTP/PLK1_compound", 1],
                         ["PP2A-ALPHA B56_compound", "SGOL1_activation", -1],
                         ["PLK1_dna", "PLK1_mrna", 1],
                         ["PLK1_mrna", "PLK1_protein", 1],
                         ["PLK1_protein", "PLK1_activation", 1],
                         ["PAK1_dna", "PAK1_mrna", 1],
                         ["PAK1_mrna", "PAK1_protein", 1],
                         ["PAK1_protein", "PAK1_activation", 1],
                         ["RAB1A_dna", "RAB1A_mrna", 1],
                         ["RAB1A_mrna", "RAB1A_protein", 1],
                         ["RAB1A_protein", "RAB1A_activation", 1],
                         ["SGOL1_dna", "SGOL1_mrna", 1],
                         ["SGOL1_mrna", "SGOL1_protein", 1],
                         ["SGOL1_protein", "SGOL1_activation", 1],
                        ]

    test_nodes = Set(["MOL:GTP_chemical", "GRASP65/GM130/RAB1/GTP/PLK1_compound",
                      "METAPHASE_abstract", "PP2A-ALPHA B56_compound",
                      "PLK1_dna", "PLK1_mrna",
                      "PLK1_mrna", "PLK1_protein",
                      "PLK1_protein", "PLK1_activation",
                      "PAK1_dna", "PAK1_mrna",
                      "PAK1_mrna", "PAK1_protein",
                      "PAK1_protein", "PAK1_activation",
                      "RAB1A_dna", "RAB1A_mrna",
                      "RAB1A_mrna", "RAB1A_protein",
                      "RAB1A_protein", "RAB1A_activation",
                      "SGOL1_dna", "SGOL1_mrna",
                      "SGOL1_mrna", "SGOL1_protein",
                      "SGOL1_protein", "SGOL1_activation"])
    test_nodes = Set([(n,"") for n in test_nodes])

    
    test_datanode_pwy = [[("MOL:GTP_chemical",""), ("GRASP65/GM130/RAB1/GTP/PLK1_compound",""), 1],
                         [("METAPHASE_abstract",""), ("PLK1_activation",""), 1],
                         [("PAK1_activation",""), ("PLK1_activation",""), 1],
                         [("RAB1A_activation",""), ("GRASP65/GM130/RAB1/GTP/PLK1_compound",""), 1],
                         [("PLK1_activation",""), ("SGOL1_activation",""), 1],
                         [("PLK1_activation",""), ("GRASP65/GM130/RAB1/GTP/PLK1_compound",""), 1],
                         [("PP2A-ALPHA B56_compound",""), ("SGOL1_activation",""), -1],
                         [("PLK1_dna",""), ("PLK1","cna"), 1],
                         [("PLK1_dna",""), ("PLK1","mutation"), -1],
                         [("PLK1_mrna",""), ("PLK1","mrnaseq"),  1],
                         [("PLK1_dna",""), ("PLK1_mrna",""), 1],
                         [("PLK1_mrna",""), ("PLK1_protein",""), 1],
                         [("PLK1_protein",""), ("PLK1_activation",""), 1],
                         [("PAK1_dna",""), ("PAK1","mutation"),  -1],
                         [("PAK1_mrna",""),("PAK1","methylation"),  -1],
                         [("PAK1_protein",""), ("PAK1","rppa"),  1],
                         [("PAK1_dna",""), ("PAK1_mrna",""), 1],
                         [("PAK1_mrna",""), ("PAK1_protein",""), 1],
                         [("PAK1_protein",""), ("PAK1_activation",""), 1],
                         [("RAB1A_dna",""), ("RAB1A_mrna",""), 1],
                         [("RAB1A_mrna",""), ("RAB1A_protein",""), 1],
                         [("RAB1A_protein",""), ("RAB1A_activation",""), 1],
                         [("SGOL1_mrna",""), ("SGOL1","mrnaseq"),  1],
                         [("SGOL1_dna",""), ("SGOL1_mrna",""), 1],
                         [("SGOL1_mrna",""), ("SGOL1_protein",""), 1],
                         [("SGOL1_protein",""), ("SGOL1_activation",""), 1],
                        ]

    @testset "Prep Pathways" begin

        # read_sif_file
        sif_data = PM.read_sif_file(test_sif_path)
        @test sif_data == test_sif_contents

        # extend_pathway
        extended_pwy = PM.extend_pathway(sif_data)
        @test Set(extended_pwy) == Set(test_extended_pwy)

        # tag_pathway
        tagged_pwy = PM.tag_pathway(extended_pwy)

        # get_all_proteins
        proteins = PM.get_all_proteins([tagged_pwy])
        @test proteins == Set(["PLK1","PAK1","RAB1A","SGOL1"])

        # get_all_nodes
        all_nodes = PM.get_all_nodes(tagged_pwy)
        @test all_nodes == test_nodes 
    
        feature_genes = ["PLK1","PLK1","PLK1", 
                         "PAK1", "PAK1", "PAK1",
                         "SGOL1", 
                         "BRCA", "BRCA"]
        feature_assays = ["cna", "mutation","mrnaseq", 
                          "rppa", "methylation", "mutation",
                          "mrnaseq",
                          "mrnaseq", "methylation"]

        # initialize_featuremap
        unique_assays = unique(feature_assays) 
        featuremap = PM.initialize_featuremap(proteins, unique_assays)

        # populate_featuremap
        featuremap = PM.populate_featuremap(featuremap, feature_genes, 
                                                        feature_assays)
        # For now, we're only concerining ourselves with data that are
        # somehow related to the pathway. 
        # It's kind of a "closed world" hypothesis.
        test_featuremap = Dict(("PLK1", "cna") => Int[1],
                                 ("PLK1", "mutation") => Int[2],
                                 ("PLK1", "mrnaseq") => Int[3],
                                 ("PLK1", "methylation") => Int[],
                                 ("PLK1", "rppa") => Int[],
                                 ("PAK1", "cna") => Int[],
                                 ("PAK1", "mrnaseq") => Int[],
                                 ("PAK1", "rppa") => Int[4],
                                 ("PAK1", "methylation") => Int[5],
                                 ("PAK1", "mutation") => Int[6],
                                 ("SGOL1", "mutation") => Int[],
                                 ("SGOL1", "cna") => Int[],
                                 ("SGOL1", "methylation") => Int[],
                                 ("SGOL1", "rppa") => Int[],
                                 ("SGOL1", "mrnaseq") => Int[7],
                                 ("RAB1A", "mrnaseq") => Int[],
                                 ("RAB1A", "methylation") => Int[],
                                 ("RAB1A", "mutation") => Int[],
                                 ("RAB1A", "cna") => Int[],
                                 ("RAB1A", "rppa") => Int[])
        @test featuremap == test_featuremap
        
        # load_pathways
        loaded_pwys, loaded_fmap = PM.load_pathways([sif_data], feature_genes, 
                                                                feature_assays)
        @test loaded_pwys == [tagged_pwy] 
        @test loaded_fmap == test_featuremap

        # load_pathway_sifs
        loaded_pwys, loaded_fmap = PM.load_pathways([test_sif_path], 
                                                     feature_genes, 
                                                     feature_assays)
        @test loaded_pwys == [tagged_pwy] 
        @test loaded_fmap == test_featuremap

        
        # add_data_nodes_sparse_latent
        unique_assays = unique(feature_assays)
        loaded_pathway = loaded_pwys[1]
        pathway = PM.add_data_nodes_sparse_latent(loaded_pathway,
                                                  loaded_fmap, 
                                                  unique_assays, 
                                                  PM.DEFAULT_ASSAY_MAP)
        @test Set(pathway) == Set(test_datanode_pwy)

        
        # prep_pathways
        prepped_pwys, prepped_features = PM.prep_pathways([test_sif_path],
                                                          feature_genes,
                                                          feature_assays)
        @test Set(prepped_pwys[1]) == Set(test_datanode_pwy)

        
        # assemble_feature_reg_mats
        feature_reg_mats, 
        assay_reg_mat, 
        augmented_features, 
        aug_feat_to_idx = PM.assemble_feature_reg_mats([test_sif_path], 
                                                       feature_genes, 
                                                       feature_assays)

        test_aug_features = Set([node for edge in pathway for node in edge[1:2]])
        union!(test_aug_features, [(assay,"") for assay in unique_assays])
        @test Set(augmented_features) == test_aug_features

        feat_mat = SparseMatrixCSC(feature_reg_mats[1])
        @test all([feat_mat[aug_feat_to_idx[edge[1]], 
                            aug_feat_to_idx[edge[2]]] != 0 for edge in test_datanode_pwy])
        @test issymmetric(feat_mat) 
              

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
    loaded_pwys, feature_map = PM.load_pathways([test_sif_path], 
                                                 feature_genes, 
                                                 feature_assays)
    loaded_pathway = loaded_pwys[1]

    pwy_sif_data = PM.read_sif_file(test_sif_path)
    

    @testset "Model Assembly" begin

        M = 10
        m_groups = 2

        sim_feature_genes = ["gene1", "gene2", "gene3", "gene2", "gene3"]
        sim_feature_assays = ["assay1", "assay1", "assay1", "assay2", "assay2"]
        sim_features = collect(zip(sim_feature_genes,sim_feature_assays))
        sim_virtual_features = [("gene1_dna",""), ("gene1_mrna",""), ("gene1_protein","")] 
        sim_augmented_features = [sim_features; sim_virtual_features]

        sim_N = length(sim_features)

        # augment_samples
        sample_ids = [string("patient_",i) for i=1:M]
        group_ids = repeat([string("group_",i) for i=1:m_groups], inner=5)
        augmented_samples = PM.augment_samples(sample_ids, group_ids; rooted=false)
        @test augmented_samples == [sample_ids; [string("group_",i) for i=1:m_groups]]

        # create_sample_edgelist
        edgelist = PM.create_sample_edgelist(sample_ids, group_ids)
        test_edgelist = [[gp, samp, 1] for (samp, gp) in zip(sample_ids, group_ids)]
        @test edgelist == test_edgelist 

        # assemble_sample_reg_mat
        sample_reg_mat, 
        augmented_samples, 
        aug_sample_to_idx = PM.assemble_sample_reg_mat(sample_ids, group_ids)
        sample_reg_mat = SparseMatrixCSC(sample_reg_mat)
        @test all([sample_reg_mat[aug_sample_to_idx[sample_id], 
                            aug_sample_to_idx[group_id]] != 0 for (sample_id, group_id) in zip(sample_ids,group_ids)])
        @test issymmetric(sample_reg_mat)
        
        # update_sample_batch_dict 
        sample_batch_dict = Dict([k => copy(group_ids) for k in unique(sim_feature_assays)])
        
        new_batch_dict = PM.update_sample_batch_dict(sample_batch_dict,
                                                     sample_ids, augmented_samples,
                                                     aug_sample_to_idx)

        test_new_batch_dict = Dict("assay1" => [group_ids; ["",""]],
                                   "assay2" => [group_ids; ["",""]],
                                   "" => repeat([""], inner=M + m_groups))

        @test new_batch_dict == test_new_batch_dict

        # assemble_model
        sample_batch_dict = Dict([k => copy(group_ids) for k in unique(feature_assays)])
        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, group_ids,
                               sample_batch_dict,
                               feature_genes, feature_assays)

        @test PM.is_contiguous(model.matfac.feature_batch_ids)
        
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

    sample_ids = [string("patient_",i) for i=1:M]
    sample_conditions = repeat([string("group_",i) for i=1:m_groups], inner=5)
        
    sample_batch_dict = Dict([k => copy(sample_conditions) for k in unique(feature_assays)])

    omic_data = randn(M,N)
    logistic_cols = Int[i for (i, a) in enumerate(feature_assays) if a in ("cna","mutation")]
    n_logistic = length(logistic_cols)
    omic_data[:,logistic_cols] .= rand([0.0,1.0], M, n_logistic)
   
    @testset "Fit" begin

        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               sample_batch_dict,
                               feature_genes, feature_assays)

        fit!(model, omic_data)

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

    test_hdf_path = "test_model.hdf"

    @testset "Model IO" begin


        model = MultiomicModel([test_sif_path, test_sif_path, test_sif_path],  
                               [string(test_pwy_name,"_",i) for i=1:3],
                               sample_ids, sample_conditions,
                               sample_batch_dict,
                               feature_genes, feature_assays)

        save_hdf(model, test_hdf_path)

        recovered_model = load_hdf(test_hdf_path)

        @test recovered_model == model

        rm(test_hdf_path)
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
    #assemble_model_tests()
    #fit_tests()
    #model_io_tests()
    #simulation_tests()

end

main()


