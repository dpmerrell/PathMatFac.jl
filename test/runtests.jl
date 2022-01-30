

using Test, PathwayMultiomics


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


    @testset "Preprocessing" begin

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

        # initialize_featuremap


        # populate_featuremap
        
        # load_pathways

        # load_pathway_sifs


    end

end


function main()

    util_tests()
    preprocess_tests()

end

main()


