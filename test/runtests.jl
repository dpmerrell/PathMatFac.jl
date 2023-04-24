

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

        # subset_ranges
        @test PM.subset_ranges(my_ranges, 2:5) == ([2:2,3:4,5:5], 1, 3)
        @test PM.subset_ranges(my_ranges, 2:8) == ([2:2,3:4,5:5], 1, 3)

        broken_ranges = [1:2,5:6,8:10]
        @test PM.subset_ranges(broken_ranges, 1:3) == ([1:2], 1, 1)
        @test PM.subset_ranges(broken_ranges, 5:8) == ([5:6,8:8], 2,3)

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

        # Binary search
        i = 1
        function eval_f(x)
            y = x - 10
            i += 1
            #println(string("i: ", i, "\tx: ", x, "\ty: ", y))
            return y
        end
        x_start = 1
        z_target = 0.375
        x, y = PM.func_binary_search(x_start, z_target, eval_f; max_iter=Inf,
                                                                x_atol=1e-16, z_atol=1e-16)
        @test isapprox(x, z_target + 10)
        @test isapprox(y, z_target)

        z_target = -Float64(pi)
        x_start = 1
        i = 1
        x, y = PM.func_binary_search(x_start, z_target, eval_f; max_iter=Inf, 
                                                                x_atol=1e-16, z_atol=1e-16)
        @test isapprox(x, z_target + 10)
        @test isapprox(y, z_target) 
    end
end


function batch_array_tests()
    
    @testset "Batch Arrays" begin

        col_batches = ["cat", "cat", "cat", "bird", "dog", "dog", "fish"]
        row_batches = Dict("cat"=>[1,1,1,2,2], "dog"=>[1,1,2,2,2], "fish"=>[1,1,1,1,2])
        values = [Dict(1=>fill(3.14,3), 2=>fill(2.7,3)), Dict(), Dict(1=>fill(0.0,2), 2=>fill(0.5,2)), Dict(1=>fill(-1.0,1), 2=>fill(1.0,1))]
        A = zeros(5,7)

        ## Fit batch shift
        #sample_conditions = [1,1,2,2,2]
        #model = PathMatFacModel(A; K=2, sample_conditions=sample_conditions, feature_views=col_batches, batch_dict=row_batches)
        #model.matfac.X .= 0
        #model.matfac.Y .= 0
        #PM.freeze_layer!(model.matfac.col_transform, [1,2,3])
        #PM.mf_fit!(model; max_epochs=10, verbosity=2, update_layers=true)

        ##############################
        # Constructor
        ba = PM.BatchArray(col_batches, row_batches, values)
        @test ba.col_ranges == (1:3, 5:6, 7:7)
        test_row_batches = (sparse(Bool[1 0; 1 0; 1 0; 0 1; 0 1]),
                            sparse(Bool[1 0; 1 0; 0 1; 0 1; 0 1]),
                            sparse(Bool[1 0; 1 0; 1 0; 1 0; 0 1]))
        @test all(map(isapprox, ba.row_batches, test_row_batches)) 
        @test ba.values == (repeat([3.14, 2.7], inner=(1,3)), 
                            repeat([0.0, 0.5], inner=(1,2)),
                            repeat([-1.0, 1.0], inner=(1,1)))

        ##############################
        # View
        # unitrange
        ba_view = view(ba, 2:4, 2:6)
        @test ba_view.col_ranges == (1:2, 4:5)
        @test ba_view.row_batches == map(b->b[2:4,:], test_row_batches[1:2])
        @test isapprox(ba_view.row_selector, sparse([0 1 0 0 0;
                                                     0 0 1 0 0;
                                                     0 0 0 1 0]
                                                    )
                      )
        # Does nested viewing work?
        ba_view_view = view(ba_view, 1:2, 1:3)

        @test ba_view.values == (repeat([3.14, 2.7], inner=(1,2)),
                                 repeat([0.0,0.5], inner=(1,2)))
        # vector{Int}
        ba_view = view(ba, [2,3,4], 2:6)
        @test ba_view.col_ranges == (1:2, 4:5)
        @test ba_view.row_batches == map(b->b[2:4,:], test_row_batches[1:2])
        @test isapprox(ba_view.row_selector, sparse([0 1 0 0 0;
                                                     0 0 1 0 0;
                                                     0 0 0 1 0]
                                                    )
                      )
        @test ba_view.values == (repeat([3.14, 2.7], inner=(1,2)),
                                 repeat([0.0,0.5], inner=(1,2)))
     
        #####################################
        # Figure out how gaps work
        gappy_col_batches = ["cat", "cat", "cat", "bird", "bird", "bird", "dog", "dog", "fish"]
        gappy_row_batches = Dict("cat"=>[1,1,1,2,2], "dog"=>[1,1,2,2,2], "fish"=>[1,1,1,1,2])
        gappy_values = [Dict(1=>fill(3.14,3), 2=>fill(2.7,3)), Dict(), Dict(1=>fill(0.0,2), 2=>fill(0.5,2)), Dict(1=>fill(-1.0,1), 2=>fill(1.0,1))]
        gappy_ba = PM.BatchArray(gappy_col_batches, gappy_row_batches, gappy_values)
        
        @test gappy_ba.col_ranges == (1:3, 7:8, 9:9)
        @test all(map((rb,trb) -> isapprox(rb,trb), gappy_ba.row_batches, test_row_batches))

        empty_view = view(gappy_ba, :, 4:6)
        @test empty_view.col_ranges == ()
        @test empty_view.row_batches == ()
        @test empty_view.values == () 

        ###############################
        # zero
        ba_zero = zero(ba)
        @test ba_zero.col_ranges == ba.col_ranges
        @test ba_zero.row_batches == ba.row_batches
        @test ba_zero.values == (repeat([0.0, 0.0], inner=(1,3)), 
                                 repeat([0.0, 0.0], inner=(1,2)), 
                                 repeat([0.0, 0.0], inner=(1,1))
                                )

        ###############################
        # Addition
        Z = A + ba
        test_mat = [3.14 3.14 3.14 0.0 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.0 0.0 0.0 -1.0;
                    3.14 3.14 3.14 0.0 0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.0 0.5 0.5 -1.0;
                    2.7  2.7  2.7  0.0 0.5 0.5  1.0]
        @test Z == test_mat
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x+y), A, ba)
        @test A_grad == ones(size(A)...)
        @test ba_grad.values == (repeat([3., 2.], inner=(1,3)), 
                                 repeat([2., 3.], inner=(1,2)),
                                 repeat([4., 1.], inner=(1,1))
                                )

        ################################
        # Multiplication
        Z = ones(5,7) * ba
        other_test_mat = deepcopy(test_mat)
        other_test_mat[:,4] .= 1
        @test isapprox(Z, other_test_mat)
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x*y), ones(5,7), ba)
        other_Z = deepcopy(Z)
        other_Z[:,4] .= 1
        @test isapprox(A_grad, other_Z)
        @test ba_grad.values == (repeat([3., 2.], inner=(1,3)), 
                                 repeat([2., 3.], inner=(1,2)),
                                 repeat([4., 1.], inner=(1,1))
                                )

        ################################
        # Exponentiation
        ba_exp = exp(ba)
        @test (ones(5,7) * ba_exp) == exp.(test_mat)
        (ba_grad,) = Zygote.gradient(x->sum(ones(5,7) * exp(x)), ba)
        @test ba_grad.values == (repeat([3.0*exp(3.14), 2.0*exp(2.7)], inner=(1,3)), 
                                 repeat([2.0*exp(0.0), 3.0*exp(0.5)], inner=(1,2)),
                                 repeat([4.0*exp(-1.0), 1.0*exp(1.0)], inner=(1,1))
                                )

        ################################
        # ba_map
        result = PM.ba_map(a->a, ba, test_mat)
        @test result == ([3*3.14 3*3.14 3*3.14;
                          2*2.7  2*2.7  2*2.7], 
                         [0.0  0.0;
                          1.5  1.5], 
                         reshape([-4.0;
                                 1.0], (2,1)))
        result = PM.ba_map(a->a, ba, test_mat)
        @test result == ([3*3.14 3*3.14 3*3.14;
                          2*2.7  2*2.7  2*2.7], 
                         [0.0  0.0;
                          1.5  1.5], 
                         reshape([-4.0;
                                 1.0], (2,1)))


        #################################
        # GPU
        ba_d = gpu(ba)
        @test ba_d.values == map(gpu, (repeat([3.14, 2.7], inner=(1,3)), 
                                       repeat([0.0, 0.5], inner=(1,2)),
                                       repeat([-1.0, 1.0], inner=(1,1)))
                                )
       
        # view
        ba_d_view = view(ba_d, 2:4, 2:6)
        ba_d_view = cpu(ba_d_view)
        @test ba_d_view.col_ranges == (1:2, 4:5)
        @test all(map((u,v)->isapprox(u,v), 
                       ba_d_view.row_batches, 
                       map(b->b[2:4,:], test_row_batches[1:2])
                     )
                  )
        @test isapprox(ba_d_view.row_selector, sparse([0 1 0 0 0;
                                                       0 0 1 0 0;
                                                       0 0 0 1 0]
                                                     )
                      )
        @test all(map(isapprox,
                      ba_d_view.values,
                      (repeat([3.14, 2.7], inner=(1,2)),
                       repeat([0.0,0.5], inner=(1,2))
                      )
                     )
                  )

        # zero 
        ba_d_zero = zero(ba_d)
        ba_d_zero = cpu(ba_d_zero)
        @test ba_d_zero.col_ranges == ba.col_ranges
        @test ba_d_zero.row_batches == ba.row_batches
        @test ba_d_zero.values == (repeat([0.0, 0.0], inner=(1,3)), 
                                   repeat([0.0, 0.0], inner=(1,2)), 
                                   repeat([0.0, 0.0], inner=(1,1))
                                  )

        # addition
        A_d = gpu(A)
        Z_d = A_d + ba_d
        @test isapprox(cpu(Z_d),test_mat)
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x+y), A_d, ba_d)
        A_grad = cpu(A_grad)
        ba_grad = cpu(ba_grad)
        @test isapprox(A_grad, ones(size(A)...))
        @test all(map(isapprox, ba_grad.values, (repeat([3., 2.], inner=(1,3)), 
                                                repeat([2., 3.], inner=(1,2)),
                                                repeat([4., 1.], inner=(1,1))
                                                )
                     )
                 )


        # multiplication
        Z_d = gpu(ones(5,7)) * ba_d
        Z_d = cpu(Z_d)
        @test isapprox(Z_d, other_test_mat)
        (A_grad, ba_grad) = Zygote.gradient((x,y)->sum(x*y), gpu(ones(5,7)), ba_d)
        A_grad = cpu(A_grad)
        ba_grad = cpu(ba_grad)
        other_Z_d = deepcopy(Z_d)
        other_Z_d[:,4] .= 1
        @test isapprox(A_grad, other_Z_d)
        @test all(map(isapprox, ba_grad.values, (repeat([3., 2.], inner=(1,3)), 
                                                repeat([2., 3.], inner=(1,2)),
                                                repeat([4., 1.], inner=(1,1))
                                                )
                     )
                 )

        # exponentiation
        ba_d_exp = exp(ba_d)
        @test isapprox(cpu(gpu(ones(5,7)) * ba_d_exp), exp.(test_mat))
        (ba_grad,) = Zygote.gradient(x->sum(gpu(ones(5,7)) * exp(x)), ba_d)
        ba_grad = cpu(ba_grad)
        @test all(map((u,v) -> isapprox(u,v), ba_grad.values, (repeat([3.0*exp(3.14), 2.0*exp(2.7)], inner=(1,3)), 
                                                               repeat([2.0*exp(0.0), 3.0*exp(0.5)], inner=(1,2)),
                                                               repeat([4.0*exp(-1.0), 1.0*exp(1.0)], inner=(1,1))
                                                              )
                     )
                 )

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
        @test size(bscale.logdelta.values[1]) == (n_row_batches, div(N, n_col_batches))
        @test bscale(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .* transpose(exp.(bscale.logdelta.values[1][1,:]))

        bscale_grads = Zygote.gradient((f,x)->sum(f(x)), bscale, xy)
        @test isapprox(bscale_grads[1].logdelta.values[end][end,:], vec(sum(xy[16:20,16:30].*transpose(exp.(bscale.logdelta.values[end][end,:])), dims=1)))
        @test bscale_grads[2] == zeros(M,N) + exp(bscale.logdelta)

        ##################################
        # Batch Shift
        bshift = PM.BatchShift(col_batches, batch_dict)
        @test length(bshift.theta.values) == n_col_batches
        @test size(bshift.theta.values[1]) == (n_row_batches, div(N, n_col_batches))
        @test bshift(xy)[1:div(M,n_row_batches),1:div(N,n_col_batches)] == xy[1:div(M,n_row_batches),1:div(N,n_col_batches)] .+ transpose(bshift.theta.values[1][1,:])

        bshift_grads = Zygote.gradient((f,x)->sum(f(x)), bshift, xy)
        @test bshift_grads[1].theta.values == Tuple(ones(n_row_batches, div(N,n_col_batches)).*div(M,n_row_batches) for j=1:n_col_batches)
        @test bshift_grads[2] == ones(M,N)

        ###################################
        # Layer composition
        transform_nobatch = PM.construct_model_layers(col_batches, nothing)
        @test length(transform_nobatch.layers) == 4 
        @test typeof(transform_nobatch.layers[1]) <: PM.ColScale 
        @test typeof(transform_nobatch.layers[2]) <: Function
        @test typeof(transform_nobatch.layers[3]) <: PM.ColShift
        @test typeof(transform_nobatch.layers[4]) <: Function
        transform_grads = Zygote.gradient((f,x)->sum(f(x)), transform_nobatch, xy)
        @test transform_grads[1].layers[1].logsigma != nothing
        @test transform_grads[1].layers[2] == nothing
        @test transform_grads[1].layers[3].mu != nothing
        @test transform_grads[1].layers[4] == nothing
 
        transform_w_batch = PM.construct_model_layers(col_batches, batch_dict) 
        @test length(transform_w_batch.layers) == 4 
        @test typeof(transform_w_batch.layers[1]) <: PM.ColScale 
        @test typeof(transform_w_batch.layers[2]) <: PM.BatchScale 
        @test typeof(transform_w_batch.layers[3]) <: PM.ColShift
        @test typeof(transform_w_batch.layers[4]) <: PM.BatchShift
        transform_grads = Zygote.gradient((f,x)->sum(f(x)), transform_w_batch, xy)
        @test transform_grads[1].layers[1].logsigma != nothing
        @test transform_grads[1].layers[2].logdelta != nothing
        @test transform_grads[1].layers[3].mu != nothing
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

    @testset "Prep Pathway Graphs" begin

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
    
    @testset "Prep Pathway featuresets" begin
        test_nodeset = Set(["MOL:GTP",    "GRASP65/GM130/RAB1/GTP/PLK1",
                            "METAPHASE",  "PLK1", "PAK1",       
                            "RAB1A",      "SGOL1", "PP2A-ALPHA B56", ])
        nodeset = PM.sif_to_nodeset(test_sif_contents)
        @test nodeset == test_nodeset 

        nodesets = PM.sifs_to_nodesets([test_sif_contents])
        @test nodesets[1] == test_nodeset

        test_featureset = ["PLK1_1","PLK1_2","PLK1_3", 
                           "PAK1_4", "PAK1_5", "PAK1_6",
                           "SGOL1_7"] 
        featuresets, new_feature_ids = prep_pathway_featuresets([test_sif_path], feature_genes)
        @test new_feature_ids == map((g,i) -> string(g, "_", i), feature_genes, 1:length(feature_genes))
        @test featuresets[1] == Set(test_featureset)
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
        reg = PM.SelectiveL1Reg(data_features, edgelists)
  
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
        data_groups = [1,1,1,2,2,2]
        test_Y = randn(3,6)
        reg = PM.GroupRegularizer(data_groups; K=size(test_Y, 1))
        means = [mean(test_Y[1,1:3]) mean(test_Y[1,4:6]); # Factor of 0.9 comes from default value of
                 mean(test_Y[2,1:3]) mean(test_Y[2,4:6]); # centrality (=0.1)
                 mean(test_Y[3,1:3]) mean(test_Y[3,4:6])] 
        lss, grad = Zygote.withgradient(reg, test_Y)
      
        @test isapprox(lss, 0.5*sum((test_Y).^2))
        #@test isapprox(lss, 0.5*sum((test_Y .- repeat(means, inner=(1,3))).^2))
                                #sum([sum((test_Y[1,1:3] .- means[1,1]).^2) sum((test_Y[1,4:5] .- means[1,2]).^2);
                                #sum((test_Y[2,1:3] .- means[2,1]).^2) sum((test_Y[2,4:5] .- means[2,2]).^2);
                                #sum((test_Y[3,1:3] .- means[3,1]).^2) sum((test_Y[3,4:5] .- means[3,2]).^2)]))
        test_grad = test_Y
        #test_grad = test_Y .- repeat(means, inner=(1,3))
                    #[(test_Y[1,1:3] .- means[1,1]) (test_Y[1,4:5] .- means[1,2]);
                    # (test_Y[2,1:3] .- means[2,1]) (test_Y[2,4:5] .- means[2,2]);
                    # (test_Y[3,1:3] .- means[3,1]) (test_Y[3,4:5] .- means[3,2])]
 
        @test isapprox(grad[1], test_grad) 

    end

    @testset "ARD regularizers" begin

        K = 3
        N = 5
        test_Y = randn(K,N)
        reg = PM.ARDRegularizer([1,1,1,2,2])
        l, grads = Zygote.withgradient(reg, test_Y)
        b = 1 .+ (0.5/reg.beta[1]).*(test_Y.*test_Y)
        @test isapprox(l, (0.5 + reg.alpha[1]).*sum(log.(b)))  # This should work only because alpha==beta==0.01 after initialization.
        @test length(grads) == 1
        @test isapprox(grads[1], ((0.5 .+ reg.alpha[1])/reg.beta[1]).* test_Y ./ b)
    end


    @testset "BatchArray regularizers" begin
        col_batches = ["cat", "cat", "cat", "dog", "dog", "fish"]
        row_batches = Dict("cat"=>[1,1,1,2,2], "dog"=>[1,1,2,2,2], "fish"=>[1,1,1,1,2])
        values = [Dict(1=>fill(3.14,3), 2=>fill(2.7,3)), Dict(1=>fill(0.0,2), 2=>fill(0.5,2)), Dict(1=>fill(-1.0,1), 2=>fill(1.0,1))]
        
        ba = PM.BatchArray(col_batches, row_batches, values)
        ba_reg = PM.BatchArrayReg(ba; weight=1.0)
        @test isapprox(ba_reg(ba), 0.5*sum(map((w,v)->sum(w .* v.*v), ba_reg.weights, ba.values)))

        grads = Zygote.gradient(arr->ba_reg(arr), ba)
        @test all(map((g, w, v)->isapprox(g, w.*v), grads[1].values, ba_reg.weights, ba.values)) 
 
    end

    @testset "Composed regularizers" begin
        K = 2
        N = 5
        edgelists = [[[1, 2, 1.0],[2, 3, 1.0],[3, 4, 1.0]],
                     [[1, 3, -1.0],[2, 4, -1.0]]
                    ]
        data_features = [1,2,3,4,5]
        test_Y = randn(2,5)

        l1_reg = PM.SelectiveL1Reg(data_features, edgelists)
        net_reg = PM.NetworkRegularizer(data_features, edgelists)
        composite_reg = PM.construct_composite_reg([l1_reg, net_reg], [0.5, 0.5])

        @test length(composite_reg.regularizers) == 2
        @test isapprox(composite_reg(test_Y), 0.5*(l1_reg(test_Y) + net_reg(test_Y)))
    end
end

function featureset_ard_tests()

    N = 40
    K = 10
    
    feature_ids = collect(1:N)
    feature_views = repeat([1,2], inner=div(N,2))
    view_1_idx = (feature_views .== 1)
    view_2_idx = (feature_views .== 2)
    feature_sets = [collect(1:10),collect(11:20),collect(21:30),collect(31:40)]
    n_sets = length(feature_sets)

    reg = PM.construct_featureset_ard(K, feature_ids, feature_views, feature_sets;
                                      beta0=1e-1, lr=0.1)
    
    test_A = (rand(Bool, n_sets, 10) .* randn(4,10))
    test_S = zeros(n_sets, N)
    for (i, s) in enumerate(feature_sets)
        scale = 1 / sqrt(length(s))
        test_S[i,s] .= scale
    end
    corrupt_S = (test_S)# .| rand([true, false, false, false], n_sets, N))
    Y = transpose(test_A)*corrupt_S .+ randn(K,N).*0.01

    @testset "Featureset ARD constructor" begin

        @test length(reg.feature_views) == 2
        @test reg.feature_view_ids == [1,2]
        @test all(reg.alpha .== 1e-1)
        @test isapprox(reg.beta0, 1e-1)

        @test size(reg.A) == (length(feature_sets), K)

        @test issparse(reg.S)
        @test isapprox(Matrix(reg.S), test_S)
    end

    gamma_normal_loss_fn = (beta, Y) -> -sum(transpose(reg.alpha).*log.(beta)) + sum(transpose(reg.alpha .+ 0.5).*sum(log.(beta .+ (0.5.*(Y.*Y))), dims=1))
    gamma_normal_loss_Y = Y -> gamma_normal_loss_fn(reg.beta, Y) 
    calibrated_loss_Y = Y -> gamma_normal_loss_Y(Y) - gamma_normal_loss_Y(zero(Y))

    @testset "Featureset ARD regularization" begin

        lss, Y_grads = Zygote.withgradient(reg, Y)
        @test isapprox(lss, calibrated_loss_Y(Y))

        test_grad = Zygote.gradient(gamma_normal_loss_Y, Y)[1]
        @test isapprox(test_grad, Y_grads[1])

    end

    @testset "Featureset ARD updates: alpha, scale" begin

        # Update alpha and scale
        test_alpha = zeros(Float32, N)
        test_scale = zeros(Float32, N)
        PM.update_alpha!(reg, Y)

        @test all(reg.alpha .> 0)

    end

    @testset "Featureset ARD updates: A" begin

        # Set A to random positive values
        reg.A .= 0.1.*abs.(randn(size(reg.A)))
        A_copy = deepcopy(reg.A)

        # Set lambda to a value that forces all entries to zero
        lambda_max = PM.set_lambda_max(reg, Y)
        reg.A_opt.lambda .= 1.25*lambda_max
        PM.update_A_inner!(reg, Y; max_epochs=5000, term_iter=100, verbosity=0, print_iter=1000, print_prefix="   ")
        @test isapprox(reg.A, zero(reg.A), atol=1e-3)
   
        PM.update_A!(reg, Y;
                     max_epochs=500, term_iter=100,
                     bin_search_max_iter=20,
                     bin_search_frac_atol=0.25,
                     bin_search_lambda_atol=0.0001,
                     target_frac=0.5,
                     print_iter=100,
                     verbosity=2, print_prefix="")

        @test !isapprox(reg.A, zero(reg.A))

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
        @test typeof(model.matfac.col_transform.layers[3]) == PM.ColShift
        @test model.sample_ids == collect(1:M)
        @test model.feature_ids == collect(1:N)
        @test typeof(model.matfac.noise_model.noises[1]) == MF.NormalNoise
    end

    @testset "Batch effect model constructor" begin

        model = PathMatFacModel(Z; K=K, sample_conditions=sample_conditions, feature_views=feature_views, batch_dict=batch_dict)
        @test size(model.matfac.X) == (K,M)
        @test size(model.matfac.Y) == (K,N)
        @test typeof(model.matfac.col_transform.layers[1]) == PM.ColScale
        @test typeof(model.matfac.col_transform.layers[2]) == PM.BatchScale
        @test typeof(model.matfac.col_transform.layers[3]) == PM.ColShift
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
        
        # Vanilla Group-L2 regularized model construction 
        model = PathMatFacModel(Z; K=7, lambda_Y_l2=3.14)
        @test size(model.matfac.Y) == (7,N)
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[1]) <: PM.GroupRegularizer 
        @test isapprox(model.matfac.Y_reg(model.matfac.Y), 0.5*3.14*sum(model.matfac.Y.^2))

        # Selective L1-regularized model construction 
        model = PathMatFacModel(Z; feature_ids=feature_ids, feature_graphs=feature_graphs, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.SelectiveL1Reg
        
        # Both, at the same time! 
        model = PathMatFacModel(Z; feature_ids=feature_ids, feature_graphs=feature_graphs, lambda_Y_graph=1.0, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.SelectiveL1Reg
        @test typeof(model.matfac.Y_reg.regularizers[3]) <: PM.NetworkRegularizer

    end

    @testset "X-regularized model constructor" begin
        
        # Vanilla L2-regularized model construction 
        model = PathMatFacModel(Z; K=8, lambda_X_l2=3.14)
        @test size(model.matfac.X) == (8,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[1]) <: PM.L2Regularizer 
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
        @test all(model.matfac.X_reg.regularizers[3].cur_weights .== 1.0)
        
        # Combined X regularization 
        model = PathMatFacModel(Z; sample_ids=sample_ids, sample_conditions=sample_conditions,
                                   sample_graphs=sample_graphs, lambda_X_graph=1.234, lambda_X_condition=5.678)
        @test size(model.matfac.X) == (K,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[2]) <: PM.GroupRegularizer
        @test typeof(model.matfac.X_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test all(map(w-> isapprox(w,fill(5.678, K)), model.matfac.X_reg.regularizers[2].group_weights)) 
        @test all(model.matfac.X_reg.regularizers[3].cur_weights .== 1.234)
    end

    @testset "Full-featured model constructor" begin
        model = PathMatFacModel(Z; sample_ids=sample_ids, sample_conditions=sample_conditions, sample_graphs=sample_graphs, 
                                   lambda_X_graph=1.234, lambda_X_condition=5.678, 
                                   feature_ids=feature_ids, feature_views=feature_views, 
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_Y_graph=1.0, lambda_Y_selective_l1=1.0)
        @test size(model.matfac.X) == (K,M)
        @test length(model.matfac.col_transform.layers) == 4 
        @test length(model.matfac.X_reg.regularizers) == 3 
        @test typeof(model.matfac.X_reg.regularizers[2]) <: PM.GroupRegularizer
        @test typeof(model.matfac.X_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test all(map(w->isapprox(w,fill(5.678, K)), model.matfac.X_reg.regularizers[2].group_weights))
        @test all(model.matfac.X_reg.regularizers[3].cur_weights .== 1.234)
        @test size(model.matfac.Y) == (K,N)
        @test length(model.matfac.Y_reg.regularizers) == 3 
        @test typeof(model.matfac.Y_reg.regularizers[2]) <: PM.SelectiveL1Reg
        @test typeof(model.matfac.Y_reg.regularizers[3]) <: PM.NetworkRegularizer
        @test all(model.matfac.Y_reg.regularizers[2].weight .== 1.0) 
        @test all(model.matfac.Y_reg.regularizers[3].cur_weights .== 1.0)
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


function lbfgs_tests()

    M = 6
    N = 10
    K = 1

    n_col_batches = 2
    n_row_batches = 4

    X = randn(K,M)
    Y = randn(K,N)
    Z = transpose(X)*Y

    sample_ids = [string("sample_", i) for i=1:M]
    sample_conditions = repeat(["condition_1", "condition_2"], inner=div(M,2))

    feature_ids = map(x->string("x_",x), 1:N)
    feature_views = repeat(1:n_col_batches, inner=div(N,n_col_batches))

    @testset "L-BFGS tests" begin

        model = PathMatFacModel(Z; K=1, sample_conditions=sample_conditions, feature_views=feature_views)
        
        PM.fit_lbfgs!(model.matfac, model.data)
    end

end


function fit_tests()
    
    M = 40
    N = 60
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

    L = 50
    feature_sets = [Set(rand(feature_ids, 4)) for _=1:L]
 
    ####################################################
    # Basic fitting
    ####################################################
    
    @testset "Basic fit CPU" begin

        model = PathMatFacModel(Z; sample_conditions=sample_conditions, 
                                   feature_ids=feature_ids,  feature_views=feature_views,
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2]) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                    fit_reg_weight=false)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
    end

    Z_d = gpu(Z)

    @testset "Basic fit GPU" begin

        model = PathMatFacModel(Z_d; sample_conditions=sample_conditions, 
                                   feature_ids=feature_ids,  feature_views=feature_views,
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2])

        model = gpu(model) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                    fit_reg_weight=false)
        model = cpu(model)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
    end

    ####################################################
    # Empirical Bayes fitting
    ####################################################

    @testset "Empirical Bayes fit CPU" begin

        model = PathMatFacModel(Z; sample_conditions=sample_conditions, 
                                   feature_ids=feature_ids,  feature_views=feature_views,
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2]) 
        h = fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                        fit_reg_weight="EB", keep_history=true)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
        @test isa(h, AbstractVector)
        @test isa(h[1], AbstractDict)
    end

    @testset "Empirical Bayes fit GPU" begin

        model = PathMatFacModel(Z_d; sample_conditions=sample_conditions, 
                                   feature_ids=feature_ids,  feature_views=feature_views,
                                   feature_graphs=feature_graphs, batch_dict=batch_dict, 
                                   lambda_X_l2=0.1, lambda_Y_graph=0.1, lambda_Y_selective_l1=0.05)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2])

        model = gpu(model) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                    fit_reg_weight="EB")
        model = cpu(model)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
    end
    
    ####################################################
    # Fitting model with ARD on Y
    ####################################################

    @testset "ARD fit CPU" begin

        model = PathMatFacModel(Z; K=4,
                                   sample_conditions=sample_conditions,
                                   feature_views=feature_views, 
                                   feature_ids=feature_ids,  
                                   batch_dict=batch_dict, 
                                   Y_ard=true)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2]) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
    end

    @testset "ARD fit GPU" begin

        model = PathMatFacModel(Z_d; K=4,
                                     #sample_conditions=sample_conditions, 
                                     feature_views=feature_views, 
                                     feature_ids=feature_ids,  
                                     #batch_dict=batch_dict, 
                                     Y_ard=true)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2])

        model = gpu(model) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                    keep_history=true)
        model = cpu(model)
        

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        #@test !all(map(isapprox, batch_scale.logdelta.values,
        #                        model.matfac.col_transform.layers[2].logdelta.values)
        #         ) # This should not have changed
    end
    
    ####################################################
    # Fitting model with ARD on Y
    ####################################################

    @testset "Featureset ARD fit CPU" begin

        model = PathMatFacModel(Z; K=4,
                                   sample_conditions=sample_conditions,
                                   feature_views=feature_views, 
                                   feature_ids=feature_ids,  
                                   batch_dict=batch_dict, 
                                   feature_sets=feature_sets, 
                                   Y_fsard=true)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2]) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-5, abs_tol=1e-5,
                    fsard_term_rtol=1e-3,
                    fsard_max_iter=10,
                    fsard_max_A_iter=500,
                    fsard_A_prior_frac=0.5,
                    fsard_frac_atol=0.25, fsard_lambda_atol=1e-2)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                 model.matfac.col_transform.layers[2].logdelta.values)
                 ) # This should not have changed
    end

    @testset "Featureset ARD fit GPU" begin

        model = PathMatFacModel(Z_d; K=4,
                                     sample_conditions=sample_conditions, 
                                     feature_views=feature_views, 
                                     feature_ids=feature_ids,  
                                     batch_dict=batch_dict,
                                     feature_sets=feature_sets, 
                                     Y_fsard=true)

        X_start = deepcopy(model.matfac.X)
        Y_start = deepcopy(model.matfac.Y)
        batch_scale = deepcopy(model.matfac.col_transform.layers[2])

        model = gpu(model) 
        fit!(model; verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-5, abs_tol=1e-5,
                    fsard_term_rtol=1e-3,
                    fsard_max_iter=10,
                    fsard_max_A_iter=500,
                    fsard_A_prior_frac=0.5,
                    fsard_frac_atol=0.25, fsard_lambda_atol=1e-2)

        model = cpu(model)

        @test !isapprox(model.matfac.X, X_start)
        @test !isapprox(model.matfac.Y, Y_start)
        @test !all(map(isapprox, batch_scale.logdelta.values,
                                model.matfac.col_transform.layers[2].logdelta.values)
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

    fit!(model; verbosity=1, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7)

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
                                         verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                                         use_gpu=false)
        
        @test size(result.matfac.X) == (K, M_new) 
        @test size(result.matfac.Y) == (K, 30) 
        @test all(result.sample_ids .== collect(1:M_new)) 
        #@test all(result.sample_conditions .== new_sample_conditions) 
              
    end

    @testset "Transform GPU" begin
        
        result = transform(model, D_new; feature_ids=new_feature_ids, feature_views=new_feature_views,
                                         sample_conditions=new_sample_conditions,
                                         verbosity=2, lr=0.05, max_epochs=1000, print_iter=10, rel_tol=1e-7, abs_tol=1e-7,
                                         use_gpu=true)
        
        @test size(result.matfac.X) == (K, M_new) 
        @test size(result.matfac.Y) == (K, 30) 
        @test all(result.sample_ids .== collect(1:M_new)) 
        #@test all(result.sample_conditions .== new_sample_conditions) 

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
                                                      lambda_X_l2=0.1, 
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
    #featureset_ard_tests()
    #model_tests()
    #score_tests()
    #lbfgs_tests()
    fit_tests()
    #transform_tests()
    model_io_tests()
    #simulation_tests()

end

main()


