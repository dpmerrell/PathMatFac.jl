
include("script_util.jl")


function generate_S(L, N, configuration; p=0.05)
    S = rand(L, N)
    S .= (S .< p)
    S_new = similar(S, L, 2*N)
    S_new[:,1:N] .= S
    S_new[:, (N+1):end] .= S
    return S_new
end

function generate_genesets(L, N, configuration; p=0.05)
    if configuration == "fsard"
        S = generate_S(L, N, configuration; p=p)
        pwys = []
        return S, pwys
    else
        return nothing, nothing
    end
end

function generate_Y_fsard(K, N; genesets=nothing)
    
end



function generate_Y(K,N; configuration="ard", genesets=nothing)
    
    if configuration=="fsard"
        Y = generate_Y_fsard(K, N; genesets=genesets)
    else
        inv_sqrt_K = 1/sqrt(K)
        Y = randn(K,N*2) .* inv_sqrt_K
    end
end




function generate_Z(K,M,N; configuration="ard", genesets=nothing)
    inv_sqrt_K = 1/sqrt(K)
    X = randn(K,M) .* inv_sqrt_K
    Y = generate_Y(K, N; configuration=configuration,
                         genesets=genesets)

    Z = transpose(X) * Y

    # Mutation data
    mut_view = view(Z, :, 1:N)
    mut_view .-= 1.1
    mut_view[mut_view .> 0] .= 1
    mut_view[mut_view .<= 0] .= 0

    # Add a lot of NaN rows to the mutation data
    mut_view[3*div(M,4):end, :] .= NaN

    # RNAseq data
    rna_view = view(Z, :, (N+1):(2*N))
    mid_M = div(M,2)
    
    # Column scales
    rna_view .*= 4.0
    # Batch scales
    rna_view[1:mid_M,:] .*= 1.33
    rna_view[(mid_M+1):end,:] .*= 0.75

    # Column shift
    rna_view .+= 10.0
    # Batch shift
    rna_view[1:mid_M,:] .-= 1.0
    rna_view[(mid_M+1):end,:] .+= 1.0

    return Z
end

function generate_feature_genes(N)
    return repeat(map(i->string("gene_",i), collect(1:N)), outer=2)
end

function generate_feature_assays(N)
    return vcat(fill("mutation", N), fill("mrnaseq",N))
end

function generate_sample_conditions(M)
    return fill("condition", M)
end

function generate_sample_ids(M)
    return map(i->string("sample_",i), collect(1:M))
end

function generate_barcodes(M)
    return repeat(["barcode-1", "barcode-2"], inner=(div(M,2), 2))
end


function main(args)

    K = 2
    M = 1000
    N = 500 
    L = 100
    configuration = "ard"
    genesets = nothing
    out_json = nothing

    output_hdf = args[1]
    if length(args) > 1
        K = parse(Int, args[2])
        L = parse(Int, args[3])
        M = parse(Int, args[4])
        N = parse(Int, args[5])
        configuration = args[6]
    end


    fgs = generate_feature_genes(N)
    fas = generate_feature_assays(N)
    S, pwys = generate_genesets(L, N, configuration)
    
    sample_ids = generate_sample_ids(M)
    sample_conditions = generate_sample_conditions(M)
    barcodes = generate_barcodes(M)

    Z = generate_Z(K,M,N; configuration=configuration, 
                          S=S)

    h5write(output_hdf, "omic_data/data", Z)
    h5write(output_hdf, "omic_data/feature_genes", fgs)
    h5write(output_hdf, "omic_data/feature_assays", fas)
    h5write(output_hdf, "omic_data/instances", sample_ids)
    h5write(output_hdf, "omic_data/instance_groups", sample_conditions)
    h5write(output_hdf, "barcodes/data", barcodes)
    h5write(output_hdf, "barcodes/instances", sample_ids)
    h5write(output_hdf, "barcodes/features", ["cna","mrnaseq"])
   
    if out_json != nothing
        open(out_json, "w") do f
            JSON.print(f, pwys)
        end
    end
 
end


main(ARGS)

