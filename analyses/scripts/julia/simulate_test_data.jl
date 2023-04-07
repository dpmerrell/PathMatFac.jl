
include("script_util.jl")


function generate_Z(K,M,N)
    inv_sqrt_K = 1/sqrt(K)
    X = randn(K,M) .* inv_sqrt_K
    Y = randn(K,N*2) .* inv_sqrt_K

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
    #rna_view .= 10.0 .+ 4.0.*randn(M,N)
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

    output_hdf = args[1]
    if length(args) > 1
        N = parse(Int, args[2])
    end

    Z = generate_Z(K,M,N)
    fgs = generate_feature_genes(N)
    fas = generate_feature_assays(N)
    sample_ids = generate_sample_ids(M)
    sample_conditions = generate_sample_conditions(M)
    barcodes = generate_barcodes(M)

    h5write(output_hdf, "omic_data/data", Z)
    h5write(output_hdf, "omic_data/feature_genes", fgs)
    h5write(output_hdf, "omic_data/feature_assays", fas)
    h5write(output_hdf, "omic_data/instances", sample_ids)
    h5write(output_hdf, "omic_data/instance_groups", sample_conditions)
    h5write(output_hdf, "barcodes/data", barcodes)
    h5write(output_hdf, "barcodes/instances", sample_ids)
    h5write(output_hdf, "barcodes/features", ["cna","mrnaseq"])
    
end


main(ARGS)

