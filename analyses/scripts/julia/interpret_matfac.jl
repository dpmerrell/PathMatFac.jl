
# interpret_matfac.jl
#

using PathwayMultiomics
using CSV

include("script_util.jl")

PM = PathwayMultiomics


function main(args)

    #################################################
    ## PARSE COMMAND LINE ARGUMENTS
    #################################################   

    fitted_bson = args[1]
    out_csv = args[2]

    model = load_model(fitted_bson)
    df = interpret(model; view_names=["Mutation", "Methylation", "RNA-seq", "CNA"])
    CSV.write(out_csv, df; delim="\t")

end


main(ARGS)


