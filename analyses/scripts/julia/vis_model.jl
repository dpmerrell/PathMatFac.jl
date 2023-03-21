
using DictVis
using PathwayMultiomics, MatFac

PM = PathwayMultiomics
MF = MatFac

import DictVis: is_plottable, leaf_trace, is_traversable


@traversable_fields PM.PathMatFacModel
@traversable_fields MF.MatFacModel
@traversable_fields PM.ViewableComposition
@traversable_fields PM.ColScale
@traversable_fields PM.BatchScale
@traversable_fields PM.ColShift
@traversable_fields PM.BatchShift
@traversable_fields PM.BatchArray

@traversable_fields PM.SequenceReg
@traversable_fields PM.ColParamReg
@traversable_fields PM.GroupRegularizer
@traversable_fields PM.CompositeRegularizer
@traversable_fields PM.BatchArrayReg

@plottable NTuple{K,Number} where K
function leaf_trace(leaf::NTuple{K,Number} where K)
    return leaf_trace(collect(leaf))
end

function main(args)
    in_model_hdf = args[1]
    out_html = args[2]

    model = PathwayMultiomics.load_model(in_model_hdf)
    generate_html(model, out_html)
end

main(ARGS)

