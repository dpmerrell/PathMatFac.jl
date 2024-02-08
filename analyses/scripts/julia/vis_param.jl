

using DictVis
using PathMatFac, MatFac
using PlotlyJS
using PlotlyBase

PM = PathMatFac
MF = MatFac

import DictVis: is_plottable, leaf_trace, is_traversable


@traversable_fields MF.MatFacModel
@traversable_fields MF.CompositeNoise
@traversable_fields MF.OrdinalNoise
@traversable_fields MF.NormalNoise
@traversable_fields MF.SquaredHingeNoise
@traversable_fields MF.OrdinalSqHingeNoise

@traversable_fields PM.PathMatFacModel
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
@traversable_fields PM.ARDRegularizer
@traversable_fields PM.FeatureSetARDReg

#DictVis.is_plottable(m::SparseMatrixCSC) = false


@plottable NTuple{K,Number} where K
function leaf_trace(leaf::NTuple{K,Number} where K)
    return leaf_trace(collect(leaf))
end

MAX_HEATMAP_SIZE = DictVis.MAX_HEATMAP_SIZE*100

function leaf_trace(leaf::AbstractMatrix{<:Real})

    M, N = size(leaf)
    total_size = M*N

    if total_size > MAX_HEATMAP_SIZE
        shrinkage = ceil(sqrt(total_size/MAX_HEATMAP_SIZE))
        leaf = DictVis.downsample(leaf, shrinkage)
    end

    trace = heatmap(z=float.(leaf), type="heatmap", colorscale="Greys", reversescale=true)
    return trace
end

function main(args)
    in_model_hdf = args[1]
    out_html = args[2]

    model = PathMatFac.load_model(in_model_hdf)
    generate_html(model, out_html)
end

main(ARGS)
