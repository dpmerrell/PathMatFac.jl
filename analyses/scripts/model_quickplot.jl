
using DictVis
using PathwayMultiomics, MatFac

import DictVis: is_plottable, leaf_trace, is_traversable

@traversable_fields PathwayMultiomics.MultiomicModel
@traversable_fields MatFac.MatFacModel
@traversable_fields PathwayMultiomics.PMLayers
@traversable_fields PathwayMultiomics.ColShift
@traversable_fields PathwayMultiomics.ColScale
@traversable_fields PathwayMultiomics.BatchScale
@traversable_fields PathwayMultiomics.BatchShift
@traversable_fields PathwayMultiomics.BatchArray


function main(args)
    in_model_hdf = args[1]
    out_html = args[2]

    model = PathwayMultiomics.load_model(in_model_hdf)
    generate_html(model, out_html)
end

main(ARGS)

