
using DictVis
using PathwayMultiomics, MatFac

import DictVis: is_plottable, leaf_trace, is_traversable

@traversable_fields PathwayMultiomics.MultiomicModel
@traversable_fields MatFac.MatFacModel

function main(args)
    in_model_hdf = args[1]
    out_html = args[2]

    model = PathwayMultiomics.load_model(in_model_hdf)
    generate_html(model, out_html)
end

main(ARGS)

