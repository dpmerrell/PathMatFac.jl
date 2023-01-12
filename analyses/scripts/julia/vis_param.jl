
using DictVis
using HDF5

import DictVis: is_plottable, leaf_trace, is_traversable

#@traversable(HDF5.File)
#@traversable(HDF5.Group)

DictVis.is_traversable(x::HDF5.File) = true
DictVis.is_traversable(x::HDF5.Group) = true

function DictVis.is_plottable(x::HDF5.Dataset)
    return DictVis.is_plottable(getindex(x))
end

function DictVis.leaf_trace(x::HDF5.Dataset)
    return DictVis.leaf_trace(getindex(x))
end 

function main(args)
    in_hdf = args[1]
    out_html = args[2]

    f = h5open(in_hdf, "r")
    generate_html(f, out_html)
end

main(ARGS)

