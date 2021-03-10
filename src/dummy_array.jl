
import Base: size, getindex

mutable struct DummyArray 
    size::Vector{Int}
end

size(da::DummyArray, idx::Int) = da.size[idx]

size(da::DummyArray) = tuple(da.size...)

getindex(da::DummyArray, i::Int64, j::Int64) = 0.0 
