
import LowRankModels: evaluate, prox, prox!

import LinearAlgebra: dot

export RowReg


mutable struct RowReg <: Regularizer
    b::Vector{Float64}
    neighbors::Vector{Tuple{Int64,Int64,Float64}}
    scale::Float64
end


function prox(r::RowReg,u::AbstractArray,alpha::Number)
    #return 1/(1+2*alpha*r.scale)*(u .- b)
    return 1/(1+2*alpha*r.scale)*u 
end

function prox!(r::RowReg,u::Array{Float64},alpha::Number)
    #axpy!(-1, b, u)
    rmul!(u, 1/(1+2*alpha*r.scale))
end


evaluate(r::RowReg,a::AbstractArray) = r.scale*sum(abs2, a) + dot(r.b,a)


# Update the regularizer's b vector
function update_b!(r::RowReg, factor_col_views::Vector)

    # reset b to zeros
    r.b .= 0.0

    # b is a weighted sum of neighboring factors
    for (idx, dim, w) in r.neighbors
        r.b[dim] += w*factor_col_views[idx][dim]
    end 

end


