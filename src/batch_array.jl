
import Base: view, zero, exp, log, deepcopy
import Flux: gpu, trainable

mutable struct BatchArray
    col_ranges::Tuple # UnitRanges
    col_range_ids::Vector  # Names of the column ranges
    row_idx::UnitRange # Stores "view-like" information
    row_batches::Tuple # boolean matrices;
                       # each column is an indicator
                       # vector for a batch. Should
                       # balance space-efficiency and performance
    row_batch_ids::Tuple  # Names of row batches, for each column range
    values::Tuple # vectors of numbers
end

@functor BatchArray

Flux.trainable(ba::BatchArray) = (values=ba.values,)

function BatchArray(col_batch_ids::Vector, row_batch_dict::AbstractDict, 
                    value_dicts::Vector{<:AbstractDict})

    unq_cbi = unique(col_batch_ids)
    row_batch_ids = [row_batch_dict[b] for b in unq_cbi]

    n_rows = length(row_batch_ids[1])
    
    col_ranges = ids_to_ranges(col_batch_ids)
    col_range_ids = unique(col_ranges)

    row_batches = [ids_to_ind_mat(rbv) for rbv in row_batch_ids]
    unq_row_batch_ids = [unique(rbv) for rbv in row_batch_ids]

    values = [[vd[ub] for ub in unique(rbi)] for (vd,rbi) in zip(value_dicts,row_batch_ids)]

    return BatchArray(Tuple(col_ranges),
                      col_range_ids, 
                      1:n_rows,
                      Tuple(row_batches), 
                      Tuple(unq_row_batch_ids),
                      Tuple(values))
end


function view(ba::BatchArray, idx1, idx2::UnitRange)

    new_col_ranges, r_min, r_max = subset_ranges(ba.col_ranges, idx2)
    new_col_ranges = shift_range.(new_col_ranges, 1 - new_col_ranges[1].start)
    new_col_range_ids = ba.col_range_ids[r_min:r_max]

    new_row_idx = ba.row_idx[idx1]

    new_row_batches = ba.row_batches[r_min:r_max]
    new_row_batch_ids = ba.row_batch_ids[r_min:r_max]
    new_values = ba.values[r_min:r_max]

    return BatchArray(new_col_ranges, new_col_range_ids, new_row_idx, 
                      new_row_batches, new_row_batch_ids,
                      new_values)
end


function zero(ba::BatchArray)
    cvalues = map(zero, ba.values)
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.col_range_ids),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches), 
                      deepcopy(ba.row_batch_ids),
                      cvalues) 
end

#########################################
# Addition
function Base.:(+)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    col_buffer = similar(A, size(A,1))

    for (j,cr) in enumerate(B.col_ranges)
        col_buffer .= view(B.row_batches[j], B.row_idx, :) * B.values[j]
        view(result, :, cr) .+= col_buffer
    end

    col_buffer = nothing
    return result
end


function ChainRulesCore.rrule(::typeof(+), A::AbstractMatrix, B::BatchArray)
    
    result = A + B
    
    function ba_plus_pullback(result_bar)
        A_bar = copy(result_bar) # Just a copy of the result tangent 
        
        values_bar = map(zero, B.values) # Just sum the result tangents corresponding
                                         # to each value of B
        for (j, cbr) in enumerate(B.col_ranges)
            values_bar[j] .= vec(sum(transpose(view(B.row_batches[j], B.row_idx, :)) * view(result_bar,:,cbr); dims=2)) 
        end
        B_bar = Tangent{BatchArray}(values=values_bar)
        return ChainRulesCore.NoTangent(), A_bar, B_bar 
    end

    return result, ba_plus_pullback
end

####################################################
# Subtraction
function Base.:(-)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    for (j,cbr) in enumerate(B.col_ranges)
        view(result, :, cbr) .-= (view(B.row_batches[j], B.row_idx, :)*B.values[j])
    end
    return result
end

# (between two batch arrays with identical layouts)
function Base.:(-)(A::BatchArray, B::BatchArray)
    result = deepcopy(A)
    for j=1:length(B.values)
        result.values[j] .= A.values[j] .- B.values[j]
    end
    return result
end

#########################################
# Multiplication
function Base.:(*)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    for (j,cbr) in enumerate(B.col_ranges)
        view(result, :, cbr) .*= (view(B.row_batches[j], B.row_idx, :)*B.values[j])
    end
    return result
end


function ChainRulesCore.rrule(::typeof(*), A::AbstractMatrix, B::BatchArray)
    
    result = A * B
    
    function ba_mult_pullback(result_bar)
        A_bar = (result_bar .+ zero(A)) * B 
        values_bar = map(zero, B.values)    

        for (j, cbr) in enumerate(B.col_ranges)
            view(A, :, cbr) .*= view(result_bar, :, cbr)
            values_bar[j] .= vec(sum(transpose(view(B.row_batches[j], B.row_idx, :)) * view(A, :, cbr); dims=2)) 
        end
        B_bar = Tangent{BatchArray}(values=values_bar)
        return ChainRulesCore.NoTangent(), A_bar, B_bar 
    end

    return result, ba_mult_pullback
end


#########################################
# Division
function Base.:(/)(A::AbstractMatrix, B::BatchArray)

    result = copy(A)
    for (j,cbr) in enumerate(B.col_ranges)
        view(result, :, cbr) ./= (view(B.row_batches[j], B.row_idx, :)*B.values[j])
    end
    return result
end


#########################################
# Exponentiation
function exp(ba::BatchArray)
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.col_range_ids),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches),
                      deepcopy(ba.row_batch_ids),
                      map(v->exp.(v), ba.values))
end


function ChainRulesCore.rrule(::typeof(exp), ba::BatchArray)
    Z = exp(ba)

    function ba_exp_pullback(Z_bar)
        values_bar = map(.*, Z_bar.values, Z.values)
        return ChainRulesCore.NoTangent(),
               Tangent{BatchArray}(values=Tuple(values_bar))
    end

    return Z, ba_exp_pullback
end

##########################################
# Logarithm
function log(ba::BatchArray)
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.col_range_ids),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches),
                      deepcopy(ba.row_batch_ids),
                      map(v->log.(v), ba.values))
end

###########################################
# Deepcopy
function deepcopy(ba::BatchArray)
    return BatchArray(deepcopy(ba.col_ranges),
                      deepcopy(ba.col_range_ids),
                      deepcopy(ba.row_idx),
                      deepcopy(ba.row_batches),
                      deepcopy(ba.row_batch_ids),
                      deepcopy(ba.values))
end




