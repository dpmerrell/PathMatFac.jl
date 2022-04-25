

mutable struct NetworkRegularizer

    AA::Tuple # Tuple of K matrices encoding relationships 
              # beween *observed* features

    AB::Tuple # Tuple of K matrices encoding relationships
              # between *observed and unobserved* features

    BB::Tuple # Tuple of K matrices encoding relationships
              # between *unobserved* features

    B_matrix::AbstractMatrix # a (K x N_unobserved) matrix
                             # containing estimates of 
                             # unobserved quantities
end


function NetworkRegularizer(edgelists; ordering=nothing, 
                                       weight=1.0, 
                                       unobserved=nothing)

end


quadratic(u::AbstractVector, 
          X::AbstractMatrix, 
          v::AbstractVector) = 0.5*dot(u, (X*v))

quadratic(X::AbstractMatrix, v::AbstractVector) = 0.5*dot(v, (X*v))


#################################################
# Matrix row-regularizer
#################################################

function (nr::NetworkRegularizer)(X::AbstractMatrix)

    loss = 0.0
    K = size(X,1)
    for k=1:K
        loss += quadratic(nr.AA[k], X[k,:])
        loss += 2*quadratic(X[k,:], nr.AB[k], nr.B_matrix[k,:])
        loss += quadratic(nr.BB[k], nr.B_matrix[k,:])
    end
    return loss
end


function ChainRules.rrule(nr::NetworkRegularizer, X::AbstractMatrix)

    loss = 0.0

    function netreg_mat_pullback(loss_bar)

        B_bar = TODO
        nr_bar = Tangent{NetworkRegularizer}(B_matrix=B_bar)

        return nr_bar, X_bar
    end

    return loss, netreg_mat_pullback
end


#################################################
# Vector regularizer
#################################################

function (nr::NetworkRegularizer)(x::AbstractVector)

    loss = 0.0
    loss += quadratic(nr.AA[1], x)
    loss += 2*quadratic(x, nr.AB[1], nr.B_matrix[1,:])
    loss += quadratic(nr.BB[1], nr.B_matrix[1,:])
    return loss
end


function ChainRules.rrule(nor::NetworkRegularizer, X::AbstractVector)
    loss = 0.0

    return loss, netreg_vec_pullback
end



