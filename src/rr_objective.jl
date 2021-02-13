
import LowRankModels: objective, get_yidxs, evaluate, calc_penalty, gemm!

export objective

function objective(glrm::RowRegGLRM, X::Array{Float64,2}, Y::Array{Float64,2},
                   XY::Array{Float64,2};
                   yidxs = get_yidxs(glrm.losses), # mapping from columns of A to columns of Y; by default, the identity
                   include_regularization=true)
    m,n = size(glrm.A)
    @assert(size(XY)==(m,yidxs[end][end]))
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,m))
    err = 0.0
    for j=1:n
        for i in glrm.observed_examples[j]
            err += evaluate(glrm.losses[j], XY[i,yidxs[j]], glrm.A[i,j])
        end
    end
    # add regularization penalty
    if include_regularization
        err += calc_penalty(glrm,X,Y; yidxs = yidxs)
    end
    return err
end

function objective(glrm::RowRegGLRM, X::Array{Float64,2}, Y::Array{Float64,2};
                   sparse=false, include_regularization=true,
                   yidxs = get_yidxs(glrm.losses), kwargs...)
    @assert(size(Y)==(glrm.k,yidxs[end][end]))
    @assert(size(X)==(glrm.k,size(glrm.A,1)))
    XY = Array{Float64}(undef, (size(X,2), size(Y,2)))
    if sparse
        # Calculate X'*Y only at observed entries of A
        m,n = size(glrm.A)
        err = 0.0
        for j=1:n
            for i in glrm.observed_examples[j]
                err += evaluate(glrm.losses[j], dot(X[:,i],Y[:,yidxs[j]]), glrm.A[i,j])
            end
        end
        if include_regularization
            err += calc_penalty(glrm,X,Y; yidxs = yidxs)
        end
        return err
    else
        # dense calculation variant (calculate XY up front)
        gemm!('T','N',1.0,X,Y,0.0,XY)
        return objective(glrm, X, Y, XY; include_regularization=include_regularization, yidxs = yidxs, kwargs...)
    end
end


objective(glrm::RowRegGLRM; kwargs...) = objective(glrm, glrm.X, glrm.Y; kwargs...)


