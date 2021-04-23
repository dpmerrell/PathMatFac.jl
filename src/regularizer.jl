

function reg_eval(X, idx, reg_mats)
    s = 0.0
    for k=1:size(X,1)
        s += X[k, idx] * dot(view(reg_mats[k], :, idx), view(X, k, :))
    end
    return 0.5*s
end


function reg_grad(X, idx, reg_mats)
    g = zeros(size(X,1))
    for k=1:size(X,1)
        g[k] = X[k, idx]*reg_mats[k][idx,idx] + dot(view(reg_mats[k], :, idx), view(X, k, :))
    end
    return 0.5 .* g
end


#####################################
# CUDA functions

function compute_reg_loss(X::CuArray{Float32,2}, 
                          reg_mats::AbstractVector)
    s = 0.0
    for i=1:length(reg_mats)
        s += 0.5 * dot(X[i,:], reg_mats[i]*X[i,:])
    end
    return s
end


function add_reg_grad!(grad_X::CuArray{Float32,2}, 
                       X::CuArray{Float32,2}, 
                       reg_mats::AbstractVector)
    for i=1:min(length(reg_mats), size(grad_X,1))
        grad_X[i,:] .+= (reg_mats[i]*X[i,:])
    end
    return nothing
end


