

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


