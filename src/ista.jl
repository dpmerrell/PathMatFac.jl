

function proj!(X; alpha=0.01)
    X .= max.(abs.(X) .- alpha, 0.0).*sign.(X)
end


"""
    ista!()

    Iterative Shrinkage-Thresholding Algorithm.
    Minimize an L1-penalized loss function.
    
"""
function ista!(X::AbstractArray, func; lr=0.1, lambda=1.0, atol=1e-6, rtol=1e-6)

    lss = Inf

    # Maintain a simple Adagrad optimizer. 
    ssq_grads = zero(X) .+ 1e-8

    for iter in 1:max_iter

        new_lss, X_grad = Zygote.withgradient(func, X)
        X_grad = X_grad[1]

        ssq_grads .+= (X_grad.*X_grad)
        eta = lr ./ sqrt.(ssq_grads)
        proj!(X; alpha=lambda.*eta)
 
        loss_diff = new_lss - lss
        if (abs(loss_diff) < abs_tol | (abs(loss_diff/new_lss) < rel_tol))
            break
        end
    end

end




