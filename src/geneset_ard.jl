# Regularize a matrix Y under these
# probabilistic assumptions:
#
# tau_kj ~ Gamma(alpha, beta .+ A_k^T S_j)
# Y_kj ~ Normal(0, 1 / tau_kj)
# 
# I.e., we assume that the variances of the
# entries of `Y` are controlled by linear combinations
# of feature sets `S`, with weights given by `A`.
#
# The idea is to use this in an 
# Expectation-Maximization procedure.
# I.e.:  alternate between: 
#     * updating `A` given fixed tau ("maximization"); 
#     * updating `tau` given fixed `A` and `Y` ("expectation");
#     * updating `Y` given fixed tau ("maximization")
 
mutable struct FeatureSetARDReg
    alpha::Float32
    beta::Float32
    tau::AbstractMatrix
    S::AbstractMatrix
    A::AbstractMatrix
    opt::Flux.Optimise.AbstractOptimiser
end

@functor FeatureSetARDReg

function FeatureSetARDReg(K::Integer, S::AbstractMatrix;
                          alpha=1e-8, beta=1e-8, lr=0.1)
    tau = zeros(K,N)
    n_sets, N = size(S)
    A = zeros(n_sets, K)
    l1_lambda = sqrt.(sum(S; dims=2))
    opt = ISTAOptimiser(A, lr, l1_lambda)
 
    return FeatureSetARDReg(alpha, beta, zeros(K,N),
                            S, A, opt) 
end

# Apply the regularizer to a matrix
function (reg::FeatureSetARDReg)(X::AbstractMatrix)
    return 0.5*sum(reg.tau .* (X.*X))
end

# Update tau via posterior expectation.
function update_tau!(reg::FeatureSetARDReg, Y)
    reg.tau .= (alpha + 1) ./(beta .+ transpose(S)*A .+ (Y.*Y))
end

function gamma_loss(A, S, tau, alpha, beta)
    AtS = beta .+ transpose(A)*S
    return sum(-alpha.*log.(AtS) .+ (AtS).*tau)
end

# Update the matrix of "activations" via 
# nonnegative projected ISTA
function update_A!(reg::FeatureSetARDReg; max_epochs=1000, 
                                          atol=1e-6, rtol=1e-6)

    A_loss = A -> gamma_loss(A, reg.S, reg.tau, 
                                reg.alpha, reg.beta)
    total_lss = Inf
    for epoch in max_epochs
        new_lss, A_grads = Zygote.withgradient(A_loss, reg.A)
        update!(reg.opt, reg.A, A_grads[1]) 
        
        new_lss += sum(reg.lambda .* abs.(reg.A))

        loss_diff = abs(new_lss - total_lss)
        if (loss_diff < atol) | (loss_diff/new_lss < rtol)
            break
        end
        total_lss = new_lss
    end
end


