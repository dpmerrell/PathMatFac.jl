# dense_distributions.jl
# (c) 2020-10-01 David Merrell
#
# Naive, dense implementations of distributions
# for a model prototype.
#

using Gen
import Gen.logpdf, Gen.logpdf_grad, Gen.random
using StatsFuns
using LinearAlgebra
import Distributions: Chisq

# Multivariate Normal parameterized by precision matrix
# Cholesky factor
struct CholPrecMVN <: Gen.Distribution{Vector{Float64}} end

const chol_prec_mvn = CholPrecMVN()

function Gen.random(cpm::CholPrecMVN, chol_prec_tril::LowerTriangular{Float64,Matrix{Float64}})
    N = size(chol_prec_tril,1) 
    z = randn(N)
    return transpose(chol_prec_tril) \ z 
end

chol_prec_mvn(chol_prec::LowerTriangular{Float64, Matrix{Float64}}) = Gen.random(chol_prec_mvn, chol_prec)

L2PI = log(2.0*pi)

function logpdf(cpm::CholPrecMVN, x::Vector{Float64}, 
		chol_prec_tril::LowerTriangular{Float64, Matrix{Float64}})
    z = transpose(x)*chol_prec_tril
    ztz = z*transpose(z)
    
    logdet = sum(log.(abs.(diag(chol_prec_tril))))
    
    return -0.5*(ztz + size(x,1)*L2PI) + logdet
end


function logpdf_grad(cpm::CholPrecMVN, x::Vector{Float64}, 
		     chol_prec_tril::LowerTriangular{Float64,Matrix{Float64}})

    x_grad = -transpose(transpose(x)*chol_prec_tril)
    chol_prec_grad = -(reshape(x,(size(x,1),1)) .* chol_prec_tril) + diagm(1.0./diag(chol_prec_tril) )

    return (x_grad, chol_prec_grad)

end


Gen.has_output_grad(chol_prec_mvn) = true
Gen.has_argument_grads(chol_prec_mvn) = true


# Wishart distribution (over cholesky factors)
# parameterized by a Cholesky factor.

struct CholWishart <: Distribution{LowerTriangular{Float64,Matrix{Float64}}} end

const chol_wishart = CholWishart()

"""
    Sample a lower triangular matrix L, 
    using the "Bartlett decomposition"
"""
function Gen.random(cw::CholWishart, dof::Float64, V_chol::LowerTriangular{Float64,Matrix{Float64}})

    p = size(V_chol,1)
    A = tril(randn(p,p))
    A[diagind(A)] = [sqrt(rand(Chisq(dof-i+1))) for i=1:p]

    return LowerTriangular(V_chol * A)

end

chol_wishart(dof::Float64, V_chol::LowerTriangular{Float64,Matrix{Float64}}) = Gen.random(chol_wishart, dof, V_chol)

function logpdf(cw::CholWishart, omega_chol::LowerTriangular{Float64, Matrix{Float64}},
		dof::Float64, V_chol::LowerTriangular{Float64, Matrix{Float64}})
    # Get dimensionality
    p = size(V_chol,1)

    # log prob of outcome
    logdiag_omega_chol = log.(abs.(diag(omega_chol)))
    lp = sum(logdiag_omega_chol)*(dof - p - 1)
    lp -= sum((V_chol \ omega_chol).^2.0)*0.5
    lp -= p*dof*0.5*log(2.0)
    lp -= dof*sum(log.(abs.(diag(V_chol))))
    lp -= logmvgamma(p, 0.5*dof)

    # log inv Jacobian of cholesky transformation
    # (adjustment accounts for transformed output -- 
    # this is a distribution over cholesky factors
    lp += p*log(2.0) + dot(p:-1:1, logdiag_omega_chol)
    return lp
end


function logpdf_grad(cw::CholWishart, omega_chol::LowerTriangular{Float64,Matrix{Float64}}, 
		     dof::Float64, V_chol::LowerTriangular{Float64,Matrix{Float64}})

    p = size(V_chol,1)
    V_chol_inv = inv(V_chol)

    omega_chol_grad = diagm( ((dof-1.0):-1.0:(dof-p)) ./ diag(omega_chol) )
    omega_chol_grad += -V_chol_inv

    V_chol_grad = -sum((-V_chol_inv * tril(ones(p,p)) * V_chol_inv ) * omega_chol)
    V_chol_grad += -dof*sum(log.(abs.(diag(V_chol))))
    
    return (omega_chol_grad, nothing, V_chol_grad)
end

Gen.has_output_grad(cw::CholWishart) = true
Gen.has_argument_grads(cw::CholWishart) = (false, true)

