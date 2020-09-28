
import Gen: Distribution, logpdf, random, logpdf_grad, has_output_grad, has_argument_grads
using SparseArrays


################################################
# Sparse Cholesky Precision Multivariate Normal
################################################
"""
    Define a multivariate normal distribution
    parameterized by the sparse Cholesky factor
    of its (sparse) precision matrix.

    For our purposes, we can assume the distribution
    has zero mean.
"""
struct SparseCholPrecMVN <: Distribution{Vector{Float64}} end

const sparse_chol_prec_mvn = SparseCholPrecMVN()

sparse_chol_prec_mvn(chol_prec::SparseMatrixCSC{Float64,Int64}) = random(sparse_chol_prec_mvn,
				                                         chol_prec)


function random(mvn::SparseCholPrecMVN, 
		chol_prec::SparseMatrixCSC{Float64,Int64})
    
    N = size(chol_prec,1) 
    z = randn(N)

    return transpose(chol_prec) \ z 
end


L2PI = log(2.0*pi)

function logpdf(mvn::SparseCholPrecMVN, x::Vector{Float64}, 
		chol_prec::SparseMatrixCSC{Float64,Int64})

    z = transpose(x)*chol_prec
    ztz = z*transpose(z)
    
    logdet = sum(log.(abs.(diag(chol_prec))))
    
    return -0.5*(ztz + size(x,1)*L2PI) + logdet
end


function logpdf_grad(mvn::SparseCholPrecMVN, x::Vector{Float64},
		     chol_prec::SparseMatrixCSC{Float64,Int64})

    x_grad = -transpose(transpose(x)*chol_prec)
    chol_prec_grad = -(reshape(x,(size(x,1),1)) .* chol_prec) + spdiagm(0 => 1.0./diag(chol_prec) )

    return (x_grad, chol_prec_grad)

end

has_output_grad(sparse_chol_prec_mvn) = true
has_argument_grads(sparse_chol_prec_mvn) = true


###############################################
# Sparse G-Wishart Distribution
###############################################


struct SparseGWishart <: Distribution{SparseMatrixCSC{Float64,Int64}} end

const sparse_g_wishart = SparseGWishart()

sparse_g_wishart(v_matrix::SparseMatrixCSC{Float64,Int64}, 
		 dof::Int64) 
                 = random(sparse_g_wishart, v_matrix, dof)

function random(spg::SparseGWishart, v_matrix::SparseMatrixCSC{Float64,Int64}, dof::Int64)
end

function logpdf(spg::SparseGWishart, omega::SparseMatrixCSC{Float64,Int64}, 
		v_matrix::SparseMatrixCSC{Float64,Int64}, dof::Int64)
end


