
using SparseArrays
using LinearAlgebra
include("src/sparse_chol_mvn.jl")
using Plots
using Gen

function cov_to_sparse_chol_prec(sparse_cov)
    prec = inv(sparse_cov)
    chol_prec = sparse(cholesky(prec).L)
    return chol_prec
end

##########################
# TEST SAMPLING METHOD
##########################

N_samples = 100

# Diagonal covariance matrix
diag_cov = diagm([1.0, 4.0])
diag_chol_prec = cov_to_sparse_chol_prec(diag_cov)

println("DIAG COVARIANCE")
println(diag_cov)

println("DIAG SPARSE CHOLESKY PRECISION")
println(diag_chol_prec)

diag_samples = [random(sparse_chol_prec_mvn,diag_chol_prec) for i=1:N_samples]
diag_samples = transpose(hcat(diag_samples...))


# Full covariance matrix
full_cov = [1.0 -0.5
	   -0.5 1.0]

full_chol_prec = cov_to_sparse_chol_prec(full_cov)
println("FULL COVARIANCE")
println(full_cov)

println("FULL SPARSE CHOLESKY PRECISION")
println(full_chol_prec)

full_samples = [random(sparse_chol_prec_mvn,full_chol_prec) for i=1:N_samples]
full_samples = transpose(hcat(full_samples...))

###################################
# Test logpdf function
###################################
x = -4.0:0.1:4.0
y = -4.0:0.1:4.0

# Diagonal covariance
contourf(x, y, (x,y)->Gen.logpdf(sparse_chol_prec_mvn, [x,y], diag_chol_prec))
scatter!(diag_samples[:,1],diag_samples[:,2], size=(400,400), xlim=(-4.0,4.0), ylim=(-4.0,4.0), plot_title="Diagonal covariance")
savefig("diag_cov.png")


# Full covariance
contourf(x, y, (x,y)->Gen.logpdf(sparse_chol_prec_mvn, [x,y], full_chol_prec))
scatter!(full_samples[:,1],full_samples[:,2], size=(400,400), xlim=(-4.0,4.0), ylim=(-4.0,4.0), plot_title="Full covariance")
savefig("full_cov.png")

ref = map((x,y)->Gen.logpdf(Gen.mvnormal,[x,y], [0.0,0.0],full_cov),x,y)
ours = map((x,y)->Gen.logpdf(sparse_chol_prec_mvn, [x,y], full_chol_prec),x,y)

diff = sum((ref .- ours).^2.0 )
println("DIFFERENCE BETWEEN LOGPDFs:")
println(ref)
println(ours)
println(diff)

##################################
# Test gradient function
##################################
