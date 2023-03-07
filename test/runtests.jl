using MarginalLogDensities
using Test
using Distributions
using Optimization, OptimizationOptimJL
using ForwardDiff
using LinearAlgebra
using HCubature
using Random
using SparseDiffTools

N = 3
μ = ones(N)
σ = 1.5
d = MvNormal(μ, σ^2 * I)
ld(u, p) = logpdf(d, u)
iw = [1, 3]
iv = [2]
dmarginal = Normal(1.0, σ)dimension(mld::MarginalLogDensity) = length(mld.u)
imarginal(mld::MarginalLogDensity) = mld.iw
ijoint(mld::MarginalLogDensity) = mld.iv
nmarginal(mld::MarginalLogDensity) = length(mld.iw)
njoint(mld::MarginalLogDensity) = length(mld.iv)

"""
Splice together the estimated (fixed) parameters `v` and marginalized (random) parameters
`w` into the single parameter vector `u`, based on their indices `iv` and `iw`.
"""
function merge_parameters(v::AbstractVector{T1}, w::AbstractVector{T2}, iv, iw) where {T1,T2}
    N = length(v) + length(w)
    u = Vector{promote_type(T1, T2)}(undef, N)
    u[iv] .= v
    u[iw] .= w
    return u
end

"""
Split the vector of all parameters `u` into its estimated (fixed) components `v` and
marginalized (random) components `w`, based on their indices `iv` and `iw`.
components
"""
split_parameters(u, iv, iw) = (u[iv], u[iw])
u = randn(N)
v = u[iv]
w = u[iw]

@testset "Constructors" begin
    for forwarddiff_sparsity in [false, true]
        mld1 = MarginalLogDensity(ld, u, iw, LaplaceApprox(), forwarddiff_sparsity)
        mld2 = MarginalLogDensity(ld, u, iw, LaplaceApprox())
        mld3 = MarginalLogDensity(ld, u, iw)
        lb = -100ones(N)
        ub = 100ones(N)
        mld4 = MarginalLogDensity(ld, u, iw, Cubature(lb, ub), forwarddiff_sparsity)
        mld5 = MarginalLogDensity(ld, u, iw, Cubature(lb, ub))

        mlds = [mld1, mld2, mld3, mld4, mld5]
        for i in 1:length(mlds)-1
            for j in i+1:length(mlds)
                mldi = mlds[i]
                mldj = mlds[j]
                @test dimension(mldi) == dimension(mldj)
                @test imarginal(mldi) == imarginal(mldj)
                @test ijoint(mldi) == ijoint(mldj)
                @test nmarginal(mldi) == nmarginal(mldj)
                @test njoint(mldi) == njoint(mldj)
            end
        end
        for mld in mlds
            @test all(mld.u .== u)
            @test all(u .== merge_parameters(v, w, iv, iw))
            v1, w1 = split_parameters(mld.u, mld.iv, mld.iw)
            @test all(v1 .== v)
            @test all(w1 .== w)
        end
    end
end

@testset "Approximations" begin
    x = 1.0:3.0
    mld_laplace = MarginalLogDensity(ld, u, iw, LaplaceApprox())
    lb = fill(-100.0, 2)
    ub = fill(100.0, 2)
    mld_cubature = MarginalLogDensity(ld, u, iw, Cubature(lb, ub))
    
    @test -mld_laplace.F(x[iw], (p=(), v=x[iv])) == ld(x, ())
    prob = OptimizationProblem(mld_laplace.F, randn(2), (p=(), v=x[iv]))
    sol = solve(prob, BFGS())
    @test all(sol.u .≈ μ[iw])
    
    # analytical: against 1D Gaussian
    logpdf_true = logpdf(dmarginal, x[only(iv)])
    logpdf_laplace = mld_laplace(x[iv], ())
    logpdf_cubature = mld_cubature(x[iv], ())

    @test logpdf_laplace  ≈ logpdf_true
    @test logpdf_cubature  ≈ logpdf_true
    # test against numerical integral
    int, err = hcubature(w -> exp(ld([w[1], x[only(iv)], w[2]], ())), lb, ub)
    @test log(int) ≈ logpdf_laplace
    @test log(int) ≈ logpdf_cubature
    # # marginalized density should be higher than joint density at same point
    @test logpdf_laplace >= mld_laplace.logdensity(x, ())
    @test logpdf_cubature >= mld_cubature.logdensity(x, ())
end
