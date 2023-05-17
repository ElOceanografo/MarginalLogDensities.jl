using MarginalLogDensities
using Test
using Distributions
using Optim
using ForwardDiff
using LinearAlgebra
using HCubature
using Random

N = 3
σ = 1.5
d = MvNormal(zeros(N), σ*I)
logdensity(x) = logpdf(d, x)
im = [1, 3]
ij = [2]
dmarginal = MvNormal(zeros(length(ij)), σ*I)

@testset "Constructors" begin
    for forwarddiff_sparsity in [false, true]
        mld2 = MarginalLogDensity(logdensity, N, im)
        mld3 = MarginalLogDensity(logdensity, N, im, LaplaceApprox(), forwarddiff_sparsity)
        mld4 = MarginalLogDensity(logdensity, N, im,
            Cubature(-100ones(N), 100ones(N)))

        for mld in [mld2, mld3, mld4]
            @test dimension(mld) == N
            @test imarginal(mld) == im
            @test ijoint(mld) == ij
            @test nmarginal(mld) == length(imarginal(mld))
            @test njoint(mld) == length(ijoint(mld))
            @test njoint(mld) + nmarginal(mld) == dimension(mld)
            @test isempty(setdiff(1:N, union(im, ij)))
        end
    end
end

# @testset "Sparse Hessians" begin
#     f(x) = -logdensity(x)
#     hconf = HessianConfig(f, im, ij)    
#     H = zeros(length(im), length(im))


# end

@testset "Approximations; using SparsityDetection" begin
    x = 1.0:3.0
    mld_laplace = MarginalLogDensity(logdensity, N, im, LaplaceApprox(), false)
    mld_cubature = MarginalLogDensity(logdensity, N, im,
        Cubature(-100ones(2), 100ones(2)))

    @test mld_laplace(x[im], x[ij]) == logdensity(x)
    @test mld_cubature(x[im], x[ij]) == logdensity(x)
    # analytical: against 1D Gaussian
    @test mld_laplace(x[ij]) ≈ logpdf(dmarginal, x[ij])
    @test mld_cubature(x[ij]) ≈ logpdf(dmarginal, x[ij])
    # test against numerical integral
    int, err = hcubature(x -> exp(logdensity([x[1], 2, x[2]])),
        -100*ones(2), 100*ones(2))
    @test log(int) ≈ mld_laplace(x[ij])
    @test log(int) ≈ mld_cubature(x[ij])
    # marginalized density should be higher than joint density at same point
    @test mld_laplace(x[ij]) >= mld_laplace(x[im], x[ij])
    @test mld_cubature(x[ij]) >= mld_cubature(x[im], x[ij])
end

@testset "Approximations; using ForwardDiff" begin
    x = 1.0:3.0
    mld_laplace = MarginalLogDensity(logdensity, N, im, LaplaceApprox(), true)
    mld_cubature = MarginalLogDensity(logdensity, N, im,
        Cubature(-100ones(2), 100ones(2)))

    @test mld_laplace(x[im], x[ij]) == logdensity(x)
    @test mld_cubature(x[im], x[ij]) == logdensity(x)
    # analytical: against 1D Gaussian
    @test mld_laplace(x[ij]) ≈ logpdf(dmarginal, x[ij])
    @test mld_cubature(x[ij]) ≈ logpdf(dmarginal, x[ij])
    # test against numerical integral
    int, err = hcubature(x -> exp(logdensity([x[1], 2, x[2]])),
        -100*ones(2), 100*ones(2))
    @test log(int) ≈ mld_laplace(x[ij])
    @test log(int) ≈ mld_cubature(x[ij])
    # marginalized density should be higher than joint density at same point
    @test mld_laplace(x[ij]) >= mld_laplace(x[im], x[ij])
    @test mld_cubature(x[ij]) >= mld_cubature(x[im], x[ij])
end
