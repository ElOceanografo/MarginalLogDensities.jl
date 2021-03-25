using MarginalLogDensities
using Test
using Distributions
using Optim
using ForwardDiff
using LinearAlgebra
using HCubature
using Random

N = 3
logdensity(x) = logpdf(MvNormal(zeros(N), 1.5), x)
im = [1, 3]
ij = [2]
mld = MarginalLogDensity(logdensity, N, im)

@testset "Constructors" begin
    mld1 = MarginalLogDensity(logdensity, N, im, ij, LaplaceApprox())
    mld2 = MarginalLogDensity(logdensity, N, im)
    mld3 = MarginalLogDensity(logdensity, N, im, LaplaceApprox())
    mld4 = MarginalLogDensity(logdensity, N, im,
        Cubature(-100ones(N), 100ones(N)))

    for mld in [mld2, mld3, mld4]
        @test dimension(mld) == dimension(mld1) == N
        @test imarginal(mld) == imarginal(mld1) == im
        @test ijoint(mld) == ijoint(mld1) == ij
        @test nmarginal(mld) == nmarginal(mld1) == length(imarginal(mld))
        @test njoint(mld) == njoint(mld1) == length(ijoint(mld))
        @test njoint(mld) + nmarginal(mld) == dimension(mld)
        @test isempty(setdiff(1:N, union(im, ij)))
    end
end

@testset "Approximations" begin
    x = 1:3
    mld_laplace = MarginalLogDensity(logdensity, N, im, LaplaceApprox())
    mld_cubature = MarginalLogDensity(logdensity, N, im,
        Cubature(-100ones(2), 100ones(2)))

    @test mld_laplace(x[im], x[ij]) == logdensity(x)
    @test mld_cubature(x[im], x[ij]) == logdensity(x)
    # analytical: against 1D Gaussian
    @test mld_laplace(x[ij]) ≈ logpdf(Normal(0, 1.5), 2)
    @test mld_cubature(x[ij]) ≈ logpdf(Normal(0, 1.5), 2)
    # test against numerical integral
    int, err = hcubature(x -> exp(logdensity([x[1], 2, x[2]])),
        -100*ones(2), 100*ones(2))
    @test log(int) ≈ mld_laplace(x[ij])
    @test log(int) ≈ mld_cubature(x[ij])
end
