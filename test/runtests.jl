using LaplaceApproximation
using Test
using Distributions
using Optim
using ForwardDiff
using LinearAlgebra
using HCubature
using Random

@testset "LaplaceApproximation.jl" begin
    # d1 = Gamma(3, 4)
    # d2 = Normal(3, 0.8)
    # f(x) = pdf(d1, x[1]) * pdf(d2, x[2])
    # f(x, y) = f([x, y])
    # xx = 0:0.1:30
    # yy = -2:0.1:7

    μ = [1.0, 2.0]
    Σ = [1 0.5; 0.5 1]
    N = length(μ)
    m = 5.0
    d = MvNormal(μ, Σ)
    f(x) = m * pdf(d, x)
    logf(x) = log(m) + logpdf(d, x)
    # xx = -2:0.1:4
    # yy = -2:0.1:5
    # contour(xx, yy, f)

    la = fitlaplace(logf, zeros(N))
    tol = 1e-6
    @test all((mode(la) .- μ) ./ μ .< tol)
    @test normalizer(la) ≈ m
    @test hcubature(x -> f(x)/normalizer(la), [-10, -10], [10, 10])[1] ≈ 1

    Random.seed!(1)
    testpoints = [rand(d) for _ in 1:10]
    @test all([logapprox(la, x) ≈ logpdf(d, x) for x in testpoints])
    @test all([approx(la, x) ≈ pdf(d, x) for x in testpoints])

end
