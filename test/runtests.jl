using LaplaceApproximation
using Test
using Distributions
using Optim
using ForwardDiff
using LinearAlgebra
using HCubature
using Random

function make_test_problem(N)
    rng = Random.MersenneTwister(1)
    μ = randn(rng, N)
    Σ = randn(rng, N, N); Σ = Σ'Σ + I
    d = MvNormal(μ, Σ)
    x = rand(rng, d)
    return μ, Σ, d, x
end

# @testset "LaplaceApproximation.jl" begin

N = 4
μ, Σ, d, x = make_test_problem(N)

logdensity(μ) = logpdf(MvNormal(μ, Σ), x)
im = [1, 3]
ij = [2, 4]

mld1 = MarginalLogDensity(logdensity, 4, im, ij, LaplaceApprox())
mld2 = MarginalLogDensity(logdensity, 4, im)
mld3 = MarginalLogDensity(logdensity, 4, im, LaplaceApprox())
mld4 = MarginalLogDensity(logdensity, 4, im, Cubature())

for mld in [mld2, mld3, mld4]
    @test dimension(mld) == dimension(mld1) == N
    @test imarginal(mld) == imarginal(mld1) == im
    @test ijoint(mld) == ijoint(mld1) == ij
    @test nmarginal(mld) == nmarginal(mld1) == length(imarginal(mld))
    @test njoint(mld) == njoint(mld1) == length(ijoint(mld))
    @test njoint(mld) + nmarginal(mld) == dimension(mld)
    @test isempty(setdiff(1:N, union(im, ij)))
end

@test mld1(μ[im], μ[ij]) == logdensity(μ)
mld1(μ[im], μ[ij])
mld1(μ[im])




#
#
# N = 2
# μ, Σ, d, x = make_test_problem(N)
# logdensity(μ) = logpdf(MvNormal(μ, Σ), x)
#
# using Plots, StatsPlots
# using QuadGK
# xx = -0:0.1:6
# yy = -5:0.1:5
# contour(xx, yy, (x, y) -> exp(logdensity([x, y])))
#
# dx, _ = quadgk(y -> exp(logdensity([2, y])), -Inf, Inf)
#
# opt = optimize(y -> -logdensity([2, y]), -10, 10)
# opt.minimizer
# vline!([2])
# hline!([opt.minimizer])
#
# d1 = y1 -> ForwardDiff.derivative(y -> logdensity([2, y]), y1)
# d1(opt.minimizer)
# H = ForwardDiff.derivative(d1, opt.minimizer)
#
# logdensity([2, opt.minimizer])
# exp(logdensity([2, opt.minimizer]))
# lap_app = exp(logdensity([2, opt.minimizer])) * sqrt((2π) / -H)
# dx ≈ lap_app
#
# plot(Normal(x[1], sqrt(Σ[1])))
# plot!(x -> quadgk(y -> exp(logdensity([x, y])), -20, 20)[1], -1, 8)



# end
