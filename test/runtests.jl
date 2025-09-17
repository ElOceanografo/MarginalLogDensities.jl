using MarginalLogDensities
using Test
using Distributions
using ComponentArrays
using Optimization, OptimizationOptimJL
using FiniteDiff, ForwardDiff, ReverseDiff, Zygote#, Mooncake
using LinearAlgebra, SparseArrays
using HCubature
using StableRNGs
using ChainRulesTestUtils
import LogDensityProblems

rng = StableRNG(15)

N = 3
μ = ones(N)
σ = 1.5
d = MvNormal(μ, σ^2 * I)
ld(u, p) = logpdf(d, u)
iw = [1, 3]
iv = [2]
dmarginal = Normal(1.0, σ)
u = randn(rng, N)
v = u[iv]
w = u[iw]
u_component = ComponentArray(v = v, w = w)

@testset "Constructors" begin
    adtype = AutoForwardDiff()
    hess_adtype = AutoZygote()
    @testset "Vector u" begin
        mld1 = MarginalLogDensity(ld, u, iw, (), LaplaceApprox(adtype=adtype),
            hess_adtype=hess_adtype)
        mld2 = MarginalLogDensity(ld, u, iw, (), LaplaceApprox(adtype=adtype))
        mld3 = MarginalLogDensity(ld, u, iw, ())
        mld4 = MarginalLogDensity(ld, u, iw)

        @test mld1.hess_adtype != mld2.hess_adtype
        @test mld2.hess_adtype isa AutoSparse
        @test mld2.method.adtype == mld2.hess_adtype.dense_ad.inner

        mlds = [mld1, mld2, mld3, mld4]
        for i in 1:length(mlds)-1
            for j in i+1:length(mlds)
                mldi = mlds[i]
                mldj = mlds[j]
                @test imarginal(mldi) == imarginal(mldj)
                @test ijoint(mldi) == ijoint(mldj)
                @test nfull(mldi) == nfull(mldj)
                @test nmarginal(mldi) == nmarginal(mldj)
                @test njoint(mldi) == njoint(mldj)
            end
        end
        for mld in mlds
            @test all(mld.u .== u)
            @test all(u .== merge_parameters(v, w, iv, iw, u))
            v1, w1 = split_parameters(mld.u, mld.iv, mld.iw)
            @test all(v1 .== v)
            @test all(w1 .== w)
        end

        @testset "ComponentVector u" begin
            u_vector = Vector(u_component)
            iw_symbol = [:w]
            iw_indices = label2index(u_component, :w)
            
            mld1 = MarginalLogDensity(ld, u_component, iw_symbol)
            mld2 = MarginalLogDensity(ld, u_vector, iw_indices)
            @test nfull(mld1) == nfull(mld2)
            @test all(mld1.u[iw_symbol] .== mld2.u[iw_indices])
            
            @test all(mld1.u .== u_vector)
            @test all(u .== merge_parameters(v, w, iv, iw, u_component))
            v1, w1 = split_parameters(mld1.u, mld1.iv, mld1.iw)
            v2, w2 = split_parameters(mld2.u, mld2.iv, mld2.iw)
            @test all(v1 .== v2)
            @test all(w1 .== w2)
        end
    end
    
    @testset "Marginalizers" begin
        adtype = AutoForwardDiff()
        solver = BFGS()
        @test_nowarn LaplaceApprox()
        @test_nowarn LaplaceApprox(solver)
        @test_nowarn LaplaceApprox(solver, adtype=adtype, grad=nothing, hess=nothing)

        @test_nowarn Cubature()
        @test_nowarn Cubature(solver=solver)
        @test_nowarn Cubature(solver=solver, adtype=adtype)
        @test_nowarn Cubature(solver=solver, adtype=adtype, grad=nothing, hess=nothing)
        @test_nowarn Cubature(upper = fill(-1, 5), lower=fill(1, 5))
        @test_nowarn Cubature(upper = fill(-1, 5), lower=fill(1, 5), solver=solver)
    end
end

@testset "Custom ChainRules" begin
    v = fill(1.0, 3)
    w = fill(2.0, 4)
    iv = 1:3
    iw = 4:7
    u = zeros(length(v) + length(w))
    u[iv] .= v
    u[iw] .= w
    test_rrule(merge_parameters, v, w, iv, iw, u)
end

@testset "LogDensityProblem interface" begin
    mld = MarginalLogDensity(ld, u, iw)
    v = ones(njoint(mld))
    @test LogDensityProblems.dimension(mld) == njoint(mld)
    @test LogDensityProblems.logdensity(mld, v) == mld(v)
    @test LogDensityProblems.capabilities(mld) == LogDensityProblems.LogDensityOrder{0}()
end

@testset "Dense approximations" begin
    x = 1.0:3.0
    x_component = ComponentVector(v = x[iv], w = x[iw])
    mld_laplace = MarginalLogDensity(ld, u, iw, (), LaplaceApprox())
    mld_laplace_component = MarginalLogDensity(ld, u_component, [:w], (), LaplaceApprox())
    lb = fill(-100.0, 2)
    ub = fill(100.0, 2)
    mld_cubature1 = MarginalLogDensity(ld, u, iw, (), Cubature(lower=lb, upper=ub))
    mld_cubature2 = MarginalLogDensity(ld, u, iw, (), Cubature())
    
    @test -mld_laplace.f_opt(x[iw], (p=(), v=x[iv])) == ld(x, ())
    prob = OptimizationProblem(mld_laplace.f_opt, randn(rng, 2), (p=(), v=x[iv]))
    sol = solve(prob, BFGS())
    @test all(sol.u .≈ μ[iw])

    # analytical: against 1D Gaussian
    logpdf_true = logpdf(dmarginal, x[only(iv)])
    @test x[iv] == x_component.v
    logpdf_laplace = mld_laplace(x[iv], ())
    logpdf_laplace_component = mld_laplace_component(x_component[[:v]], ())
    logpdf_cubature1 = mld_cubature1(x[iv], ())
    logpdf_cubature2 = mld_cubature2(x[iv], ())

    @test logpdf_laplace ≈ logpdf_true
    @test logpdf_laplace ≈ logpdf_laplace_component
    @test logpdf_cubature1 ≈ logpdf_true
    @test logpdf_cubature2 ≈ logpdf_true
    # test against numerical integral
    int, err = hcubature(w -> exp(ld([w[1], x[only(iv)], w[2]], ())), lb, ub)
    @test log(int) ≈ logpdf_laplace
    @test log(int) ≈ logpdf_cubature1
    @test log(int) ≈ logpdf_cubature2
    # # marginalized density should be higher than joint density at same point
    @test logpdf_laplace >= mld_laplace.logdensity(x, ())
    @test logpdf_cubature1 >= mld_cubature1.logdensity(x, ())
    @test logpdf_cubature2 >= mld_cubature2.logdensity(x, ())

    # test for correct type promotion
    v_int = ones(Int, njoint(mld_laplace))
    v_float = ones(Float64, njoint(mld_laplace))
    @test mld_laplace(v_int) == mld_laplace(v_float)
end


@testset "AD types" begin

    adtypes = [
        AutoForwardDiff(), 
        AutoReverseDiff(),
        AutoZygote(),
        # AutoMooncake(config=nothing)
    ]
    solvers = [NelderMead, LBFGS, BFGS]

    u = randn(rng, N)
    v = u[iv]
    w = u[iw]
    u_component = ComponentArray(v = v, w = w)

    marginalizer = LaplaceApprox(NelderMead(); adtype=AutoForwardDiff())
    mld = MarginalLogDensity(ld, u, iw, (), marginalizer)
    mld_component = MarginalLogDensity(ld, u_component, [:w], (), marginalizer)
    L0 = mld(v, ())
    @test L0 ≈ mld_component(u_component[[:v]])

    results = []
    for adtype in adtypes
        for solver in solvers
            marginalizer = LaplaceApprox(solver(), adtype=adtype)
            mld = MarginalLogDensity(ld, u, iw, (), marginalizer)
            mld_component = MarginalLogDensity(ld, u_component, [:w], (), marginalizer)
            @test L0 ≈ mld(v, ())
            @test L0 ≈ mld_component(u_component[[:v]])
            t0 = time()
                for i in 1:100 
                    mld(v, ())
                end
            t = round(Int, (time() - t0) / 100 * 1e6)
            push!(results, (; adtype, solver, t))
        end
    end
    idx = sortperm([x.t for x in results])
    for i in idx
        adtype, solver, t = results[i]
        println("AD: $(adtype),\tSolver: $(solver),\tTime: $t μs")
    end
end

@testset "Sparse LaplaceApprox" begin
    N = 100
    μ = ones(N)
    σ = 1.5
    d = MvNormal(μ, σ^2 * I)
    ld(u, p) = logpdf(MvNormal(p.μ, p.σ * I), u)
    iv = 50:60
    iw = setdiff(1:N, iv)
    u = randn(rng, N)
    v = u[iv]
    w = u[iw]
    p = (;μ, σ)

    mldd = MarginalLogDensity(ld, u, iw, p, LaplaceApprox(),
        hess_adtype=AutoZygote())

    mlds = MarginalLogDensity(ld, u, iw, p, LaplaceApprox(),
        hess_adtype=AutoSparse(
            SecondOrder(AutoForwardDiff(), AutoZygote()),
            DenseSparsityDetector(AutoZygote(), atol=1e-9),
            GreedyColoringAlgorithm()
        )
    )
    @test issparse(cached_hessian(mlds))
    @test ! issparse(cached_hessian(mldd))
    @test mlds(v, p) ≈ mldd(v, p)
    @test all(Matrix(cached_hessian(mlds)) .≈ cached_hessian(mldd))
end

@testset "Outer Optimization" begin
    ncategories = 8
    categories = 1:ncategories
    μ0 = 5.0
    σ0 = 5.0
    aa = rand(rng, Normal(μ0, σ0), ncategories)
    b = 4.5
    σ = 0.5
    category = repeat(categories, inner=200)
    n = length(category)
    x = rand(rng, Uniform(-1, 1), n)
    μ = [aa[category[i]] + b * x[i] for i in 1:n]
    y = rand.(Normal.(μ, σ))
        
    function loglik(θ::Vector{T}, p) where T
        μ0 = θ[1]
        σ0 = exp(θ[2])
        aa = θ[3:10]
        b = θ[11]
        σ = exp(θ[12])
        μ = [aa[p.category[i]] + b * p.x[i] for i in 1:p.n]
        return loglikelihood(Normal(μ0, σ0), aa) + sum(logpdf.(Normal.(μ, σ), p.y))
    end
    
    θtrue = [μ0; log(σ0); aa; b; log(σ)]
    p = (; category, x, y, n)
    nθ = length(θtrue)
    
    θ0 = ones(length(θtrue))
    θmarg = θ0[[1, 2, 11, 12]]
    mld_laplace = MarginalLogDensity(loglik, θ0, collect(3:10), p, LaplaceApprox())
    # mld_cubature = MarginalLogDensity(loglik, θ0, collect(3:10), p,
    #     Cubature(lower=fill(-5.0, 8), upper=fill(5, 8)))

    opt_func = OptimizationFunction(mld_laplace, AutoFiniteDiff())
    v0 = ones(length(θmarg))
    opt_prob1 = OptimizationProblem(opt_func, v0, p)
    opt_prob2 = OptimizationProblem(mld_laplace, v0) 
    opt_sol1 = solve(opt_prob1, NelderMead())
    opt_sol2 = solve(opt_prob2, NelderMead())
    @test all(isapprox.(opt_sol1.u, opt_sol2.u))

    opt_sol1_1 = solve(opt_prob1, LBFGS())
    @test all(isapprox.(opt_sol1.u, opt_sol1_1.u, rtol=0.01))

    # opt_prob3 = OptimizationProblem(mld_cubature, v0)
    # opt_sol3 = solve(opt_prob3, NelderMead())
    # println(maximum(abs.(opt_sol1.u .- opt_sol3.u)))
    # @test all(isapprox.(opt_sol1.u, opt_sol3.u, atol=0.01))
end