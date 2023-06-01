using MarginalLogDensities
using Test
using Distributions
using Optimization, OptimizationOptimJL
using ForwardDiff, ReverseDiff, Zygote
using LinearAlgebra, SparseArrays
using HCubature
using Random
using SparseDiffTools
using ChainRulesTestUtils

N = 3
μ = ones(N)
σ = 1.5
d = MvNormal(μ, σ^2 * I)
ld(u, p) = logpdf(d, u)
iw = [1, 3]
iv = [2]
dmarginal = Normal(1.0, σ)
u = randn(N)
v = u[iv]
w = u[iw]

@testset "Constructors" begin
    @testset "MarginalLogDensity" begin
        for forwarddiff_sparsity in [false, true]
            mld1 = MarginalLogDensity(ld, u, iw, (), LaplaceApprox())
            mld2 = MarginalLogDensity(ld, u, iw, ())
            mld3 = MarginalLogDensity(ld, u, iw)

            mlds = [mld1, mld2, mld3]
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
    
    @testset "Marginalizers" begin
        adtype = Optimization.AutoForwardDiff()
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
    v = fill(1, 3)
    w = fill(2, 4)
    iv = 1:3
    iw = 4:7
    test_rrule(merge_parameters, v, w, iv, iw)
end

@testset "Sparsity detection" begin
    f(u, p) = dot(u, u) + p.x
    u = ones(5)
    p = (x = 1,)

    # helper to check that hessian for this function is correct (-2 on diag, 0 elsewhere)
    function has_correct_pattern(H)
        ncorrect = 0
        for i in 1:size(H, 1)
            for j in 1:size(H, 2)
                if i == j
                    ncorrect += (H[i, j] != 0)
                else
                    ncorrect += (H[i, j] == 0)
                end
            end
        end
        return ncorrect == size(H, 1)^2
    end

    for autosparsity in [:none, :finitediff, :forwarddiff]#, :sparsitydetection]#, :symbolics]
        hess_prototype = MarginalLogDensities.get_hessian_prototype(f, u, p, autosparsity)
        @test eltype(hess_prototype) == eltype(u)
        @test size(hess_prototype, 1) == size(hess_prototype, 2)
        @test size(hess_prototype, 1) == length(u)
        if autosparsity != :none
            @test has_correct_pattern(hess_prototype)
        end
    end
end

@testset "Dense approximations" begin
    x = 1.0:3.0
    mld_laplace = MarginalLogDensity(ld, u, iw, LaplaceApprox())
    lb = fill(-100.0, 2)
    ub = fill(100.0, 2)
    mld_cubature1 = MarginalLogDensity(ld, u, iw, Cubature(lower=lb, upper=ub))
    mld_cubature2 = MarginalLogDensity(ld, u, iw, Cubature())
    
    @test -mld_laplace.F(x[iw], (p=(), v=x[iv])) == ld(x, ())
    prob = OptimizationProblem(mld_laplace.F, randn(2), (p=(), v=x[iv]))
    sol = solve(prob, BFGS())
    @test all(sol.u .≈ μ[iw])

    # analytical: against 1D Gaussian
    logpdf_true = logpdf(dmarginal, x[only(iv)])
    logpdf_laplace = mld_laplace(x[iv], ())
    logpdf_cubature1 = mld_cubature1(x[iv], ())
    logpdf_cubature2 = mld_cubature2(x[iv], ())

    @test logpdf_laplace  ≈ logpdf_true
    @test logpdf_cubature1  ≈ logpdf_true
    @test logpdf_cubature2  ≈ logpdf_true
    # test against numerical integral
    int, err = hcubature(w -> exp(ld([w[1], x[only(iv)], w[2]], ())), lb, ub)
    @test log(int) ≈ logpdf_laplace
    @test log(int) ≈ logpdf_cubature1
    @test log(int) ≈ logpdf_cubature2
    # # marginalized density should be higher than joint density at same point
    @test logpdf_laplace >= mld_laplace.logdensity(x, ())
    @test logpdf_cubature1 >= mld_cubature1.logdensity(x, ())
    @test logpdf_cubature2 >= mld_cubature2.logdensity(x, ())
end

@testset "Parameters" begin
    Random.seed!(1234)
    ncategories = 8
    categories = 1:ncategories
    μ0 = 5.0
    σ0 = 5.0
    aa = rand(Normal(μ0, σ0), ncategories)
    b = 4.5
    σ = 0.5
    category = repeat(categories, inner=200)
    n = length(category)
    x = rand(Uniform(-1, 1), n)
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
    mld_laplace = MarginalLogDensity(loglik, θ0, collect(3:10), LaplaceApprox())
    mld_cubature = MarginalLogDensity(loglik, θ0, collect(3:10), 
        Cubature(lower=fill(-5.0, 8), upper=fill(5, 8)))

    opt_laplace = optimize(θ -> -mld_laplace(θ, p), ones(4))
    # opt_cubature = optimize(θ -> -mld_cubature(θ, p), ones(4))
    # println(opt_laplace.minimizer)
    # println(opt_cubature.minimizer)
    # @test all(opt_laplace.minimizer .≈ opt_cubature.minimizer)
end

@testset "AD types" begin
    adtypes = [Optimization.AutoFiniteDiff, 
        Optimization.AutoForwardDiff, Optimization.AutoReverseDiff]
        # Optimization.AutoZygote]
    solvers = [NelderMead, LBFGS, BFGS]

    marginalizer = LaplaceApprox(NelderMead(); adtype=SciMLBase.NoAD())
    mld = MarginalLogDensity(ld, u, iw, marginalizer)
    L0 = mld(v, ())
    marginalizer = LaplaceApprox(NelderMead(); adtype=Optimization.AutoForwardDiff())
    mld = MarginalLogDensity(ld, u, iw, marginalizer)
    L1 = mld(v, ())
    @test L0 ≈ L1
    for adtype in adtypes
        for solver in solvers
            println("AD: $(adtype), Solver: $(solver)")
            marginalizer = LaplaceApprox(solver(), adtype=adtype())
            mld = MarginalLogDensity(ld, u, iw, marginalizer)
            @test L0 ≈ mld(v, ())
        end
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
    u = randn(N)
    v = u[iv]
    w = u[iw]
    p = (;μ, σ)

    mldd = MarginalLogDensity(ld, u, iw, p, LaplaceApprox(), hess_autosparse=:none)
    mlds = MarginalLogDensity(ld, u, iw, p, LaplaceApprox(), hess_autosparse=:forwarddiff)
    @test issparse(cached_hessian(mlds))
    @test ! issparse(cached_hessian(mldd))
    @test mlds(v, p) ≈ mldd(v, p)
    @test all(Matrix(cached_hessian(mlds)) .≈ cached_hessian(mldd))
end