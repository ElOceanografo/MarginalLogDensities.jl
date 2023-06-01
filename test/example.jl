using MarginalLogDensities
using Distributions
using StatsPlots
using Random
using Optim
using BenchmarkTools

Random.seed!(123)
ncategories = 40
categories = 1:ncategories
μ0 = 5.0
σ0 = 5.0
aa = rand(Normal(μ0, σ0), ncategories)
b = 4.5
σ = 1.5
category = repeat(categories, inner=10)
n = length(category)
x = rand(Uniform(-1, 1), n)
μ = [aa[category[i]] + b * x[i] for i in 1:n]
y = rand.(Normal.(μ, σ))

scatter(x, y, color=category, label="")

function loglik(u::Vector{T}, p) where T
    μ0 = u[1]
    σ0 = exp(u[2])
    σ = exp(u[3])
    b = u[4]
    aa = u[5:end]
    μ = [aa[p.category[i]] + b * p.x[i] for i in 1:p.n]
    return loglikelihood(Normal(μ0, σ0), aa) + sum(logpdf.(Normal.(μ, σ), p.y))
end

utrue = [μ0; log(σ0); b; log(σ); aa]
p = (; category, x, y, n)
nu = length(utrue)
@code_warntype loglik(utrue, p)

u0 = ones(length(utrue))
iθ = 1:4
ix = 5:length(u0)
θ0 = u0[iθ]
mld = MarginalLogDensity(loglik, u0, ix, p, LaplaceApprox(), hess_autosparse=:forwarddiff)

@code_warntype mld(θ0, p)
@btime mld($θ0, $p) # 
@profview for i in 1:10_000
    mld(θ0, p)
end

opt = optimize(θ -> -mld(θ, p), ones(4))
μ0_opt, logσ0_opt, logσ_opt, b_opt = opt.minimizer

                    # should be:
μ0_opt              # 5.0
exp(logσ0_opt)      # 5.0
b_opt               # 4.5
exp(logσ_opt)       # 1.5

θ_opt = [μ0_opt, exp(logσ0_opt), b_opt, exp(logσ_opt)]

using FiniteDiff, LinearAlgebra
H = FiniteDiff.finite_difference_hessian(θjoint -> -mld(θjoint, p), opt.minimizer)
std_errors = 1 ./ sqrt.(diag(H))

θ_names = ["μ₀", "σ₀", "b", "σ"]
scatter(θ_names, θ_opt, yerror=2*std_errors, label="Estimate",
    xlabel="Parameter", ylabel="Value")
scatter!(θ_names, [μ0, σ0, b, σ], label="Truth")
