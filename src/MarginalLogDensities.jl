module MarginalLogDensities

using Optim
using ForwardDiff
using LinearAlgebra
using SuiteSparse
using SparseArrays
using HCubature
using Distributions
using SparsityDetection
# using Symbolics
using SparseDiffTools
using NLSolversBase


export MarginalLogDensity,
    HessianConfig,
    AbstractMarginalizer,
    LaplaceApprox,
    Cubature,
    dimension,
    imarginal,
    ijoint,
    nmarginal,
    njoint,
    merge_parameters,
    split_parameters

abstract type AbstractMarginalizer end

struct LaplaceApprox <: AbstractMarginalizer end

struct Cubature{T} <: AbstractMarginalizer
    upper::AbstractVector{T}
    lower::AbstractVector{T}
end

"""
    `MarginalLogDensity(logdensity, n, imarginal, [method=LaplaceApprox()])`

Construct a callable object which wraps the function `logdensity` and
integrates over a subset of its arguments.
* `logdensity` : function taking a vector of `n` parameters and returning a positive
log-probability (e.g. a log-pdf, log-likelihood, or log-posterior).
* `imarginal` : Vector of indices indicating which arguments to `logdensity` to marginalize
* `method` : How to perform the marginalization.  Defaults to `LaplaceApprox()`; `Cubature()`
is also available.

If `length(imarginal) == m`, then the constructed `MarginalLogDensity` object it `mld`
can be called as `mld(θ)`, where `θ` is a vector with length `n-m`.  It can also be called
as `mld(u, θ)`, where `u` is a length-`m` vector of the marginalized variables.  In this
case, the return value is the same as the full conditional `logdensity` with `u` and `θ`
"""
struct MarginalLogDensity{TI<:Integer, TM<:AbstractMarginalizer,
        TV<:AbstractVector{TI}, TF, THP}
    logdensity::TF
    n::TI
    imarginal::TV
    ijoint::TV
    method::TM
    hessconfig::THP
end

function MarginalLogDensity(logdensity, n, im, method=LaplaceApprox(), forwarddiff_sparsity=false)
    ij = setdiff(1:n, im)
    u = zeros(n)
    # hessconfig = HessianConfig(logdensity, im, ij, forwarddiff_sparsity)
    if forwarddiff_sparsity
        println("Detecting Hessian sparsity via ForwardDiff...")
        H = ForwardDiff.hessian(logdensity, u)
        Hsparsity = sparse(H) .!= 0
    else
        println("Detecting Hessian sparsity via SparsityDetection...")
        Hsparsity = hessian_sparsity(logdensity, u)
    end
    Hcolors = matrix_colors(Hsparsity)
    hessconfig = ForwardColorHesCache(logdensity, u, Hcolors, Hsparsity)
    return MarginalLogDensity(logdensity, n, im, ij, method, hessconfig)
end


# Was using this for testing/troubleshooting, will probably delete later
# """
#     `num_hessian_sparsity(f, x, [δ=1.0])`
#
# Calculate the sparsity pattern of the Hessian matrix of function `f`. This is a brute-force
# approach, but more robust than the one in SparsityDetection
# """
# function num_hessian_sparsity(f, x, δ=1.0)
#     N = length(x)
#     g(x) = ForwardDiff.gradient(f, x)
#     y = g(x)
#     ii = Int[]
#     jj = Int[]
#     vv = Float64[]
#     for j in 1:N
#         x[j] += δ
#         yj = g(x)
#         di = findall(.! (yj .≈ y))
#         for i in di
#             push!(jj, j)
#             push!(ii, i)
#             push!(vv, 1.0)
#         end
#         x[j] -= δ
#     end
#     return sparse(ii, jj, vv)
# end



# struct HessianConfig{THS, THC, TI<:Integer, TD, TG}
#     Hsparsity::THS
#     Hcolors::THC
#     ncolors::TI
#     D::TD
#     Hcomp_buffer::TD
#     G::TG
#     δG::TG
# end

# function HessianConfig(logdensity, imarginal, ijoint, forwarddiff_sparsity=false)
#     x = ones(length(imarginal) + length(ijoint))
#     if forwarddiff_sparsity
#         println("Detecting Hessian sparsity via ForwardDiff...")
#         H = ForwardDiff.hessian(logdensity, x)
#         Hsparsity = sparse(H)[imarginal, imarginal] .!= 0
#     else
#         println("Detecting Hessian sparsity via SparsityDetection...")
#         Hsparsity = hessian_sparsity(logdensity, x)[imarginal, imarginal]
#     end
#     Hcolors = matrix_colors(Hsparsity)
#     D = hcat([float.(i .== Hcolors) for i in 1:maximum(Hcolors)]...)
#     Hcomp_buffer = similar(D)
#     G = zeros(length(imarginal))
#     δG = zeros(length(imarginal))
#     return HessianConfig(Hsparsity, Hcolors, size(Hcolors, 2), D, Hcomp_buffer, G, δG)
# end

# function sparse_hessian!(H, f, g!, θ, hessconfig::HessianConfig, δ=sqrt(eps(Float64)))
#     nc = hessconfig.ncolors
#     for j in one(nc):nc
#         g!(hessconfig.G, θ)
#         g!(hessconfig.δG, θ + δ * @view hessconfig.D[:, j])
#         hessconfig.Hcomp_buffer[:, j] .= (hessconfig.δG .- hessconfig.G) ./ δ
#     end
#     ii, jj, vv = findnz(hessconfig.Hsparsity)
#     for (i, j) in zip(ii, jj)
#         H[i, j] = hessconfig.Hcomp_buffer[i, hessconfig.Hcolors[j]]
#     end
# end

# function sparse_hessian(f, g!, θ,  hessconfig::HessianConfig, δ=sqrt(eps(Float64)))
#     i, j, v = findnz(hessconfig.Hsparsity)
#     H = sparse(i, j, zeros(eltype(θ), length(v)))
#     sparse_hessian!(H, f, g!, θ, hessconfig, δ)
#     return H
# end


# function MarginalLogDensity(logdensity::Function, n::TI,
#         imarginal::AbstractVector{TI}; method=LaplaceApprox(), forwarddiff_sparsity=false) where {TI<:Integer}
#     ijoint = setdiff(1:n, imarginal)
#     hessconfig = HessianConfig(logdensity, imarginal, ijoint, forwarddiff_sparsity)
#     mld  = MarginalLogDensity(logdensity, n, imarginal, ijoint, method, hessconfig)
#     return mld
# end

dimension(mld::MarginalLogDensity) = mld.n
imarginal(mld::MarginalLogDensity) = mld.imarginal
ijoint(mld::MarginalLogDensity) = mld.ijoint
nmarginal(mld::MarginalLogDensity) = length(mld.imarginal)
njoint(mld::MarginalLogDensity) = length(mld.ijoint)

function merge_parameters(θmarg::AbstractVector{T1}, θjoint::AbstractVector{T2}, imarg, ijoint) where {T1,T2}
    N = length(θmarg) + length(θjoint)
    θ = Vector{promote_type(T1, T2)}(undef, N)
    θ[imarg] .= θmarg
    θ[ijoint] .= θjoint
    return θ
end

split_parameters(θ, imarg, ijoint) = (θ[imarg], θ[ijoint])

function (mld::MarginalLogDensity)(θmarg::AbstractVector{T1}, θjoint::AbstractVector{T2}) where {T1, T2}
    θ = merge_parameters(θmarg, θjoint, imarginal(mld), ijoint(mld))
    return mld.logdensity(θ)
end

function (mld::MarginalLogDensity)(θjoint::AbstractVector{T}, verbose=false) where T
    integral = _marginalize(mld, θjoint, mld.method, verbose)
    return integral
end

############################################################################################
# This section implements a `logabsdet` method for sparse LU decompositions, so we can
# calculate the Laplace approximation for sparse Hessians.  This is an import from the
# future; the method will be in the next Julia release but it's copy-pasted in here for now
# so that this package will work.

# compute the sign/parity of a permutation
function _signperm(p)
    n = length(p)
    result = 0
    todo = trues(n)
    while any(todo)
        k = findfirst(todo)
        todo[k] = false
        result += 1 # increment element count
        j = p[k]
        while j != k
            result += 1 # increment element count
            todo[j] = false
            j = p[j]
        end
        result += 1 # increment cycle count
    end
    return ifelse(isodd(result), -1, 1)
end

function LinearAlgebra.logabsdet(F::SuiteSparse.UMFPACK.UmfpackLU{T, TI}) where {T<:Union{Float64,ComplexF64}, TI<:Integer} # return log(abs(det)) and sign(det)
    n = LinearAlgebra.checksquare(F)
    issuccess(F) || return log(zero(real(T))), zero(T)
    U = F.U
    Rs = F.Rs
    p = F.p
    q = F.q
    s = _signperm(p)*_signperm(q)*one(real(T))
    P = one(T)
    abs_det = zero(real(T))
    @inbounds for i in 1:n
        dg_ii = U[i, i] / Rs[i]
        P *= sign(dg_ii)
        abs_det += log(abs(dg_ii))
    end
    return abs_det, s * P
end
############################################################################################

function _marginalize(mld::MarginalLogDensity, θjoint::AbstractVector{T},
        method::LaplaceApprox, verbose) where T
    N = nmarginal(mld)
    θmarginal0 = ones(T, N)
    f = (θmarginal) -> -mld(θmarginal, θjoint)
    gconfig = ForwardDiff.GradientConfig(f, θmarginal0)
    g! = (G, x) -> ForwardDiff.gradient!(G, f, x, gconfig)
    hsparsity = mld.hessconfig.sparsity[imarginal(mld), imarginal(mld)]
    hcolors = matrix_colors(hsparsity)
    hescache = ForwardColorHesCache(f, θmarginal0, hcolors, hsparsity, g!)
    # h! = (H, x) -> sparse_hessian!(H, f, g!, x, mld.hessconfig)
    h! = (H, x) -> numauto_color_hessian!(H, f, x, hescache)
    H0 = numauto_color_hessian(f, θmarginal0, hescache)
    td = TwiceDifferentiable(f, g!, h!, θmarginal0, zero(T), zeros(T, N), H0)

    verbose && println("Optimizing...")
    opt = optimize(td, θmarginal0)
    verbose && println("Calculating Hessian at mode...")
    h!(H0, opt.minimizer)
    verbose && println("Laplace approximating...")
    integral = -opt.minimum + 0.5 * (log((2π)^N) - logdet(H0))
    return integral
end

function _marginalize(mld::MarginalLogDensity, θjoint, method::Cubature, verbose)
    f = θmarginal -> exp(mld(θmarginal, θjoint))
    int, err = hcubature(f, method.lower, method.upper)
    return log(int)
end

end # module
