module MarginalLogDensities

using Optim
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using SuiteSparse
using SparseArrays
using HCubature
using Distributions
using SparseDiffTools

export MarginalLogDensity,
    AbstractMarginalizer,
    LaplaceApprox,
    Cubature,
    dimension,
    imarginal,
    ijoint,
    nmarginal,
    njoint

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
        TV<:AbstractVector{TI}, TF}
    logdensity::TF
    n::TI
    imarginal::TV
    ijoint::TV
    method::TM
end

function MarginalLogDensity(logdensity::Function, n::TI,
        imarginal::AbstractVector{TI}, method=LaplaceApprox()) where {TI<:Integer}
    ijoint = setdiff(1:n, imarginal)
    return MarginalLogDensity(logdensity, n, imarginal, ijoint, method)
end

dimension(mld::MarginalLogDensity) = mld.n
imarginal(mld::MarginalLogDensity) = mld.imarginal
ijoint(mld::MarginalLogDensity) = mld.ijoint
nmarginal(mld::MarginalLogDensity) = length(mld.imarginal)
njoint(mld::MarginalLogDensity) = length(mld.ijoint)


function (mld::MarginalLogDensity)(θmarg::AbstractVector{T1}, θjoint::AbstractVector{T2}) where {T1, T2}
    θ = Vector{promote_type(T1, T2)}(undef, dimension(mld))
    θ[imarginal(mld)] .= θmarg
    θ[ijoint(mld)] .= θjoint
    return mld.logdensity(θ)
end

function (mld::MarginalLogDensity)(θjoint::AbstractVector{T}) where T
    integral = _marginalize(mld, θjoint, mld.method)
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
        method::LaplaceApprox) where T
    f(θmarginal) = -mld(θmarginal, θjoint)
    # g!(G, θmarginal) = ForwardDiff.gradient!(G, f, θmarginal)
    N = nmarginal(mld)
    opt = optimize(f, zeros(N), LBFGS(), autodiff=:forward)
    # H = ForwardDiff.hessian(f, opt.minimizer)
    Hv = HesVec(f, opt.minimizer)
    H = reduce(hcat, sparse(Hv * i) for i in eachcol(I(N)))
    integral = -opt.minimum + 0.5 * (log((2π)^N) - logdet(H))
    return integral
end

function _marginalize(mld::MarginalLogDensity, θjoint, method::Cubature)
    f = θmarginal -> exp(mld(θmarginal, θjoint))
    int, err = hcubature(f, method.lower, method.upper)
    return log(int)
end

end # module
