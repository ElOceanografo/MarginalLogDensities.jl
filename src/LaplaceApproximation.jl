module LaplaceApproximation

using Optim
using ForwardDiff
using LinearAlgebra
using Distributions

export LaplaceApprox,
    fitlaplace,
    logapprox,
    approx,
    mode,
    normalizer

struct LaplaceApprox
    logf
    mode
    hessian
    normalizer
end
Distributions.mode(la::LaplaceApprox) = la.mode
normalizer(la::LaplaceApprox) = la.normalizer

function fitlaplace(logf::Function, x0::AbstractVector)
    n = length(x0)
    xfit = optimize(x -> -logf(x), x0)
    xhat = xfit.minimizer
    H = ForwardDiff.hessian(logf, xhat)
    f(x) = exp(logf(x))
    z = f(xhat) * sqrt((2π)^length(xhat) / det(H))
    return LaplaceApprox(logf, xhat, H, z)
end

logapprox(la::LaplaceApprox, x::AbstractVector) = la.logf(x) - log(normalizer(la))
approx(la::LaplaceApprox, x::AbstractVector) = exp(logapprox(la, x))

end # module
