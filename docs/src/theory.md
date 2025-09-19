
```@meta
CurrentModule = MarginalLogDensities
```

## How it works

By default, this package uses Laplace's method to approximate this integral. The Laplace approximation
is fast in high dimensions, and works well for log-densities that are approximately Gaussian. The
marginalization can also be done via numerical integration, a.k.a. cubature, which may be more accurate
but will not scale well for higher dimensional problems. You can choose which marginalizer to use by passing the
appropriate `AbstractMarginalizer` to `MarginalLogDensity`:

```julia
MarginalLogDensity(logdensity, u, iw, data, Cubature())
MarginalLogDensity(logdensity, u, iw, data, LaplaceApprox())
```

Both `Cubature()` and `LaplaceApprox()` can be specified with various options; refer to their 
docstrings for details.

