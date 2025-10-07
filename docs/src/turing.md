
```@meta
CurrentModule = MarginalLogDensities
```

# Turing.jl integration

Starting with Turing [v0.40.4](https://github.com/TuringLang/Turing.jl/releases/tag/v0.40.4)
 / DynamicPPL [v0.37.4](https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.37.4),
it is possible to construct a `MarginalLogDensity` directly from a Turing model, 
specifying the variables to marginalize by their names.

