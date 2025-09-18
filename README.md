# MarginalLogDensities.jl

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/ElOceanografo/MarginalLogDensities.jl)


[![Build Status](https://github.com/ElOceanografo/MarginalLogDensities.jl/workflows/CI/badge.svg)](https://github.com/ElOceanografo/MarginalLogDensities.jl/actions?query=workflow%CI+branch%3Amaster)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ElOceanografo.github.io/MarginalLogDensities.jl/dev/)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ElOceanografo.github.io/MarginalLogDensities.jl/stable/)


This package implements tools for integrating out (marginalizing) parameters
from log-probability functions, such as likelihoods and posteriors of statistical
models. The functionality is similar to that of the TMB (Template Model Builder) package
for R, but implemented in pure Julia.

This approach is useful for fitting models that incorporate random effects or 
latent variables that are important for structuring the model, but whose exact 
values may not be of interest in and of themselves. In the language of regression, 
these are known as "random effects," and in other contexts, they may be called "nuisance
parameters." Whatever they are called, we hope that our log-density function can be
optimized faster and/or more reliably if we focus on the parameters we
are *actually* interested in, averaging the log-density over all possible values of 
the ones we are not. Read more in the [docs](https://ElOceanografo.github.io/MarginalLogDensities.jl/stable/).
   