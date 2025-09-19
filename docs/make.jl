using MarginalLogDensities
using Documenter

DocMeta.setdocmeta!(MarginalLogDensities, :DocTestSetup, :(using MarginalLogDensities); recursive=true)

makedocs(;
    modules=[MarginalLogDensities],
    authors="Sam Urmy <oceanographerschoice@gmail.com> and contributors",
    repo="https://github.com/ElOceanografo/MarginalLogDensities.jl/blob/{commit}{path}#{line}",
    sitename="MarginalLogDensities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ElOceanografo.github.io/MarginalLogDensities.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Getting started" => "index.md",
        "User guide" => [
            "Theory" => "theory.md",
            "Sparsity and AD" => "sparse_ad.md",
            "Tips for success" => "tips.md"
        ],
        "Examples" => [
            "Hierarchical regression" => "hreg.md",
            "Using ComponenetArrays" => "componentarrays.md",
            "Turing integration" => "turing.md"
        ],
        "API Reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/ElOceanografo/MarginalLogDensities.jl.git",
    devbranch="master",
)