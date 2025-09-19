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
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ElOceanografo/MarginalLogDensities.jl.git",
    devbranch="master",
)