using JITrench
using Documenter

DocMeta.setdocmeta!(JITrench, :DocTestSetup, :(using JITrench); recursive=true)

makedocs(;
    modules=[JITrench],
    authors="Yuchi Yamaguchi",
    repo="https://github.com/abap34/JITrench.jl/blob/{commit}{path}#{line}",
    sitename="JITrench.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://abap34.github.io/JITrench.jl",
        assets=String[]
    ),
    pages=["Home" => "index.md",
        "API" => "api.md",]
)

deploydocs(; repo="github.com/abap34/JITrench.jl")
