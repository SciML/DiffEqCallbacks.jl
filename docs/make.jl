using Documenter, DiffEqCallbacks

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(
    sitename = "DiffEqCallbacks.jl",
    authors = "Chris Rackauckas",
    modules = [DiffEqCallbacks],
    clean = true, doctest = false, linkcheck = true,
    linkcheck_ignore = [
        "https://www.sciencedirect.com/science/article/pii/S0096300304009683",
    ],
    warnonly = [:missing_docs],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqCallbacks/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/DiffEqCallbacks.jl.git";
    push_preview = true
)
