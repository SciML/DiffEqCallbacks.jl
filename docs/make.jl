using Documenter, DiffEqCallbacks

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(sitename = "DiffEqCallbacks.jl",
    authors = "Chris Rackauckas",
    modules = [DiffEqCallbacks],
    linkcheck = true,
    linkcheck_ignore = [
        "https://www.sciencedirect.com/science/article/pii/S0096300304009683",
    ],
    strict = [
        :doctest,
        :linkcheck,
        :parse_error,
        :example_block,
        # Other available options are
        # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
    ],
    clean = true, doctest = false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqCallbacks/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/DiffEqCallbacks.jl.git";
    push_preview = true)
