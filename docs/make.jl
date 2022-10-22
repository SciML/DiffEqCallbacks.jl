using Documenter, DiffEqCallbacks

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(sitename = "DiffEqCallbacks.jl",
         authors = "Chris Rackauckas",
         modules = [DiffEqCallbacks],
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
