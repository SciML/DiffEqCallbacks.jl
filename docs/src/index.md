# DiffEqCallbacks.jl: Prebuilt Callbacks for extending the solvers of DifferentialEquations.jl

[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) has an expressive callback system
which allows for customizable transformations of the solver behavior. DiffEqCallbacks.jl
is a library of pre-built callbacks which makes it easy to transform the solver into a
domain-specific simulation tool.

## Installation

To install DiffEqCallbacks.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("DiffEqCallbacks")
```

## Usage

To use the callbacks provided in this library with solvers, simply pass the callback to
the solver via the `callback` keyword argument:

```julia
sol = solve(prob, alg; callback = cb)
```

For more information on using callbacks,
[see the manual page](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions).

## Note About Dependencies

Note that DiffEqCallbacks.jl is not a required dependency for the callback mechanism.
DiffEqCallbacks.jl is simply a library of pre-made callbacks, not the library which defines
the callback system. Callbacks are defined in the SciML interface at SciMLBase.jl.

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
