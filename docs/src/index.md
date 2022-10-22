# DiffEqCallbacks.jl: Prebuilt Callbacks for extending the solvers of DifferentialEquations.jl

[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) has an expressive callback system
which allows for customizable transformations of te solver behavior. DiffEqCallbacks.jl
is a library of pre-built callbacks which makes it easy to transform the solver into a
domain-specific simulation tool.

## Installation

To install DiffEqCallbacks.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("DiffEqCallbacks")
```

## Usage

To use the callbacks provided in this library with  solvers, simply pass the callback to
the solver via the `callback` keyword argument:

```julia
sol = solve(prob,alg;callback=cb)
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
- There are a few community forums:
    - the #diffeq-bridged channel in the [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - on the [Julia Discourse forums](https://discourse.julialang.org)
    - see also [SciML Community page](https://sciml.ai/community/)
