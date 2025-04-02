# DiffEqCallbacks.jl: Prebuilt Callbacks for extending the solvers of DifferentialEquations.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/DiffEqCallbacks/stable/)

[![codecov](https://codecov.io/gh/SciML/DiffEqCallbacks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/DiffEqCallbacks.jl)
[![Build Status](https://github.com/SciML/DiffEqCallbacks.jl/workflows/CI/badge.svg)](https://github.com/SciML/DiffEqCallbacks.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) has an expressive callback system
which allows for customizable transformations of the solver behavior. DiffEqCallbacks.jl
is a library of pre-built callbacks which makes it easy to transform the solver into a
domain-specific simulation tool.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/DiffEqCallbacks/stable/). Use the
[in-development documentation](https://docs.sciml.ai/DiffEqCallbacks/dev/) for the version of
the documentation, which contains the unreleased features.

## Manifold Projection Example

Here we solve the harmonic oscillator:

```julia
using OrdinaryDiffEq

u0 = ones(2)
function f(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end
prob = ODEProblem(f, u0, (0.0, 100.0))
```

However, this problem is supposed to conserve energy, and thus we define our manifold
to conserve the sum of squares:

```julia
function g(resid, u, p, t)
    resid[1] = u[2]^2 + u[1]^2 - 2
end
```

To build the callback, we call

```julia
using DiffEqCallbacks, ADTypes
cb = ManifoldProjection(g, autodiff=AutoForwardDiff(), resid_prototype = zeros(1))
```

Using this callback, the Runge-Kutta method `Vern7` conserves energy. Note that the
standard saving occurs after the step and before the callback, and thus we set
`save_everystep=false` to turn off all standard saving and let the callback
save after the projection is applied.

```julia
using Test
sol = solve(prob, Vern7(), save_everystep = false, callback = cb)
@test sol[end][1]^2 + sol[end][2]^2 ≈ 2
```

![manifold_projection](https://user-images.githubusercontent.com/1814174/184501895-38f081b6-3d7a-434c-adca-63b6b36a315c.png)
